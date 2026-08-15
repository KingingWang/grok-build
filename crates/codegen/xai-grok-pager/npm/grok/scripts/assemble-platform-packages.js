#!/usr/bin/env node
// Assemble the six per-platform npm packages prior to `npm publish`.
//
// For each supported (platform, arch) target this:
//   1. Brotli-compresses the built binary into `../grok-<platform>/bin/<bin>.br`
//   2. Stamps the sub-package's version to match the meta package
//
// Each per-platform package is its own npm publish target. The meta package
// (`@xai-official/grok`) lists all six as `optionalDependencies` pinned to
// the same version; npm installs only the one matching the host's
// `os` + `cpu` filters.
//
// Why brotli? npm's tarball ceiling is ~200 MB and the raw grok binary is
// 100–150 MB per platform. Brotli at max quality cuts that to 30–40 MB,
// leaves plenty of headroom for binary growth, and is decoded by Node's
// built-in zlib.brotliDecompressSync (no native deps required).
//
// Source paths come from environment variables (set in CI) and fall back to
// the default cargo target dirs for local testing.
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const zlib = require('zlib');

const brotliCompress = promisify(zlib.brotliCompress);

const xaiRoot = process.env.XAI_ROOT || path.resolve(__dirname, '..', '..', '..', '..', '..');
const npmRoot = path.resolve(__dirname, '..', '..');

const NOTICES_SOURCE = path.resolve(
    npmRoot, '..', '..', 'xai-grok-tools', 'THIRD_PARTY_NOTICES.md');
const NOTICES_NAME = 'THIRD_PARTY_NOTICES.md';

const META_PKG_JSON = path.resolve(__dirname, '..', 'package.json');
const meta = JSON.parse(fs.readFileSync(META_PKG_JSON, 'utf8'));
const VERSION = meta.version;

// Optional allowlist of "<platform>-<arch>" targets to assemble, e.g.
// GROK_PLATFORMS="darwin-arm64,darwin-x64,linux-x64,linux-arm64" when the
// release only ships a subset of the six upstream targets. Unset = all six.
const ONLY = (process.env.GROK_PLATFORMS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);

function ensureDir(p) { fs.mkdirSync(path.dirname(p), { recursive: true }); }

async function packPlatform({ platform, arch, envVar, defaultSource, binName }) {
    const pkgDir = path.join(npmRoot, `grok-${platform}-${arch}`);
    const pkgJsonPath = path.join(pkgDir, 'package.json');

    if (!fs.existsSync(pkgJsonPath)) {
        console.error(`[assemble] Missing per-platform package at ${pkgDir}`);
        return false;
    }

    const source = process.env[envVar] || defaultSource;
    if (!fs.existsSync(source)) {
        console.error(`[assemble] Missing binary for ${platform}-${arch}: ${source}`);
        console.error(`            Set ${envVar} or build to the default location.`);
        return false;
    }

    // Stamp the sub-package's version to match the meta package.
    const subPkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
    subPkg.version = VERSION;
    fs.writeFileSync(pkgJsonPath, JSON.stringify(subPkg, null, 4) + '\n');

    if (!fs.existsSync(NOTICES_SOURCE)) {
        console.error(`[assemble] Missing third-party notices file: ${NOTICES_SOURCE}`);
        return false;
    }
    fs.copyFileSync(NOTICES_SOURCE, path.join(pkgDir, NOTICES_NAME));

    // Brotli-compress into the sub-package's bin/.
    const outBr = path.join(pkgDir, 'bin', `${binName}.br`);
    ensureDir(outBr);
    const raw = fs.readFileSync(source);
    const compressed = await brotliCompress(raw, {
        params: { [zlib.constants.BROTLI_PARAM_QUALITY]: zlib.constants.BROTLI_MAX_QUALITY },
    });
    fs.writeFileSync(outBr, compressed);
    console.log(
        `[assemble] grok-${platform}-${arch}@${VERSION}: ` +
        `${(raw.length / 1048576).toFixed(1)} MB -> ${(compressed.length / 1048576).toFixed(1)} MB ` +
        `(${path.relative(npmRoot, outBr)})`
    );
    return true;
}

async function main() {
    const targets = [
        {
            platform: 'darwin', arch: 'arm64', binName: 'grok',
            envVar: 'GROK_DARWIN_ARM64',
            defaultSource: path.join(xaiRoot, 'target', 'release', 'xai-grok-pager'),
        },
        {
            platform: 'darwin', arch: 'x64', binName: 'grok',
            envVar: 'GROK_DARWIN_X64',
            defaultSource: path.join(xaiRoot, 'target', 'x86_64-apple-darwin', 'release', 'xai-grok-pager'),
        },
        {
            platform: 'linux', arch: 'x64', binName: 'grok',
            envVar: 'GROK_LINUX_X64',
            defaultSource: path.join(xaiRoot, 'target',
                'explorer_cross_x86_64-unknown-linux-gnu',
                'x86_64-unknown-linux-gnu', 'release', 'xai-grok-pager'),
        },
        {
            platform: 'linux', arch: 'arm64', binName: 'grok',
            envVar: 'GROK_LINUX_ARM64',
            defaultSource: path.join(xaiRoot, 'target',
                'explorer_cross_aarch64-unknown-linux-gnu',
                'aarch64-unknown-linux-gnu', 'release', 'xai-grok-pager'),
        },
        {
            platform: 'win32', arch: 'x64', binName: 'grok.exe',
            envVar: 'GROK_WIN32_X64',
            defaultSource: path.join(xaiRoot, 'target', 'x86_64-pc-windows-msvc', 'release', 'xai-grok-pager.exe'),
        },
        {
            platform: 'win32', arch: 'arm64', binName: 'grok.exe',
            envVar: 'GROK_WIN32_ARM64',
            defaultSource: path.join(xaiRoot, 'target', 'aarch64-pc-windows-msvc', 'release', 'xai-grok-pager.exe'),
        },
    ];

    const selected = targets.filter(t =>
        ONLY.length === 0 || ONLY.includes(`${t.platform}-${t.arch}`));
    if (selected.length === 0) {
        console.error(`[assemble] GROK_PLATFORMS matched no targets: ${ONLY.join(', ')}`);
        process.exit(1);
    }
    const unknown = ONLY.filter(name => !targets.some(t => `${t.platform}-${t.arch}` === name));
    if (unknown.length > 0) {
        console.error(`[assemble] Unknown GROK_PLATFORMS entries: ${unknown.join(', ')}`);
        process.exit(1);
    }

    // Compress in parallel — brotliCompress runs on the libuv thread pool so
    // calls genuinely overlap (set UV_THREADPOOL_SIZE>=6 in CI for full
    // parallelism; Node's default pool size is 4).
    const results = await Promise.all(selected.map(packPlatform));
    const failed = results.filter(r => !r).length;
    if (failed > 0) {
        console.error(`[assemble] ${failed} target(s) failed.`);
        process.exit(1);
    }

    console.log(`[assemble] ${selected.length} per-platform package(s) assembled at version ${VERSION}.`);
}

main().catch((err) => { console.error(err); process.exit(1); });
