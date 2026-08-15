#!/usr/bin/env node
// Stamp a release version (and optionally retarget the npm scope) across the
// meta package, the per-platform packages, and the JS launchers that embed
// the package scope at runtime.
//
// Usage:
//   node scripts/prepare-publish.js --version <semver> [--scope <@scope>]
//
// What it touches:
//   - grok/package.json:                name, version, optionalDependencies pins
//   - grok-<platform>-<arch>/package.json: name, version, description
//   - grok/bin/grok, grok/bin/postinstall.js: literal "@xai-official" strings
//
// The launchers resolve the per-platform package by name at runtime, so a
// scope change must rewrite them too, not just the package.json files.
// Dirties the working tree by design — run on a throwaway checkout in CI.

const fs = require('fs');
const path = require('path');

const UPSTREAM_SCOPE = '@xai-official';

const npmRoot = path.resolve(__dirname, '..', '..');
const metaDir = path.resolve(__dirname, '..');

function parseArgs() {
    const args = { version: undefined, scope: undefined };
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--version') {
            args.version = argv[++i];
        } else if (argv[i] === '--scope') {
            args.scope = argv[++i];
        } else {
            console.error(`Unknown option: ${argv[i]}`);
            process.exit(1);
        }
    }
    if (!args.version || !/^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/.test(args.version)) {
        console.error(`--version must be semver like 1.0.3 or 1.0.3-alpha.1, got: ${args.version}`);
        process.exit(1);
    }
    if (args.scope !== undefined && !/^@[a-z0-9][a-z0-9._-]*$/i.test(args.scope)) {
        console.error(`--scope must look like @my-scope, got: ${args.scope}`);
        process.exit(1);
    }
    return args;
}

function readJson(p) {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function writeJson(p, obj) {
    fs.writeFileSync(p, JSON.stringify(obj, null, 4) + '\n');
}

function retarget(value, scope) {
    return typeof value === 'string' ? value.split(UPSTREAM_SCOPE).join(scope) : value;
}

function main() {
    const { version, scope } = parseArgs();

    // Meta package: name, version, and the pinned optionalDependencies.
    const metaPkgPath = path.join(metaDir, 'package.json');
    const metaPkg = readJson(metaPkgPath);
    if (scope) metaPkg.name = retarget(metaPkg.name, scope);
    metaPkg.version = version;
    if (metaPkg.optionalDependencies) {
        const pinned = {};
        for (const dep of Object.keys(metaPkg.optionalDependencies)) {
            pinned[scope ? retarget(dep, scope) : dep] = version;
        }
        metaPkg.optionalDependencies = pinned;
    }
    writeJson(metaPkgPath, metaPkg);
    console.log(`[prepare] ${metaPkg.name}@${version}`);

    // Per-platform packages: name, version, description.
    const platformDirs = fs.readdirSync(npmRoot)
        .filter(d => /^grok-.+/.test(d) && fs.existsSync(path.join(npmRoot, d, 'package.json')))
        .sort();
    for (const dir of platformDirs) {
        const pkgPath = path.join(npmRoot, dir, 'package.json');
        const pkg = readJson(pkgPath);
        if (scope) {
            pkg.name = retarget(pkg.name, scope);
            if (pkg.description) pkg.description = retarget(pkg.description, scope);
        }
        pkg.version = version;
        writeJson(pkgPath, pkg);
        console.log(`[prepare] ${pkg.name}@${version}`);
    }

    // Launchers embed the scope in require.resolve() calls and messages.
    if (scope) {
        for (const rel of ['bin/grok', 'bin/postinstall.js']) {
            const file = path.join(metaDir, rel);
            const before = fs.readFileSync(file, 'utf8');
            const after = before.split(UPSTREAM_SCOPE).join(scope);
            if (before !== after) {
                fs.writeFileSync(file, after);
                console.log(`[prepare] retargeted scope in ${rel}`);
            }
        }
    }
}

main();
