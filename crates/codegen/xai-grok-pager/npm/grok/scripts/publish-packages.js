#!/usr/bin/env node
// Publish the assembled per-platform packages, then the meta package.
//
// Usage:
//   node scripts/publish-packages.js [--dry-run]
//
// A platform package counts as assembled when its bin/ holds a brotli payload
// (produced by assemble-platform-packages.js); platforms without one are
// skipped. Platforms publish first so the meta package's optionalDependencies
// always resolve on the registry, and versions already on the registry are
// skipped so reruns are idempotent.

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const npmRoot = path.resolve(__dirname, '..', '..');
const metaDir = path.resolve(__dirname, '..');

const dryRun = process.argv.includes('--dry-run');

function readJson(p) {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function isPackageVersionPublished(name, version) {
    const spec = `${name}@${version}`;
    const result = spawnSync('npm', ['view', spec, 'version', '--json'], { encoding: 'utf8' });
    if (result.status === 0) {
        return true;
    }
    const output = `${result.stderr ?? ''}\n${result.stdout ?? ''}`;
    if (/\bE404\b|404 Not Found/i.test(output)) {
        return false;
    }
    console.error(`Failed to check whether ${spec} is already published.`);
    const detail = output.trim().split('\n')[0];
    if (detail) console.error(detail);
    process.exit(result.status ?? 1);
}

function assembledPlatformDirs() {
    return fs.readdirSync(npmRoot)
        .filter(d => /^grok-.+/.test(d))
        .filter(d => {
            const binDir = path.join(npmRoot, d, 'bin');
            return fs.existsSync(binDir) &&
                fs.readdirSync(binDir).some(f => f.endsWith('.br'));
        })
        .sort()
        .map(d => path.join(npmRoot, d));
}

const platformDirs = assembledPlatformDirs();
if (platformDirs.length === 0) {
    console.error('No assembled platform packages found (bin/*.br missing).');
    console.error('Run assemble-platform-packages.js first.');
    process.exit(1);
}

if (!dryRun) {
    const whoami = spawnSync('npm', ['whoami'], { encoding: 'utf8' });
    if (whoami.status !== 0) {
        console.error('Not authenticated with npm. Run `npm login` or set NODE_AUTH_TOKEN.');
        process.exit(1);
    }
    console.log(`Publishing as ${whoami.stdout.trim()}`);
}

// Platforms first, meta package last.
for (const dir of [...platformDirs, metaDir]) {
    const { name, version } = readJson(path.join(dir, 'package.json'));
    if (!dryRun && isPackageVersionPublished(name, version)) {
        console.log(`${name}@${version} is already published; skipping.`);
        continue;
    }
    console.log(`${dryRun ? '[dry-run] ' : ''}Publishing ${name}@${version}...`);
    const args = ['publish', '--access', 'public', '--ignore-scripts'];
    if (dryRun) args.push('--dry-run');
    const result = spawnSync('npm', args, { cwd: dir, stdio: 'inherit', encoding: 'utf8' });
    if (result.status !== 0) {
        process.exit(result.status ?? 1);
    }
}

console.log(dryRun ? 'Dry run complete.' : 'Publish complete.');
