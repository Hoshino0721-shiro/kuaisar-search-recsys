"""Check the frozen release and separately recorded documentation revisions."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


DEFAULT_MANIFEST = 'experiments/documentation_20261023/manifest.json'
FROZEN_RELEASE = 'experiments/closure_20261022/release.json'
SNAPSHOT_ROOT = 'experiments/documentation_20261023/frozen/'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Expected a repository-relative POSIX path')
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError('Path outside repository: '+name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside repository: '+name)
    return path


def is_document(name):
    return name == 'README.md' or (name.startswith('docs/') and name.endswith('.md'))


def verify(root, manifest_name=DEFAULT_MANIFEST):
    root = Path(root).resolve()
    manifest = json.loads(safe_path(root, manifest_name).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('scope') != 'documentation-only revision':
        raise ValueError('Unsupported documentation manifest')
    if manifest['frozen_release'] != FROZEN_RELEASE:
        raise ValueError('Unexpected frozen release')
    release_path = safe_path(root, FROZEN_RELEASE)
    if digest(release_path) != manifest['frozen_release_sha256']:
        raise ValueError('Original release manifest changed')
    release = json.loads(release_path.read_text(encoding='utf-8'))
    if release.get('status') != 'frozen' or release.get('schema_version') != 1:
        raise ValueError('Unsupported frozen release')
    documents = manifest['documents']
    for name, entry in documents.items():
        safe_path(root, name)
        if name not in release['public_assets'] or not is_document(name):
            raise ValueError('Only frozen Markdown documents may be revised: '+name)
        if not entry['snapshot'].startswith(SNAPSHOT_ROOT):
            raise ValueError('Snapshot must be in the documentation archive')
        if digest(safe_path(root, entry['snapshot'])) != release['public_assets'][name]:
            raise ValueError('Original document snapshot changed: '+name)
        if entry['frozen_sha256'] != release['public_assets'][name]:
            raise ValueError('Snapshot declaration differs from release: '+name)
    for name, expected in release['public_assets'].items():
        wanted = documents[name]['current_sha256'] if name in documents else expected
        if digest(safe_path(root, name)) != wanted:
            raise ValueError('Published file changed: '+name)
    for name, expected in manifest['additional_files'].items():
        if name in release['public_assets']:
            raise ValueError('Additional files cannot override a frozen asset: '+name)
        if digest(safe_path(root, name)) != expected:
            raise ValueError('Additional file changed: '+name)
    return {'status': 'passed', 'frozen_public_assets': len(release['public_assets']),
            'revised_documents': len(documents), 'additional_files': len(manifest['additional_files']),
            'training_run': False, 'test_labels_read': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest), indent=2))
