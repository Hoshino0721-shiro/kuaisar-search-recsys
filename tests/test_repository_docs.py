import json
from pathlib import Path
import tempfile
import unittest
from scripts.verify_repository import digest, verify, safe_path


class RepositoryDocsTests(unittest.TestCase):
    def fixture(self, root):
        def write(name, content):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return path
        readme = write('README.md', 'original docs')
        source = write('src/model.py', '# model unchanged\n')
        snapshot = write('experiments/documentation_20261023/frozen/README.md', 'original docs')
        added = write('docs/README.md', '# index')
        release = {'schema_version': 1, 'status': 'frozen',
                   'public_assets': {'README.md': digest(readme), 'src/model.py': digest(source)}}
        release_path = write('experiments/closure_20261022/release.json', json.dumps(release))
        original_digest = digest(readme)
        readme.write_text('edited docs', encoding='utf-8')
        manifest = {'schema_version': 1, 'scope': 'documentation-only revision',
                    'frozen_release': 'experiments/closure_20261022/release.json',
                    'frozen_release_sha256': digest(release_path),
                    'documents': {'README.md': {'snapshot': str(snapshot.relative_to(root)).replace('\\','/'),
                        'frozen_sha256': original_digest, 'current_sha256': digest(readme)}},
                    'additional_files': {'docs/README.md': digest(added)}}
        path = write('experiments/documentation_20261023/manifest.json', json.dumps(manifest))
        return path, manifest

    def test_revised_docs_and_original_code_pass_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            result = verify(root)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['revised_documents'], 1)
            self.assertFalse(result['training_run'])
            self.assertFalse(result['test_labels_read'])

    def test_changes_to_code_snapshots_or_new_docs_are_rejected(self):
        for name in ['src/model.py', 'README.md', 'docs/README.md',
                     'experiments/documentation_20261023/frozen/README.md',
                     'experiments/closure_20261022/release.json']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root)
                with (root/name).open('a',encoding='utf-8') as stream:
                    stream.write('unexpected edit')
                with self.assertRaises(ValueError):
                    verify(root)

    def test_revision_cannot_exempt_model_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, manifest = self.fixture(root)
            manifest['documents']['src/model.py'] = manifest['documents']['README.md']
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Only frozen Markdown'):
                verify(root)

    def test_extra_files_cannot_override_original_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, manifest = self.fixture(root)
            manifest['additional_files']['src/model.py'] = digest(root/'src/model.py')
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'cannot override'):
                verify(root)

    def test_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['../README.md', '/outside/file', 'C:/outside/file', r'docs\..\secret']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(Path(tmp), name)


if __name__ == '__main__':
    unittest.main()
