"""Commit exactly the checked public code and aggregate boundary addendum."""
import hashlib
import json
import re
import subprocess
from publish_raw_mask_audit import REPO, CONTROL, read


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, encoding='utf-8').strip()


def main():
    files = read(CONTROL / 'raw_mask_publication_files.json')
    assert not git('diff', '--cached', '--name-only')
    for name in files['paths']:
        path = (REPO / name).resolve()
        assert path.is_relative_to(REPO.resolve()) and path.suffix in ('.py', '.json', '.md')
        assert path.stat().st_size < 2 * 1024 ** 2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == files['sha256'][name]
        assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', path.read_text(encoding='utf-8'))
    subprocess.run(['git', '-C', str(REPO), 'add', '--', *files['paths']], check=True)
    subprocess.run(['git', '-C', str(REPO), 'diff', '--cached', '--check'], check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files['paths'])
    body = CONTROL / 'raw_mask_commit_message.txt'
    body.write_text('Audit SAM internal masks separately from returned polygons\n\nReproduce component filtering and contour simplification with explicit CPU stubs. Preserve the historical wire-mapping proof while documenting its limits. Add opt-in private-module capture with failure receipts; do not change the running service, annotations, acceptance limits or incumbent.\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(REPO), 'commit', '-F', str(body)], check=True)
    print(json.dumps({'commit': git('rev-parse', 'HEAD'), 'files': len(files['paths'])}), flush=True)


if __name__ == '__main__':
    main()
