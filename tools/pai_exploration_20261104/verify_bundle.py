import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.sha256.json').read_text(encoding='utf-8'))
    for rel,digest in manifest['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Bundle verification failed: '+rel)
    print(json.dumps({'status':'verified','files':len(manifest['files'])}))


if __name__=='__main__':main()
