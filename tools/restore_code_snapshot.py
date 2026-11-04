"""Restore a complete source tree from a full ZIP and a verified binary diff."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
def restore(receipt,outdir):
 receipt=Path(receipt).resolve();spec=json.loads(receipt.read_text(encoding='utf-8'));home=receipt.parent
 outdir=Path(outdir).resolve()
 if outdir.exists() and any(outdir.iterdir()):raise ValueError('Restore into an empty directory')
 base=home/spec['base_zip'];patch=home/spec['patch']
 for path,expected in [(base,spec['base_zip_sha256']),(patch,spec['patch_sha256'])]:
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Backup digest mismatch')
 outdir.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(base) as z:
  if any(not (outdir/info.filename).resolve().is_relative_to(outdir) for info in z.infolist()):raise ValueError('Unsafe ZIP path')
  z.extractall(outdir)
 if patch.read_text(encoding='utf-8').strip():subprocess.run(['git','apply','--binary',str(patch)],cwd=outdir,check=True)
 normalized=0
 for name,expected in spec['git_blobs'].items():
  path=outdir/name
  if not path.resolve().is_relative_to(outdir):raise ValueError('Unsafe target path')
  content=path.read_bytes();actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
  if actual!=expected:
   # Windows git archive may apply checkout CRLF conversion. Compare the
   # canonical Git text bytes, without ever changing binary data or files.
   canonical=content.replace(b'\r\n',b'\n') if b'\0' not in content[:8000] else content
   actual=hashlib.sha1(b'blob '+str(len(canonical)).encode()+b'\0'+canonical).hexdigest()
   if actual!=expected:raise ValueError('Restored file differs: '+name)
   normalized+=1
 print(json.dumps({'restored':spec['target_commit'],'files_verified':len(spec['git_blobs']),'text_CRLF_canonicalized_for_hash_only':normalized,'passed':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--outdir',required=True);args=p.parse_args();restore(args.receipt,args.outdir)
