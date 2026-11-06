import subprocess
from pathlib import Path
from plan_visible_bases import ROOT,read
c=ROOT/'task_workspaces/labelagent_handoff_control_20261104';r=read(c/'visible_base_bundle_verification.json');target=c/'verify_visible_bundle.cjs';target.write_text((c/'verify_visible_base.cjs').read_text(encoding='utf-8').replace('visible_base','visible_bundle'),encoding='utf-8');subprocess.run(['C:/Users/seer/AppData/Local/pallet-annotation-studio/node-v24.19.0-win-x64/node.exe',str(target),str(Path(r['bundle'])/'01_主报告与全部试跑')],check=True)
