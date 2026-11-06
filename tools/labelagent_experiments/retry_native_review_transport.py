"""Preserve busy-only failure, then retry missing transport without new rounds."""
from pathlib import Path
import shutil,sys
from native_resume_common import *
source=Path(__file__).with_name('review_native_resume.py');dest=OUT/'instance_generation'
assert not (dest/'completed.json').exists()
traces=read(dest/'model_trace.json');assert len(traces)==1 and traces[0].get('status')=='failed' and traces[0].get('error')=='Shared model busy' and not traces[0].get('output')
archive=OUT/'transport_attempts/attempt01_shared_busy';assert not archive.exists();shutil.copytree(dest,archive/'instance_generation');shutil.copy2(source,archive/'executed_runner.py')
save(archive/'failure.json',{'status':'failed_shared_model_busy','actual_new_semantic_responses':0,'all_existing_neutral_cache_reused_unchanged':True,'no_new_Agent_rounds':True,'failed_source_sha256':sha(source)})
code=source.read_text(encoding='utf-8');code=code.replace('for _ in range(100):\n   with urllib.request.urlopen','for _ in range(1500):\n   with urllib.request.urlopen',1).replace('time.sleep(3)','time.sleep(.2)')
code=code.replace("except ValueError as e:\n    if 'Another model request' not in str(e):raise", "except ValueError as e:\n    if 'Another model request' not in str(e):raise")
target=source.with_name('review_native_resume_retry.py');target.write_text(code,encoding='utf-8')
save(OUT/'transport_retry_amendment.json',{'reason':'100 three-second shared-service polls all busy; no new semantic answer','retry_only_missing_responses':True,'same_model_images_prompt_schema_and_output_budget':True,'poll_interval_seconds':.2,'poll_attempts_max':1500,'Agent_rounds_added':0,'no_other_service_or_client_stopped':True,'prior_attempt_preserved':str(archive),'executed_retry_source_sha256':sha(target)})
print(str(target),flush=True)
