"""Run inherited Agent, metric and geometry guards plus exact coordinate cases."""
from pathlib import Path
p=Path(__file__).with_name('check_instance_regressions.py')
s=p.read_text(encoding='utf-8').replace("'test_instance_separation']","'test_instance_separation','test_target_identity','test_visible_base','test_native_resume']").replace('instance_separation_regression','native_resume_regression').replace('tests/test_instance_separation.py','tests/test_native_resume.py')
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
