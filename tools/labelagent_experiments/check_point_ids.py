"""Run inherited strict guards plus ID binding and zero-output integration."""
from pathlib import Path
p=Path(__file__).with_name('check_instance_regressions.py')
s=p.read_text(encoding='utf-8').replace("'test_instance_separation']","'test_instance_separation','test_target_identity','test_visible_base','test_native_resume','test_point_id_policy','test_point_ids_zero_generation']").replace('instance_separation_regression','point_ids_regression').replace('tests/test_instance_separation.py','tests/test_point_ids_zero_generation.py')
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
