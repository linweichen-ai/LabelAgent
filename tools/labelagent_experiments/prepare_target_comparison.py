"""Bind the latest whole proposals to the original three-round comparison."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
src=(helpers/'prepare_instance_comparison.py').read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_target_identity_20261106').replace('labelagent_instance_separation_comparison_20261105','labelagent_target_identity_comparison_20261106').replace("new=parent/'instance_generation_consistency'","new=parent/'instance_generation'").replace('review_instance_consistency.py','review_target_identity.py').replace("'new_fixed_model_grid_forwards':16","'new_fixed_model_grid_forwards':0")
src=src.replace("runner=runner.replace(\"'diagnostic_only_mask2former_forwards':0\",\"'diagnostic_only_mask2former_forwards':16\")","runner=runner.replace(\"'diagnostic_only_mask2former_forwards':0\",\"'diagnostic_only_mask2former_forwards':0\")")
src=src.replace("'instance_consistency_sha256':sha(Path(__file__).with_name('instance_identity_consistency.py')),", "'instance_consistency_sha256':sha(Path(__file__).with_name('instance_identity_consistency.py')),\\n            'target_input_policy_sha256':sha(Path(__file__).with_name('target_identity_policy.py')),")
src=src.replace("(helpers/'run_instance_comparison.py')", "(helpers/'run_target_identity_comparison.py')")
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
