from pathlib import Path
helper=Path(__file__).with_name('summarize_projection_results.py')
src=helper.read_text(encoding='utf-8').replace('labelagent_projection_retained_comparison_20261105','labelagent_projection_wire_comparison_20261105').replace("'config_comparison.json'","'config_wire_comparison.json'").replace("'instance_generation_blue/selection.json'","'instance_generation_wire_exact/selection.json'")
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
