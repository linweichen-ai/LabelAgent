from pathlib import Path
helper=Path(__file__).with_name('audit_point_instance_tradeoff.py')
src=helper.read_text(encoding='utf-8').replace('/point_instance_audit','/append_point_audit').replace('labelagent_semantic_comparison_20261105','labelagent_projection_wire_comparison_20261105').replace('==493','==491').replace('==12','==13').replace('rejected12','append13')
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
