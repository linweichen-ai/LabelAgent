"""Use the identical neutral/SAM runner with a frozen exact cache catalogue."""
import os,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_base_material as runner
OUT=ROOT/'outputs/labelagent_base_blue_surface_20261108'

def main():
    plan=OUT/'blue_execution_plan.json';catalogue=OUT/'exact_cache_catalogue/completed.json'
    if not plan.exists():
        pair=Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])/'completed.json';material=ROOT/'outputs/labelagent_base_material_20261108/review_completed.json';rows=read(pair)['rows']+read(material)['rows'];save(catalogue,{'rows':rows,'input_files_sha256':{str(pair):sha(pair),str(material):sha(material)}});save(plan,{'wrapper_sha256':sha(__file__),'runner_sha256':sha(runner.__file__),'frozen_source_plan_sha256':sha(OUT/'frozen_plan.json'),'cache_catalogue_sha256':sha(catalogue),'frozen_before_new_requests':True})
    frozen=read(plan);assert frozen['wrapper_sha256']==sha(__file__) and frozen['runner_sha256']==sha(runner.__file__) and frozen['frozen_source_plan_sha256']==sha(OUT/'frozen_plan.json') and frozen['cache_catalogue_sha256']==sha(catalogue);runner.OUT=OUT;runner.PAIR=catalogue.parent
    {'review':runner.review,'prepare_SAM':runner.prepare_SAM,'run_SAM':runner.run_SAM}[sys.argv[1]]()

if __name__=='__main__':main()
