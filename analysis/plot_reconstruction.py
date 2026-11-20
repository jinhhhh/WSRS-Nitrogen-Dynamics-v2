"""Plot existing frozen reconstruction diagnostics without rerunning model fitting."""
from pathlib import Path
import importlib.util,shutil,sys

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def main():
    reconstruction=ROOT/'reconstruction'
    sys.path.insert(0,str(reconstruction/'code'))
    for sub in ['paper_package','error_ranges']:
        output=reconstruction/'regenerated'/sub
        for folder in ['figures','qa','notes']: (output/folder).mkdir(parents=True,exist_ok=True)
        source=reconstruction/sub/('tables' if sub=='paper_package' else 'data')
        shutil.copytree(source,output/source.name,dirs_exist_ok=True)
    # Figure plotting does not import or refit the predictive models.
    load('paper_figures',reconstruction/'code/paper_figures.py').main()
    load('range_figures',reconstruction/'code/plot_error_ranges.py').main()
if __name__=='__main__':main()
