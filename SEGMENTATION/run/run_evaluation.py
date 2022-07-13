import argparse
import importlib
from package_handler.evaluation import evaluationClassIMC, evaluationClassFW

if __name__ == '__main__':
    # -- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of package_parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- far wall evaluation (outliers' detection)
    evaluationFW=evaluationClassFW(p)
    evaluationFW.compute_MAE_FW()

    # --- Intima media complexe evaluation
    evaluationIMC=evaluationClassIMC(p)
    evaluationIMC.compute_MAE()
    evaluationIMC.compute_DICE()