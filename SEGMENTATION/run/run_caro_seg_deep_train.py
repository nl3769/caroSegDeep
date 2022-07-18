'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_cores.train import train
from package_cores.test import test

import argparse
import importlib

if __name__ == '__main__':

    # -- using a parser with set_parameters.py allows us to run several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of package_parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- get package_parameters
    p = param.setParameters()
    
    # --- run the training
    train(p)

    # --- run the metrics on validation set
    test(p, set='validation')
    # --- run the metrics on train set
    test(p, set='training')
    # --- run the metrics on test set
    test(p, set='test')