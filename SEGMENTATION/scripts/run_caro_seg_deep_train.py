'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from caroSegDeepBuildModel.scriptsCaroSeg.train import train
from caroSegDeepBuildModel.scriptsCaroSeg.test import test
import argparse
import importlib

if __name__ == '__main__':

    # -- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    # --- we get parameters
    p = param.setParameters()
    # --- we run the training
    train(p)
    # --- we run the metrics on test set
    test(p, dataset='test')
    # --- we run the metrics on train set
    test(p, dataset='train')
    # --- we run the metrics on validation set
    test(p, dataset='validation')
