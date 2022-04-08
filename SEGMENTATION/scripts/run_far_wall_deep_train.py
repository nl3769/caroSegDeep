'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from caroSegDeepBuildModel.scriptsCaroSeg.train import train
from caroSegDeepBuildModel.scriptsCaroSeg.test import test
import argparse
import importlib

def main(p):

    # --- run training
    train(p)

    # --- run metrics on test set
    test(p, set=p.PATH_FOLD['testing'])

    # --- run metrics on validation set

    test(p, set=p.PATH_FOLD['validation'])
    #--- run metrics on train set
    test(p, set=p.PATH_FOLD['training'])

if __name__ == '__main__':

    # --- get parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    # --- get parameters
    p = param.setParameters()

    main(p)

