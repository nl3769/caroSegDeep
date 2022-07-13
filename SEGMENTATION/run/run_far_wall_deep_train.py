'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_cores.train    import train
from package_cores.test     import test
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

    # --- get package_parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of package_parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- get package_parameters
    p = param.setParameters()

    main(p)

