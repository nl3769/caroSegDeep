'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_dataset.datasetBuilderCUBS import datasetBuilderIMC, datasetBuilderFarWall
import argparse
import importlib

if __name__ == "__main__":

    # --- using a parser with set_parameters.py allows us to run several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of package_parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- we get package_parameters
    p = param.setParameters()

    # --- we create the datasetBuilderIMC object to create the dataset used to train IMC segmentation model
    dataSetIMC = datasetBuilderIMC(p=p)
    dataSetIMC.build_data()
    del dataSetIMC

    # --- we create the datasetBuilderFarWall object to create the dataset used to train FW segmentation model
    dataSetFW = datasetBuilderFarWall(p=p)
    dataSetFW.build_data()
    del dataSetFW