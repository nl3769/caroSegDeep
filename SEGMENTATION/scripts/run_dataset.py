'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from classes.datasetBuilderCUBS import datasetBuilderWall, datasetBuilderFarWall
import argparse
import importlib

if __name__ == "__main__":
    # --- using a parser with set_parameters.py allows us to run several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])
    # --- we get parameters
    p = param.setParameters()

    # --- we create the datasetBuilderFarWall object to create the dataset used for training a far wall detection model
    dataSetFarWall = datasetBuilderFarWall(p=p)
    # --- we create the dataset
    dataSetFarWall.build_data()

    ############################################################
    ############################################################
    ############################################################

    # --- we create the datasetBuilderWall object to create the dataset used to train a IMC segmentation model
    dataSetWall = datasetBuilderWall(p=p)
    # --- we create the dataset
    dataSetWall.build_data()


