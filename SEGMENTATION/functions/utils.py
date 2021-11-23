import numpy as np
import os

def getDatabase(name, path):
    databases_n = os.listdir(path)
    patients = {}
    for key in databases_n:
        patients[key] = os.listdir(os.path.join(path, key))

        for k in range(len(patients[key])):
            patients[key][k] = patients[key][k].split('.')[0]

    for db in patients.keys():
        if name in patients[db]:
            return db
            break