import os

def chek_dir(path: str):

    try:
        os.rmdir(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)