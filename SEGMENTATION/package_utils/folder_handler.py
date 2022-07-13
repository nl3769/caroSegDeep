import os

# ----------------------------------------------------------------------------------------------------------------------
def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory '%s' created successfully" % path)
    except OSError as error:
        print("Directory '%s' can not be created" % path)

# ----------------------------------------------------------------------------------------------------------------------