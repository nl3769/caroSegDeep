The main functions are in scripts/. There are five possibilities:

# --- 1
run_far_wall_detection.py
-> Detection of the far wall. Results are saved in.txt file.  The argument of the code is set_parameters_*.py in which the variables are explained (in parameters/).

# --- 2
run_segmentation_automatic.py
-> Automatic segmentation of the IMC. The code automatically load the roght and the left border, any interaction is recquired. The argument of the code is set_parameters_*.py in which the variables are explained (in parameters/).

# --- 3
run_segmentation.py
-> Semi automatic segmentation of the IMC. The code can load the results of run_far_wall_detection.py, the experts' annotation or the GUI can be used to manually initalize the method. The argument of the code is set_parameters_*.py in which the variables are explained (in parameters/).
