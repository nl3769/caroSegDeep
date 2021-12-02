The three main functions are in 'caroSegDeep/SEGMENTATION/scripts/'. The argument parameters are detailed in 'caroSegDeep/SEGMENTATION/parameters/set_parameters_*.py'.

# --- 1
run_far_wall_detection.py
-> Detection of the far wall. Results are saved in .txt files.

# --- 2
run_segmentation_automatic.py
-> Fully-automatic segmentation of the IMC. The code automatically load the right and the left border. No interaction is required.

# --- 3
run_segmentation.py
-> Semi-automatic segmentation of the IMC. The code can either load the results of 'run_far_wall_detection.py', or the experts manual annotations, or the GUI can be used to manually initialize the method.
