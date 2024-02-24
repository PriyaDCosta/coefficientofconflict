import os
import sys
from importlib import import_module

# # Get the current working directory
# current_directory = os.getcwd()

# # Get the parent directory
# parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# # Add the parent directory to the Python path
# sys.path.append(parent_directory)

# Import the module dynamically
fflow = import_module('team-process-map.feature_engine.features.fflow').get_forward_flow
print(type(fflow))