# ek505-litter-bug
Code base for EK505: Introduction to Robotics (Fall 2023) Litter Bug Project team

## POMPDP Simulations
To run the simulation, install the pomdp-py project using their installation instructions 
https://github.com/h2r/pomdp-py
https://h2r.github.io/pomdp-py/html/installation.html

For installation in windows, you might need to append "py -m " to the pip commands

After installing and testing pomdp-py, paste litterbug pomdp_problems folder into the pomdp-py directory, overwriting any multi_object_search (mos) files 

Run the "problems.py" file in the mos folder to begin the simulation

## YOLO Simulations
The code to run our YOLO model can be found in classifier directory by running the litterbug_yolo.py script. This requires numpy, pandas, and the ultralytics package, as well as the submoduled repository from Jeremy Rico.
