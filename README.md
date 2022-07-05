# Plane removal in point cloud data

The goal of this repository is to provide a simple interface for plane removal in point cloud data.

### Setup

To set up the folder structure, simply run the setup.sh from the repository's root folder. After that, place
your point cloud images in the data/raw/ folder for processing. Alternatively, the user can specify
the source and destination path manually in the query. Finally, define the hyperparameters in a config file.
Please find an example file attached to this repository. To query the interface, run the main function
(with specified source and destination path). 

### Results

State | Raw                                    | Final                                    | 
--- |----------------------------------------|------------------------------------------| 
Pointclouds | <img src="docs/raw.png" width="500" /> | <img src="docs/final.png" width="452" /> | 