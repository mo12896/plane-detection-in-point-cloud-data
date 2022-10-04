import yaml
import os

from multiprocessing import Pool
from argparse import ArgumentParser
from enum import Enum, auto

import system_setup as setup
from utils.iterative_ransac import IterativeRANSAC
from utils.dataset import DataLoader_DS, DataLoader_STD
from utils.outlier_removal import Context, StatisticalOutlierRemoval, RadiusOutlierRemoval
from utils.plane_removal import PlaneRemovalAll
from utils.utils import timer


# Readable point cloud formats for Open3D
pc_formats = ('xyz', 'xyzn', 'xyzrgb', 'pts', 'ply', 'pcd')

class Mode(Enum):
    """Point cloud Folder to choose from."""

    TEST = "test"
    RAW = "raw"
    

"""Define the config file to be used"""
argparser = ArgumentParser(description="Plane Removal")
argparser.add_argument('--config',
                       type=str,
                       default="config",
                       help="File for the iterative RANSAC configuration. Supported:\n"
                            "- config")
argparser.add_argument('--clean',
                       type=bool,
                       default=True,
                       help="Remove intermediate and final point cloud files from previous calls.")

args = argparser.parse_args()

config_file = args.config + '.yaml'
config_path = os.path.join(setup.CONFIG_DIR, config_file)

"""Parse the config.yaml into a python dict"""
with open(config_path, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        print("Loaded config file!")
    except yaml.YAMLError as exc:
        print(exc)

"""Set up variables"""
if configs['DATASET'] == Mode.RAW.value:
    raw_data_dir = setup.RAW_DATA_DIR
elif configs['DATASET'] == Mode.TEST.value:
    raw_data_dir = setup.TEST_DATA_DIR
else:
    raise ValueError('The chosen data mode does not exist!')

int_data_dir = setup.INT_DATA_DIR
final_data_dir = setup.FINAL_DATA_DIR
logs_data_dir = setup.LOGS_DIR
directory = os.fsencode(raw_data_dir)

"""Clean up relevant directories"""
if args.clean:
    for f in os.listdir(int_data_dir):
        os.remove(os.path.join(int_data_dir, f))
    for f in os.listdir(final_data_dir):
        os.remove(os.path.join(final_data_dir, f))
    for f in os.listdir(logs_data_dir):
        os.remove(os.path.join(logs_data_dir, f))

"""Instantiate relevant objects"""
data = DataLoader_DS(
    dir_path=raw_data_dir,
    down_params=configs['DOWN'],
    verbose=configs['VERBOSE']
)

ransac = IterativeRANSAC(
    dataloader=data,
    data_dir=int_data_dir,
    ransac_params=configs['RANSAC'],
    debug=configs['DEBUG']
)

raw_data = DataLoader_STD(raw_data_dir)

plane_remove = PlaneRemovalAll(
    dataloader = raw_data,
    out_dir=int_data_dir,
    eqs_dir=logs_data_dir,
    remove_params=configs['PLANE_REMOVAL']
)

pre_data = DataLoader_STD(int_data_dir)

# detect planes in a single point cloud
def detect_plane(file):
    """Process single point cloud data file"""

    filename = os.fsdecode(file)
    if filename.endswith(pc_formats):
        ransac.detect_planes(filename)
        ransac.store_best_eqs(filename)
        if configs['VERBOSE']:
            ransac.display_final_pc()


# remove planes from a single point cloud
def remove_plane(file):

    filename = os.fsdecode(file)
    if filename.endswith(pc_formats):
        plane_remove.remove_planes(filename)

        if configs['OUT_REMOVAL']['USE']:
            if configs['VERBOSE']:
                plane_remove.display_final_pc()
            context = Context(eval(configs['OUT_REMOVAL']['METHOD'])(out_dir=final_data_dir,
                                                                     dataloader=pre_data,
                                                                     out_params=configs['OUT_REMOVAL']))
            context.run(filename, configs['DEBUG'])
        else:
            plane_remove.display_final_pc()


@timer
# Multiprocessing interface for plane detection and removal
def main():
    # plane detection in downsampled point cloud data
    pool = Pool()
    pool.map(detect_plane, os.listdir(directory))
    pool.close()
    pool.join()

    # plane removal from original point cloud data
    if configs['PLANE_REMOVAL']['USE']:
        pool_post = Pool()
        pool_post.map(remove_plane, os.listdir(directory))
        pool_post.close()
        pool_post.join()


if __name__ == '__main__':
    main()