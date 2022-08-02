import yaml
import os

from multiprocessing import Pool
from argparse import ArgumentParser
import open3d as o3d

import system_setup as setup
from utils.iterative_ransac import IterativeRANSAC
from utils.dataset import DataLoader
from utils.outlier_removal import Context, StatisticalOutlierRemoval, RadiusOutlierRemoval
from utils.plane_removal import PlaneRemoval
from utils.utils import timer


# Readable point cloud formats for Open3D
pc_formats = ('xyz', 'xyzn', 'xyzrgb', 'pts', 'ply', 'pcd')

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
logs_data_dir = setup.LOGS_DIR
print(logs_data_dir)
if configs['DATASET'] == 'raw':
    raw_data_dir = setup.RAW_DATA_DIR
elif configs['DATASET'] == 'test':
    raw_data_dir = setup.TEST_DATA_DIR
else:
    raise ValueError('The chosen data mode does not exist!')
int_data_dir = setup.INT_DATA_DIR
final_data_dir = setup.FINAL_DATA_DIR
directory = os.fsencode(raw_data_dir)

"""Clean up the directories"""
if args.clean:
    for f in os.listdir(int_data_dir):
        os.remove(os.path.join(int_data_dir, f))
    for f in os.listdir(final_data_dir):
        os.remove(os.path.join(final_data_dir, f))
    for f in os.listdir(logs_data_dir):
        os.remove(os.path.join(logs_data_dir, f))

"""Instantiate relevant objects"""
data = DataLoader(
    raw_data_dir,
    configs['HEURISTICS']['LARGE_PC'],
    configs['HEURISTICS']['VOXEL_SIZE'],
    configs['HEURISTICS']['VOXEL_STEP'],
    configs['VERBOSE']
)

ransac = IterativeRANSAC(
    int_data_dir,
    configs['HEURISTICS']['PLANE_SIZE'],
    configs['THRESH'],
    configs['DEBUG']
)

cloud_post = PlaneRemoval(
    int_data_dir,
    logs_data_dir,
    configs['RAW_REMOVAL']['THRESH']
)

# detect planes in a single point cloud
def process_single_pc(file):
    """Process single point cloud data file"""

    filename = os.fsdecode(file)
    if filename.endswith(pc_formats):
        pcd = data.load_data(filename)
        ransac.detect_planes(pcd, filename)
        ransac.store_best_eqs()
        if configs['VERBOSE']:
            ransac.display_final_pc()


# remove planes in a single point cloud
def post_process_single_pc(file):

    filename = os.fsdecode(file)
    if filename.endswith(pc_formats):
        file_path = os.path.join(raw_data_dir, filename)
        eqs = filename.split('.')[0] + "_best_eqs"
        cloud_post.remove_planes(file_path, eqs)

        if configs['OUT_REMOVAL']['USE']:
            if configs['VERBOSE']:
                cloud_post.display_final_pc()
            context = Context(eval(configs['OUT_REMOVAL']['METHOD'])(final_data_dir,
                                                                     configs['OUT_REMOVAL']['NB_NEIGHBORS'],
                                                                     configs['OUT_REMOVAL']['STD_RATIO']))
            context.run(int_data_dir, filename, configs['DEBUG'])
        else:
            cloud_post.display_final_pc()


@timer
def main():
    # plane detection in downsampled point cloud data
    pool = Pool()
    pool.map(process_single_pc, os.listdir(directory))
    pool.close()
    pool.join()

    # plane removal from original point cloud data
    if configs['RAW_REMOVAL']['USE']:
        pool_post = Pool()
        pool_post.map(post_process_single_pc, os.listdir(directory))
        pool_post.close()
        pool_post.join()


if __name__ == '__main__':
    main()