import yaml
import os

from argparse import ArgumentParser

import system_setup as setup
from utils.iterative_ransac import IterativeRANSAC
from utils.dataset import DataLoader
from utils.outlier_removal import StatisticalOutlierRemoval


# Readable point cloud formats for Open3D
pc_formats = ('xyz', 'xyzn', 'xyzrgb', 'pts', 'ply', 'pcd')


def main():
    """Main function to call the plane removal method"""

    """Define the config file to be used"""
    argparser = ArgumentParser(description="Plane Removal")
    argparser.add_argument('--config',
                           type=str,
                           default="config",
                           help="File for the iterative RANSAC configuration. Supported:\n"
                                "- config")
    args = argparser.parse_args()
    config_file = args.config + '.yaml'
    config_path = os.path.join(setup.CONFIG_DIR, config_file)

    """Parse the config.yaml into a python dict"""
    with open(config_path, 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    """Loop over directory"""
    raw_data_dir = setup.RAW_DATA_DIR
    int_data_dir = setup.INT_DATA_DIR
    final_data_dir = setup.FINAL_DATA_DIR
    directory = os.fsencode(raw_data_dir)

    data = DataLoader(raw_data_dir, configs['DEBUG'])

    ransac = IterativeRANSAC(int_data_dir,
                             configs['N_PLANES'],
                             configs['THRESH'],
                             configs['DEBUG'])

    stat = StatisticalOutlierRemoval(final_data_dir,
                                     configs['OUT_REMOVAL']['STATS']['NB_NEIGHBORS'],
                                     configs['OUT_REMOVAL']['STATS']['STD_RATIO'])

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(pc_formats):
            data = data.load_data(filename)
            pcd = ransac.remove_planes(data, filename)
            stat.remove_outliers(pcd, filename)
            stat.display_final_pc()


if __name__ == '__main__':
    main()