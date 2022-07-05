import yaml
import os

from argparse import ArgumentParser

import system_setup as setup
from utils.iterative_ransac import IterativeRANSAC
from utils.dataset import DataLoader
from utils.outlier_removal import StatisticalOutlierRemoval


def main():
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

    data = DataLoader()
    iran = IterativeRANSAC(data, configs['N_PLANES'], configs['THRESH'], configs['DEBUGGING'])
    pcd = iran.remove_planes()
    pcd_out = StatisticalOutlierRemoval(pcd, configs['NB_NEIGHBORS'], configs['STD_RATIO'])
    pcd_out.display_final_pcd()


    return pcd_out


if __name__ == '__main__':
    main()