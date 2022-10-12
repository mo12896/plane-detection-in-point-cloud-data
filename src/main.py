from ast import PyCF_ONLY_AST
import pstats
import yaml
import os

from multiprocessing import Pool
from argparse import ArgumentParser

import pyransac3d as pyrsc

import system_setup as setup
from utils.plane_detection import IterativeRANSAC
from utils.dataloader import DataLoaderDS, DataLoaderSTD
from utils.outlier_removal import (
    Context,
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
)
from utils.plane_removal import PlaneRemovalAll
from utils.utils import timer
from utils.runner import Runner, PCFormats
from utils.enums import Mode


# Define the config file to be used
argparser = ArgumentParser(description="Plane Removal")
argparser.add_argument(
    "--config",
    type=str,
    default="config",
    help="File for the iterative RANSAC configuration. Supported:\n" "- config",
)
argparser.add_argument(
    "--clean",
    type=bool,
    default=True,
    help="Remove intermediate and final point cloud files from previous calls.",
)

args = argparser.parse_args()

config_file = args.config + ".yaml"
config_path = setup.CONFIG_DIR / config_file

# Parse the config.yaml into a python dicts
try:
    configs = yaml.safe_load(config_path.read_text())
    print("Loaded config file!")
except yaml.YAMLError as exc:
    print(exc)


# Set up variables
if configs["DATASET"] == Mode.RAW.name:
    raw_data_dir = setup.RAW_DATA_DIR
elif configs["DATASET"] == Mode.TEST.name:
    raw_data_dir = setup.TEST_DATA_DIR
else:
    raise ValueError("The chosen data mode does not exist!")

int_data_dir = setup.INT_DATA_DIR
final_data_dir = setup.FINAL_DATA_DIR
logs_data_dir = setup.LOGS_DIR
directory = os.fsencode(raw_data_dir)

# Clean up relevant directories
if args.clean:
    for f in os.listdir(int_data_dir):
        os.remove(int_data_dir / f)
    for f in os.listdir(final_data_dir):
        os.remove(final_data_dir / f)
    for f in os.listdir(logs_data_dir):
        os.remove(logs_data_dir / f)


plane_detector = IterativeRANSAC(
    dataloader=DataLoaderDS(
        dir_path=raw_data_dir, down_params=configs["DOWN"], verbose=configs["VERBOSE"]
    ),
    geometry=pyrsc.Plane(),
    out_dir=int_data_dir,
    ransac_params=configs["RANSAC"],
    debug=configs["DEBUG"],
)

plane_remover = PlaneRemovalAll(
    dataloader=DataLoaderSTD(raw_data_dir),
    out_dir=int_data_dir,
    eqs_dir=logs_data_dir,
    remove_params=configs["PLANE_REMOVAL"],
)

context = Context(
    eval(configs["OUT_REMOVAL"]["METHOD"])(
        out_dir=final_data_dir,
        dataloader=DataLoaderSTD(int_data_dir),
        out_params=configs["OUT_REMOVAL"],
    )
)

runner = Runner(
    plane_detector=plane_detector,
    plane_remover=plane_remover,
    out_remover=context,
    pc_formats=PCFormats,
    configs=configs,
)


@timer
def main():
    """Multiprocessing interface for plane detection and removal"""
    # plane detection in downsampled point cloud data
    pool = Pool()
    pool.map(runner.detect_plane, os.listdir(directory))
    pool.close()
    pool.join()

    # plane removal from original point cloud data
    if configs["PLANE_REMOVAL"]["USE"]:
        pool_post = Pool()
        pool_post.map(runner.remove_plane, os.listdir(directory))
        pool_post.close()
        pool_post.join()


if __name__ == "__main__":
    main()
