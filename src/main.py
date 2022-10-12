import os
from argparse import ArgumentParser

import pyransac3d as pyrsc

import system_setup as setup
from processors.plane_detection import IterativeRANSAC
from processors.plane_removal import PlaneRemovalAll
from processors.outlier_removal import (
    Context,
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
)
from utils.dataloader import DataLoaderDS, DataLoaderSTD
from utils.utils import multi_processing, timer, folder_cleanup, load_dict_from_yaml
from utils.runner import Runner, PCFormats
from utils.enums import Mode


@timer
def main():
    """Main Function to run the plane detection and removal pipeline."""
    argparser = ArgumentParser(description="Plane Removal")
    argparser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Config File for the iterative RANSAC. Supported:\n" "- config",
    )
    argparser.add_argument(
        "--clean",
        type=bool,
        default=True,
        help="Remove intermediate and final point cloud files from previous calls.",
    )
    args = argparser.parse_args()

    # Setup the configs directory
    config_path = setup.CONFIG_DIR / (args.config + ".yaml")
    configs = load_dict_from_yaml(config_path)

    # Set up path constants
    if configs["DATASET"] == Mode.RAW.name:
        RAW_DATA_DIR = setup.RAW_DATA_DIR
    elif configs["DATASET"] == Mode.TEST.name:
        RAW_DATA_DIR = setup.TEST_DATA_DIR
    else:
        raise ValueError("The chosen data mode does not exist!")
    INT_DATA_DIR = setup.INT_DATA_DIR
    FINAL_DATA_DIR = setup.FINAL_DATA_DIR
    LOGS_DIR = setup.LOGS_DIR
    DIRECTORY = os.fsencode(RAW_DATA_DIR)

    # Clean up relevant directories
    if args.clean:
        folder_cleanup([INT_DATA_DIR, FINAL_DATA_DIR, LOGS_DIR])

    # Instantiate relevant objects for the runner
    plane_detector = IterativeRANSAC(
        dataloader=DataLoaderDS(
            dir_path=RAW_DATA_DIR,
            down_params=configs["DOWN"],
            verbose=configs["VERBOSE"],
        ),
        geometry=pyrsc.Plane(),
        out_dir=INT_DATA_DIR,
        ransac_params=configs["RANSAC"],
        debug=configs["DEBUG"],
    )

    plane_remover = PlaneRemovalAll(
        dataloader=DataLoaderSTD(RAW_DATA_DIR),
        out_dir=INT_DATA_DIR,
        eqs_dir=LOGS_DIR,
        remove_params=configs["PLANE_REMOVAL"],
    )

    context = Context(
        eval(configs["OUT_REMOVAL"]["METHOD"])(
            out_dir=FINAL_DATA_DIR,
            dataloader=DataLoaderSTD(INT_DATA_DIR),
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

    # plane detection in downsampled point cloud data
    multi_processing(runner.detect_plane, os.listdir(DIRECTORY))

    # plane removal from original point cloud data
    if configs["PLANE_REMOVAL"]["USE"]:
        multi_processing(runner.remove_plane, os.listdir(DIRECTORY))


if __name__ == "__main__":
    main()
