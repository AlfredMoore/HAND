# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``isaacsim.asset.importer.urdf``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html


positional arguments:
  input               The path to the input URDF files.
  output              The path to store the USD files.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --joint-stiffness         The stiffness of the joint drive. (default: 100.0)
  --joint-damping           The damping of the joint drive. (default: 1.0)
  --joint-target-type       The type of control to use for the joint drive. (default: "position")

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
# parser.add_argument("input", type=str, help="The path to the input URDF files.")
# parser.add_argument("output", type=str, help="The path to store the USD files.")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="The type of control to use for the joint drive.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


from HAND.tasks.direct.hand.hand_utils import ASSETS_DIR, AssetManager
from tqdm import tqdm

def main():
    
    urdf_path_list = AssetManager().asset_files
    
    # print(AssetManager().get_usd_files())
    # input()

    urdf_converter_cfg_collection = []

    for iter in urdf_path_list:
        urdf_file = str(iter)
        usd_file = os.path.join(os.path.dirname(iter), "model", "model.usd")
        if not os.path.isabs(usd_file):
            usd_file = os.path.abspath(usd_file)
        if os.path.exists(usd_file):
            print(f"Existing Path: {usd_file}")

        else:
            print(f"Converting to {usd_file}")

            
            # Create Urdf converter config
            urdf_converter_cfg = UrdfConverterCfg(
                asset_path=urdf_file,
                usd_dir=os.path.dirname(usd_file),
                usd_file_name=os.path.basename(usd_file),
                fix_base=args_cli.fix_base,
                merge_fixed_joints=args_cli.merge_joints,
                force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=args_cli.joint_stiffness,
                        damping=args_cli.joint_damping,
                    ),
                    target_type=args_cli.joint_target_type,
                ),
            )
            urdf_converter_cfg_collection.append(urdf_converter_cfg)

    # # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF files: {len(urdf_path_list)}")
    print(f"Output USD files: {len(urdf_converter_cfg_collection)}")
    # print("URDF importer config:")
    # print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # # Create Urdf converter and import the file
    # urdf_converter = UrdfConverter(urdf_converter_cfg)
    # # print output
    # print("URDF importer output:")
    # print(f"Generated USD file: {urdf_converter.usd_path}")
    # print("-" * 80)
    # print("-" * 80)
    # input()
    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        for iter in tqdm(urdf_converter_cfg_collection):
            
            urdf_converter = UrdfConverter(iter)
            print(f"Generated USD file: {urdf_converter.usd_path}")

            stage_utils.open_stage(urdf_converter.usd_path)
            # Reinitialize the simulation
            app = omni.kit.app.get_app_interface()
            app.update()
            # # Run simulation
            # with contextlib.suppress(KeyboardInterrupt):
            #     while app.is_running():
            #         # perform step
            #         app.update()
            #         break
            # input()
            # break

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
