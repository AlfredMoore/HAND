
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg


# TODO: import your EnvCfg to view
from HAND.tasks.direct.hand.hand_env_cfg import HandEnvCfg, ARM_DIS, ASSETS_DIR
# from hand_env_cfg import HandEnvCfg, ARM_DIS, ASSETS_DIR



def main():
    cfg = HandEnvCfg()
    left_arm_cfg = cfg.left_arm
    
    left_arm_cfg = left_arm_cfg.replace(prim_path="/World/LeftArm")
    
    left_arm = Articulation(left_arm_cfg)
    res = left_arm.find_bodies("panda_link7")[0][0]
    
    print("Bodies Name: ", left_arm.body_names)
    print("Left Arm Body: ", res)
    
if __name__ == "__main__":
    # Run this in terminal: python minimal_viewer.py --num_envs 5 --livestream 0
    main()
    simulation_app.close()