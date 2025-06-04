
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg


# TODO: import your EnvCfg to view
from HAND.tasks.direct.hand.hand_env_cfg import HandEnvCfg, ARM_DIS, ASSETS_DIR
# from hand_env_cfg import HandEnvCfg, ARM_DIS, ASSETS_DIR

cfg = HandEnvCfg()
left_arm = cfg.left_arm
right_arm = cfg.right_arm
workstation = cfg.workstation
bottle = cfg.bottle

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    left_arm = left_arm.replace(prim_path="{ENV_REGEX_NS}/LeftArm")
    right_arm = right_arm.replace(prim_path="{ENV_REGEX_NS}/RightArm")

    workstation = workstation.replace(prim_path="{ENV_REGEX_NS}/Workstation")
    bottle = bottle.replace(prim_path="{ENV_REGEX_NS}/Bottle")
    # cfg = HandEnvCfg()

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    while simulation_app.is_running():
        # if count % 500 == 0:
        #     # reset counters
        #     count = 0
        #     break
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
    
    
if __name__ == "__main__":
    # Run this in terminal: python minimal_viewer.py --num_envs 5 --livestream 0
    main()
    simulation_app.close()