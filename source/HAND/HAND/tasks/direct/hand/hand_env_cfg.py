# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners import CuboidCfg, PreviewSurfaceCfg, RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

import importlib.util
import os
spec = importlib.util.find_spec("HAND")
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(spec.origin)))),
    "assets"
)
ARM_DIS: float = 1.2
# Scene parameters
NUM_ENVS: int = 4096
ENV_SPACING: float = 4.0

@configclass
class HandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 8.3333  # 500 timesteps
    # - spaces definition
    arm_action = 7
    hand_action = 12
    arm_num = 2 
    action_space = arm_num * (arm_action + hand_action)    # 2*19 if single agent for dual arm
    observation_space = 23 # TODO: 23 is a placeholder
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, 
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        ),
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING, replicate_physics=True)
    
    
    # Parallele Environment Cfg
    # workstation
    workstation_size = (2.0, 1.0, 0.5)  # [m]
    workstation = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, workstation_size[2] / 2.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=workstation_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # rigid_body_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, 
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.6), 
                metallic=1.0
            ),
        ),
    )
    
    # robot(s)
    right_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RightArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "HAND_ARM_collection", "HAND_ARM.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "j_*_*": 0.0,
            },
            pos=(ARM_DIS / 2, 0.0, workstation_size[2]),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "finger_1": ImplicitActuatorCfg(
                joint_names_expr=["j_1_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_2": ImplicitActuatorCfg(
                joint_names_expr=["j_2_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_3": ImplicitActuatorCfg(
                joint_names_expr=["j_3_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    
    left_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "HAND_ARM_collection", "HAND_ARM.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "j_*_*": 0.0,
            },
            pos=(-ARM_DIS / 2, 0.0, workstation_size[2]),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "finger_1": ImplicitActuatorCfg(
                joint_names_expr=["j_1_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_2": ImplicitActuatorCfg(
                joint_names_expr=["j_2_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_3": ImplicitActuatorCfg(
                joint_names_expr=["j_3_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # TODO: customize parameters
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0
    
    
if __name__ == "__main__":
    pass