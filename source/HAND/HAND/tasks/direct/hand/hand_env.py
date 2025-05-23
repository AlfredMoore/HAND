# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .hand_env_cfg import HandEnvCfg
# from HAND.tasks.direct.hand.hand_env_cfg import HandEnvCfg, ARM_DIS, ASSETS_DIR
from HAND.tasks.direct.hand.hand_utils import ASSETS_DIR, AssetManager


class HandEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: HandEnvCfg

    def __init__(self, cfg: HandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """
            Compute pose in env-local coordinates
            
            :return: Tensor([px, py, pz, qw, qx, qy, qz], device=device)
            """
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            
            # Isaac Sim and Isaac Lab use wxyz, but Isaac Gym preview uses xyzw
            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real
            
            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        stage = get_current_stage()
        
        # Left Arm
        # auxiliary limits for action, observation and rewards from env_0
        # data.soft_joint_pos_limits shape: (num_instances, num_joints, 2)
        # left/right_arm_dof_lower_limits shape: (num_joints, )
        self.left_arm_dof_lower_limits = self._left_arm.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.left_arm_dof_upper_limits = self._left_arm.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        
        self.left_arm_dof_speed_scales = torch.ones_like(self.left_arm_dof_lower_limits)
        self.left_arm_dof_speed_scales[self._left_arm.find_joints("j_.*_.*")[0]] = 0.1
        
        self.left_arm_dof_targets = torch.zeros(
            (self.num_envs, self._left_arm.num_joints), 
            device=self.device
        )
        
        left_arm_ee_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/LeftArm/panda_link7")),
            self.device,
        )
        # TODO: Do we need palm pose?
        left_gripper_finger1_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/l_1_4")),
            self.device,
        )
        left_gripper_finger2_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/l_2_4")),
            self.device,
        )
        left_gripper_finger3_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/l_3_4")),
            self.device,
        )
        
        # Right Arm
        # auxiliary limits for action, observation and rewards from env_0
        # data.soft_joint_pos_limits shape: (num_instances, num_joints, 2)
        # left/right_arm_dof_lower_limits shape: (num_joints, )
        self.right_arm_dof_lower_limits = self._right_arm.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.right_arm_dof_upper_limits = self._right_arm.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        
        self.right_arm_dof_speed_scales = torch.ones_like(self.right_arm_dof_lower_limits)
        self.right_arm_dof_speed_scales[self._right_arm.find_joints("j_.*_.*")[0]] = 0.1
        
        self.right_arm_dof_targets = torch.zeros(
            (self.num_envs, self._right_arm.num_joints), 
            device=self.device
        )

        right_arm_ee_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RightArm/panda_instanceable/panda_link7")),
            self.device,
        )
        # TODO: Do we need palm pose?
        right_gripper_finger1_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RightArm/dg3f/l_1_4")),
            self.device,
        )
        right_gripper_finger2_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RightArm/dg3f/l_2_4")),
            self.device,
        )
        right_gripper_finger3_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RightArm/dg3f/l_3_4")),
            self.device,
        )
        
        # TODO: grasp pose variables for reward function
        # TODO: twist pose variables for reward function
        
        # Potentially useful variable for reward function
        self.left_arm_ee_link_idx = self._left_arm.find_bodies("panda_link7")[0][0]
        self.left_gripper_base = self._left_arm.find_bodies("l_dg_mount")[0][0]
        self.left_gripper_finger1_link_idx = self._left_arm.find_bodies("l_1_4")[0][0]
        self.left_gripper_finger2_link_idx = self._left_arm.find_bodies("l_2_4")[0][0]
        self.left_gripper_finger3_link_idx = self._left_arm.find_bodies("l_3_4")[0][0]
        
        self.right_arm_ee_link_idx = self._right_arm.find_bodies("panda_link7")[0][0]
        self.right_gripper_base = self._right_arm.find_bodies("l_dg_mount")[0][0]
        self.right_gripper_finger1_link_idx = self._right_arm.find_bodies("l_1_4")[0][0]
        self.right_gripper_finger2_link_idx = self._right_arm.find_bodies("l_2_4")[0][0]
        self.right_gripper_finger3_link_idx = self._right_arm.find_bodies("l_3_4")[0][0]
        
        # TODO: bottle articulation
        # self.bottle_link_idx = self._bottle.find_bodies("bottle")[0][0]
        
        # TODO: grasp, hold, twist
        

    def _setup_scene(self):
        # self.scene has been created through self.cfg.scene
        
        # Articulation
        self._right_arm = Articulation(self.cfg.right_arm)
        self._left_arm = Articulation(self.cfg.left_arm)
        # TODO: bottle articulation
        # self._bottle = Articulation(self.cfg.bottle)
        # Rigid objects
        self._workstation = RigidObject(self.cfg.workstation)
        
        self.scene.articulations["right_arm"] = self._right_arm
        self.scene.articulations["left_arm"] = self._left_arm
        # TODO
        # self.scene.articulations["bottle"] = self._bottle
        self.scene.rigid_objects["workstation"] = self._workstation

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Action scaling
        # actions: (left_arm+right_arm , )
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        # left_arm
        left_arm_actions = self.actions[:, :self._left_arm.num_dofs]
        left_arm_targets = self.left_arm_dof_targets + self.left_arm_dof_speed_scales * self.dt * left_arm_actions * self.cfg.action_scale
        self.left_arm_dof_targets[:] = torch.clamp(left_arm_targets, self.left_arm_dof_lower_limits, self.left_arm_dof_upper_limits)
        
        # right_arm
        right_arm_actions = self.actions[:, self._left_arm.num_dofs:]
        right_arm_targets = self.right_arm_dof_targets + self.right_arm_dof_speed_scales * self.dt * right_arm_actions * self.cfg.action_scale
        self.right_arm_dof_targets[:] = torch.clamp(right_arm_targets, self.right_arm_dof_lower_limits, self.right_arm_dof_upper_limits)
        
    def _apply_action(self) -> None:
        # left_arm
        self._left_arm.set_joint_position_target(self.left_arm_dof_targets)
        
        # right_arm
        self._right_arm.set_joint_position_target(self.right_arm_dof_targets)


    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: terminated logic. When succeed, the bottle has been twist off
        terminated = ...
        
        # TODO: truncated logic
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # TODO: reward function
        reward, log = compute_rewards(
            # self.actions,
            # self._cabinet.data.joint_pos,
            # self.robot_grasp_pos,
            # self.drawer_grasp_pos,
            # self.robot_grasp_rot,
            # self.drawer_grasp_rot,
            # robot_left_finger_pos,
            # robot_right_finger_pos,
            # self.gripper_forward_axis,
            # self.drawer_inward_axis,
            # self.gripper_up_axis,
            # self.drawer_up_axis,
            # self.num_envs,
            # self.cfg.dist_reward_scale,
            # self.cfg.rot_reward_scale,
            # self.cfg.open_reward_scale,
            # self.cfg.action_penalty_scale,
            # self.cfg.finger_reward_scale,
            # self._robot.data.joint_pos,
        )
        
        self.extras["log"] = log
        
        return reward
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # robot state
        # left arm
        left_joint_pos = self._left_arm.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._left_arm.num_joints),
            self.device,
        )
        left_joint_pos = torch.clamp(left_joint_pos, self.left_arm_dof_lower_limits, self.left_arm_dof_upper_limits)
        left_joint_vel = torch.zeros_like(left_joint_pos)
        self._left_arm.set_joint_position_target(left_joint_pos, env_ids=env_ids)
        self._left_arm.write_joint_state_to_sim(left_joint_pos, left_joint_vel, env_ids=env_ids)
        
        # right arm
        right_joint_pos = self._left_arm.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._right_arm.num_joints),
            self.device,
        )
        right_joint_pos = torch.clamp(right_joint_pos, self.right_arm_dof_lower_limits, self.right_arm_dof_upper_limits)
        right_joint_vel = torch.zeros_like(right_joint_pos)
        self._right_arm.set_joint_position_target(right_joint_pos, env_ids=env_ids)
        self._right_arm.write_joint_state_to_sim(right_joint_pos, right_joint_vel, env_ids=env_ids)

        # TODO: bottle state
        # zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        # self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)
        
    def _get_observations(self) -> dict:
        # left arm
        left_arm_dof_pos_scaled = (
            2.0
            * (self._left_arm.data.joint_pos - self.left_arm_dof_lower_limits)
            / (self.left_arm_dof_lower_limits - self.left_arm_dof_upper_limits)
            - 1.0
        )
        # TODO: Heuristic function to compute the left arm position to target
        left_arm_to_target = ...
        
        # right arm
        right_arm_dof_pos_scaled = (
            2.0
            * (self._right_arm.data.joint_pos - self.right_arm_dof_lower_limits)
            / (self.right_arm_dof_lower_limits - self.right_arm_dof_upper_limits)
            - 1.0
        )
        # TODO: Heuristic function to compute the right arm position to target
        right_arm_to_target = ...
        
        obs = torch.cat(
            (
                left_arm_dof_pos_scaled,
                right_arm_dof_pos_scaled,
                self._left_arm.data.joint_vel * self.cfg.dof_velocity_scale,
                self._right_arm.data.joint_vel * self.cfg.dof_velocity_scale,
                left_arm_to_target,
                right_arm_to_target,
                # TODO: bottle state
            ),
            dim=-1,
        )
        
        observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            raise ValueError("env_ids cannot be None")
        
        # TODO: compute intermediate values
        # TODO: grasp pose variables for reward function
        # TODO: twist pose variables for reward function


# TODO: reward function
@torch.jit.script
def compute_rewards(
    # rew_scale_alive: float,
    # rew_scale_terminated: float,
    # rew_scale_pole_pos: float,
    # rew_scale_cart_vel: float,
    # rew_scale_pole_vel: float,
    # pole_pos: torch.Tensor,
    # pole_vel: torch.Tensor,
    # cart_pos: torch.Tensor,
    # cart_vel: torch.Tensor,
    # reset_terminated: torch.Tensor,
):
    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    # return total_reward
    pass