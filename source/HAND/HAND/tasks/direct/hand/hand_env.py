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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .hand_env_cfg import HandEnvCfg


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

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

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
        
        # Dual Arm
        # auxiliary limits for action, observation and rewards from env_0
        # robot_dof_lower_limits shape: (2*num_joints, )
        self.robot_dof_lower_limits = torch.cat((self.left_arm_dof_lower_limits, self.right_arm_dof_lower_limits))
        self.robot_dof_upper_limits = torch.cat((self.left_arm_dof_upper_limits, self.right_arm_dof_upper_limits))
        self.robot_dof_speed_scales = torch.cat((self.left_arm_dof_speed_scales, self.right_arm_dof_speed_scales))
        
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._left_arm.num_joints + self._right_arm.num_joints),
            device=self.device
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
        # (num_envs, all_joints) + (all_joints,) * (num_envs, all_joints)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)


    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward