from pxr import UsdPhysics, Sdf, Gf
import omni.usd
import os
import torch

def add_fixed_joint(stage, name: str, body0_path: str, body1_path: str,
                    local_pos0=(0.0, 0.0, 0.0), local_pos1=(0.0, 0.0, 0.0)):
    """
    Adds a PhysicsFixedJoint between two rigid body prims.
    
    Parameters:
        stage: Usd.Stage
        name: name of the joint prim (e.g. "LeftArmJoint")
        body0_path: path to first prim (e.g. table)
        body1_path: path to second prim (e.g. arm)
        local_pos0: position on body0 in local frame
        local_pos1: position on body1 in local frame
    """
    joint_path = f"/World/{name}"
    joint_prim = stage.DefinePrim(Sdf.Path(joint_path), "PhysicsFixedJoint")
    joint = UsdPhysics.FixedJoint(joint_prim)

    # Set body targets
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    # Set relative positions (optional, usually zero)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))

    print(f"Added fixed joint: {joint_path}")
