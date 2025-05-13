from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import omni.usd
import os
import torch

# from .add_DualArm import add_dual_arm_to_stage
# from .add_table import add_table_to_stage


def add_table_to_stage(
    stage,
    parent_path: str = "/World",
    name: str = "workstation",
    translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    size: torch.Tensor = torch.tensor([2.0, 1.0, 0.5]),  # (length, width, height)
    color: tuple = (0.5, 0.5, 0.5),  # RGB float values in [0, 1]
    # save: bool = False,
    # save_path: str = "workstation.usd",
):
    # Ensure parent prim exists
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim '{parent_path}' does not exist.")

    translation = Gf.Vec3d(*translation.tolist())
    rotation = Gf.Vec3f(*rotation.tolist())  # degrees
    size = Gf.Vec3f(*size.tolist())  # size in meters
    
    # Define the table root prim
    table_path = f"{parent_path}/{name}"
    table_prim = stage.DefinePrim(Sdf.Path(table_path), "Xform")
    table_prim.SetInstanceable(True)
    
    UsdGeom.XformCommonAPI(table_prim).SetTranslate(translation)
    UsdGeom.XformCommonAPI(table_prim).SetRotate(rotation)

    # Create the cube shape under the table
    cube_path = f"{table_path}/Surface"
    cube_prim = stage.DefinePrim(Sdf.Path(cube_path), "Cube")
    cube_geom = UsdGeom.Cube(cube_prim)

    UsdGeom.XformCommonAPI(cube_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, size[2] / 2.0))  # Center the cube on the Z-axis
    UsdGeom.XformCommonAPI(cube_prim).SetScale(size/2.0)  # Scale to half size

    # Apply visual color
    cube_geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    # Physics collision
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
    meshCollisionAPI.CreateApproximationAttr(
        UsdPhysics.Tokens.boundingCube
    )
    # print(f"Using collider approximation: {meshCollisionAPI.GetApproximationAttr().Get()}")

    print(f"Added table '{name}' under '{parent_path}' with size {size} and color {color}")
    

def add_dual_arm_to_stage(
    stage,
    parent_path: str,
    collection_path: str, 
    usd_path: str,
    translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    scale: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
    distance: float = 1.2,
    # save: bool = False,
    # save_path: str = "dual_arm.usd",
    ):
    """
    Add a dual-arm group to a parent prim in the stage.
    """
    
    # Check file existence
    if os.path.exists(collection_path):
        print("Collection_path exists.")
    else:
        print("collection_path does not exist.")
        raise FileNotFoundError(f"Collection path does not exist: {collection_path}")
    usd_path = os.path.join(collection_path, usd_path)
    if os.path.exists(usd_path):
        print("USD path exists.")
    else:
        print("USD path does not exist.")
        raise FileNotFoundError(f"USD path does not exist: {usd_path}")

    # Convert torch -> Gf
    translation = Gf.Vec3d(*translation.tolist())
    rotation = Gf.Vec3f(*rotation.tolist())  # degrees
    scale = Gf.Vec3f(*scale.tolist())
    
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim '{parent_path}' does not exist.")
    
    DualArm_path = f"{parent_path}/DualArm"
    xgroup_path = Sdf.Path(DualArm_path)
    DualArm_prim = stage.DefinePrim(xgroup_path, "Xform")
    
    xform_group = UsdGeom.XformCommonAPI(DualArm_prim)
    xform_group.SetTranslate(translation)
    xform_group.SetRotate(rotation)
    xform_group.SetScale(scale)

    # Create referenced arms under the group
    right_path = xgroup_path.AppendChild("RightArm")
    right_prim = stage.DefinePrim(right_path, "Xform")
    right_prim.GetReferences().AddReference(usd_path)

    left_path = xgroup_path.AppendChild("LeftArm")
    left_prim = stage.DefinePrim(left_path, "Xform")
    left_prim.GetReferences().AddReference(usd_path)

    UsdGeom.XformCommonAPI(right_prim).SetTranslate(Gf.Vec3d(distance / 2, 0, 0))
    UsdGeom.XformCommonAPI(left_prim).SetTranslate(Gf.Vec3d(-distance / 2, 0, 0))
    UsdGeom.XformCommonAPI(right_prim).SetRotate(Gf.Vec3f(0, 0, 180))  # Face each other
    UsdGeom.XformCommonAPI(left_prim).SetRotate(Gf.Vec3f(0, 0, 0))  # Face each other

    print(f"Dual-arm group created in {DualArm_path}.")


def save(save_path: str):
    if os.path.exists(save_path):
        print("Saved path existing, overwriting", save_path)
    else:
        print("Saved path does not exist, creating", save_path)
    omni.usd.get_context().save_as_stage(save_path)


def main():
    stage = omni.usd.get_context().get_stage()

    table_size = torch.tensor([2.0, 1.5, 0.5])
    
    add_table_to_stage(
        stage=stage, 
        parent_path="/World",
        name="workstation",
        size=table_size,
        color=(0.6, 0.6, 0.6),
    )
    
    add_dual_arm_to_stage(
        stage=stage,
        parent_path="/World",
        collection_path="/home/linuxmo/robotics/IsaacLab/source/HAND/HAND_ARM_collection",
        usd_path="HAND_ARM.usd",
        distance=1.2,
        translation=torch.tensor([0.0, 0.0, table_size[2]]),
        rotation=torch.tensor([0.0, 0.0, 0.0]),
        scale=torch.tensor([1.0, 1.0, 1.0]),
    )


if __name__ == "__main__":
    main()