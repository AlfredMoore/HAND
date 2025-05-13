from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import omni.usd
import os
import torch

def add_table_to_stage(
    stage,
    parent_path: str = "/World",
    name: str = "workstation",
    translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    size: tuple = (2.0, 1.0, 0.5),  # (length, width, height)
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

    UsdGeom.XformCommonAPI(cube_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, size[2] / 2.0))
    UsdGeom.XformCommonAPI(cube_prim).SetScale(Gf.Vec3f(*size))

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
    
    # if save:
    #     if os.path.exists(save_path):
    #         print(f"Current workspace: {os.getcwd()}")
    #         print("Saved path existing, overwriting", save_path)
    #     else:
    #         print("Saved path does not exist, creating", save_path)
    #     omni.usd.get_context().save_as_stage(save_path)
    
if __name__ == "__main__":
    # Example usage
    stage = omni.usd.get_context().get_stage()
    parent_path = "/World"
    add_table_to_stage(
        stage, 
        parent_path, 
        name="workstation",
        size=(2.0, 1.0, 0.5), 
        color=(0.6, 0.6, 0.6),
)
