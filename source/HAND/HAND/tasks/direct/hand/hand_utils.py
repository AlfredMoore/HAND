import importlib.util
import os
from pathlib import Path


spec = importlib.util.find_spec("HAND")
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(spec.origin)))),
    "assets"
)   # HAND/source/HAND/HAND/__init__.py -> HAND/assets


class AssetManager:
    """
    Manages the assets in asset root diretory. Structure:
    ```
    <asset_root>
        └── <model_root_path>
            ├── <model_1>
            │   ├── model.urdf
            │   ├── info.json
            │   └── model
            │       ├── model.usd
            │       └── ...
            ├── <model_2>
            │   ├── model.urdf
            │   ├── info.json
            │   └── model
            │       ├── model.usd
            │       └── ...
            └── ...
    ```
    """
    def __init__(
        self, 
        asset_root: str = ASSETS_DIR, 
        model_root_path: str = "bottle",
    ) -> None:
        
        self.asset_root = asset_root
        self.model_root_path = model_root_path  # STR or LIST
        self.asset_files = self._scan()
        self.assets = []
        self.assets_dof_props = []
        self.rigid_body_counts = 0
        self.rigid_shape_counts = 0

        # self.hand_arm_usd = os.path.join(asset_root, "HAND_ARM_collection", "HAND_ARM.usd")

    def __len__(self):
        # How many object instances do we have.
        return len(self.asset_files)

    def _scan(self):
        """
        Scan the asset root directory for all the assets.
        
        :return: List of urdf files.
        """
        if isinstance(self.model_root_path, str):
            self.model_root_path = [self.model_root_path]   # Convert str to list

        # Usually just one root folder
        for root in self.model_root_path:
            root_folder = Path(self.asset_root) / root  # self.model_root_path
            asset_files = []
            # subfolders
            for item in root_folder.iterdir():
                # print("Iterating", item)
                if os.path.isdir(item) and os.path.exists(item / "model.urdf"):
                    urdf_path = item / "model.urdf"
                    # urdf_relative_path = os.path.relpath(
                    #     urdf_path, Path(self.asset_root)
                    # )

                    asset_files.append(urdf_path)
                    # print(urdf_relative_path)
        return asset_files
    
    def get_usd_files(self):
        """
        Get the existing USD files path.
        """
        usd_files = []
        
        for urdf_file in self.asset_files:
            # Get the USD file path
            usd_file = urdf_file.parent / "model" / "model.usd"
            if os.path.exists(usd_file):
                usd_files.append(str(usd_file))
            
        return usd_files


    def get_markers(self):
        # Marker handles
        cap_marker_handle_names = [
            "l10",
            "l11",
            "l12",
            "l13",
            "l10b",
            "l11b",
            "l12b",
            "l13b",
        ]
        base_marker_handle_names = [
            "l20",
            "l21",
            "l22",
            "l23",
            "l24",
            "l25",
            "l26",
            "l27",
        ]
        base_marker_handle_names += [
            "l30",
            "l31",
            "l32",
            "l33",
            "l34",
            "l35",
            "l36",
            "l37",
        ]

        return cap_marker_handle_names, base_marker_handle_names
    


if __name__ == "__main__":
    
    # AssetManager
    asset_manager = AssetManager(
        asset_root = ASSETS_DIR, 
        model_root_path = "bottle",
    )
    print(asset_manager.asset_files)
    print("Len: ",len(asset_manager))
    # for asset in asset_manager.assets:
    #     print(asset)