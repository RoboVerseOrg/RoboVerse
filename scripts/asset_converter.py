class AssetConverter:
    def __init__(self, asset_root: str):
        self.asset_root = asset_root
        self.obj = None

    def from_urdf(self, urdf_path: str):
        pass

    def from_obj(self, obj_path: str):
        pass

    def from_mjcf(self, mjcf_path: str):
        pass

    def to_urdf(self, output_path):
        pass

    def to_mjcf(self, output_path):
        pass

    def to_usd(self, output_path):
        pass
