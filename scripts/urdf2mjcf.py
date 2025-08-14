# From https://docs.kscale.dev/utils/urdf2mjcf
import argparse

from urdf2mjcf import run

parser = argparse.ArgumentParser(description="Convert URDF to MJCF")
parser.add_argument("--urdf", type=str, required=True, help="Path to input URDF file")
parser.add_argument("--mjcf", type=str, required=True, help="Path to output MJCF file")
parser.add_argument("--copy-meshes", action="store_true", help="Copy mesh files to output directory")
args = parser.parse_args()

run(
    urdf_path=args.urdf,
    mjcf_path=args.mjcf,
    copy_meshes=args.copy_meshes,
)
