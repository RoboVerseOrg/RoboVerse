#!/usr/bin/env python3
"""Check USD file object dimensions and compare with real-world settings.

Usage:
    python check_usd_size.py <usd_file_path> [--real_height REAL_HEIGHT]

Example:
    python check_usd_size.py roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd --real_height 0.12
"""

import argparse
import sys
from pathlib import Path

try:
    from pxr import Usd, UsdGeom
except ImportError:
    print("Error: pxr (USD) library not found. Please run this script in an environment with USD support.")
    sys.exit(1)


def get_usd_dimensions(usd_path: str, prim_path: str = None):
    """Get bounding box dimensions from USD file.

    Args:
        usd_path: Path to USD file
        prim_path: Path to specific prim (default: root prim)

    Returns:
        dict with min_point, max_point, size (width, depth, height), center
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    # Find root prim if not specified
    if prim_path is None:
        # Try common prim paths
        common_paths = ["/bbq_sauce", "/World", "/Root", "/Scene"]
        prim = None
        for path in common_paths:
            test_prim = stage.GetPrimAtPath(path)
            if test_prim and test_prim.IsValid():
                prim = test_prim
                prim_path = path
                break

        # If not found, get first valid non-pseudo-root prim
        if prim is None:

            def find_valid_prim(p, depth=0):
                if depth > 10:  # Limit recursion
                    return None
                if p.IsValid() and p != stage.GetPseudoRoot():
                    # Check if it has geometry
                    if UsdGeom.Boundable(p):
                        return p
                for child in p.GetChildren():
                    result = find_valid_prim(child, depth + 1)
                    if result:
                        return result
                return None

            prim = find_valid_prim(stage.GetPseudoRoot())
            if prim:
                prim_path = prim.GetPath().pathString
            else:
                raise ValueError("No valid geometry prim found in USD file")
    else:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

    # Compute bounding box
    bbox = UsdGeom.Boundable(prim).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    bbox_range = bbox.ComputeAlignedRange()

    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    size = max_point - min_point
    center = (min_point + max_point) / 2.0

    return {
        "prim_path": prim_path,
        "min_point": (min_point[0], min_point[1], min_point[2]),
        "max_point": (max_point[0], max_point[1], max_point[2]),
        "size": (size[0], size[1], size[2]),  # width, depth, height
        "center": (center[0], center[1], center[2]),
    }


def format_dimensions(dims: dict):
    """Format dimensions for display."""
    print("=" * 80)
    print(f"USD File Dimensions: {dims['prim_path']}")
    print("=" * 80)
    print(f"Bounding Box Min: ({dims['min_point'][0]:.6f}, {dims['min_point'][1]:.6f}, {dims['min_point'][2]:.6f}) m")
    print(f"Bounding Box Max: ({dims['max_point'][0]:.6f}, {dims['max_point'][1]:.6f}, {dims['max_point'][2]:.6f}) m")
    print(f"Center:           ({dims['center'][0]:.6f}, {dims['center'][1]:.6f}, {dims['center'][2]:.6f}) m")
    print("-" * 80)
    print(f"Width (X):        {dims['size'][0]:.6f} m = {dims['size'][0] * 100:.2f} cm")
    print(f"Depth (Y):        {dims['size'][1]:.6f} m = {dims['size'][1] * 100:.2f} cm")
    print(f"Height (Z):       {dims['size'][2]:.6f} m = {dims['size'][2] * 100:.2f} cm")
    print("=" * 80)


def compare_with_real(usd_height: float, real_height: float):
    """Compare USD height with real-world height."""
    print("\n" + "=" * 80)
    print("Comparison with Real-World Object:")
    print("=" * 80)
    print(f"USD Height:       {usd_height:.6f} m = {usd_height * 100:.2f} cm")
    print(f"Real Height:      {real_height:.6f} m = {real_height * 100:.2f} cm")

    diff = abs(usd_height - real_height)
    diff_percent = (diff / real_height) * 100 if real_height > 0 else 0

    print(f"Difference:       {diff:.6f} m = {diff * 100:.2f} cm ({diff_percent:.2f}%)")

    if diff < 0.001:  # Less than 1mm difference
        print("Status:          ✓ MATCH (within 1mm)")
    elif diff < 0.005:  # Less than 5mm difference
        print("Status:          ⚠ CLOSE (within 5mm)")
    else:
        print("Status:          ✗ DIFFERENT (more than 5mm difference)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check USD file object dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("usd_path", type=str, help="Path to USD file")
    parser.add_argument("--prim_path", type=str, default=None, help="Prim path (default: root prim)")
    parser.add_argument("--real_height", type=float, default=None, help="Real-world height in meters for comparison")

    args = parser.parse_args()

    usd_path = Path(args.usd_path)
    if not usd_path.exists():
        print(f"Error: USD file not found: {usd_path}")
        sys.exit(1)

    try:
        dims = get_usd_dimensions(str(usd_path), args.prim_path)
        format_dimensions(dims)

        if args.real_height is not None:
            compare_with_real(dims["size"][2], args.real_height)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
