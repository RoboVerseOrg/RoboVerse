#!/usr/bin/env python3
"""Test script to verify XML fixes for the open cabinet task."""

import tempfile
from pathlib import Path
import mujoco
import mjcf

def test_xml_files():
    """Test if the XML files can be loaded without errors."""
    
    # Test handle.xml
    print("Testing handle.xml...")
    try:
        handle_xml = mjcf.from_path("roboverse_data/assets/playground/open_cabinet/mjcf/handle.xml")
        print("✓ handle.xml loaded successfully with mjcf")
        
        # Export to temporary file and test with mujoco
        tmp_dir = tempfile.mkdtemp()
        mjcf.export_with_assets(handle_xml, tmp_dir)
        xml_path = next(Path(tmp_dir).glob("*.xml"))
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print("✓ handle.xml compiled successfully with mujoco")
        
    except Exception as e:
        print(f"✗ handle.xml failed: {e}")
        return False
    
    # Test barrier.xml
    print("Testing barrier.xml...")
    try:
        barrier_xml = mjcf.from_path("roboverse_data/assets/playground/open_cabinet/mjcf/barrier.xml")
        print("✓ barrier.xml loaded successfully with mjcf")
        
        # Export to temporary file and test with mujoco
        tmp_dir = tempfile.mkdtemp()
        mjcf.export_with_assets(barrier_xml, tmp_dir)
        xml_path = next(Path(tmp_dir).glob("*.xml"))
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print("✓ barrier.xml compiled successfully with mujoco")
        
    except Exception as e:
        print(f"✗ barrier.xml failed: {e}")
        return False
    
    print("All XML files passed the test!")
    return True

if __name__ == "__main__":
    test_xml_files()
