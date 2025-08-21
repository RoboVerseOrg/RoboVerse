import zarr

path = (
    "/home/priosin/murphy/RoboVerse/roboverse_learn/data_policy/CloseBoxFrankaL0_obs:joint_pos_act:joint_pos_100.zarr"
)
root = zarr.open(path, mode="r")

print(root.tree())

if "data" in root:
    print("Keys under /data:")
    print(list(root["data"].keys()))  # 列出所有子数组的名字
