import bpy
import numpy as np
from mathutils import Matrix, Vector


def set_direction_to(object, focus_point=(0.0, 0.0, 0.0), distance=10.0):
    ## source: https://blender.stackexchange.com/a/100442
    """
    #### Parameters
    - `object`: can be camera or light
    """
    looking_direction = object.location - Vector(focus_point)
    rot_quat = looking_direction.to_track_quat("Z", "Y")
    object.rotation_euler = rot_quat.to_euler()
    object.location = rot_quat @ Vector((0.0, 0.0, distance))


def get_blender_camera_from_KRT(K, R_world2cv, T_world2cv, scale=1):
    assert np.isclose(Matrix(R_world2cv).determinant(), 1)

    scene = bpy.context.scene
    sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0, 0] / s_u
    # recover original resolution
    scene.render.resolution_x = int(resolution_x_in_px / scale)
    scene.render.resolution_y = int(resolution_y_in_px / scale)
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # http://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T  # (3, 3)
    rotation = Matrix((R_cv2world @ R_bcam2cv).tolist())
    location = -R_cv2world @ T_world2cv  # (3, )

    # create a new camera
    bpy.ops.object.camera_add()
    ob = bpy.context.object
    cam = ob.data

    # Lens
    cam.type = "PERSP"
    cam.lens = f_in_mm
    cam.lens_unit = "MILLIMETERS"
    cam.sensor_width = sensor_width_in_mm

    # composition (inverse process of decompose method)
    ob.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()

    location, rotation = ob.matrix_world.decompose()[0:2]

    # Display
    cam.show_name = True
    scene.camera = ob

    return ob
