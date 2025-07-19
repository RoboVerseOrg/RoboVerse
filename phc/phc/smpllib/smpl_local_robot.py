import os
import sys
import time
import argparse
import torch
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
import mujoco
from uhc.khrylib.mocap.skeleton_local import Skeleton
from uhc.khrylib.mocap.skeleton_mesh_local import Skeleton as SkeletonMesh
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from collections import defaultdict
from scipy.spatial import ConvexHull
from stl import mesh
from uhc.utils.geom import quadric_mesh_decimation, center_scale_mesh
import uuid
import atexit
import shutil
import joblib
# from scipy.spatial.qhull import _Qhull
from uhc.utils.flags import flags
import cv2
import mujoco.viewer


def parse_vec(string):
    return np.fromstring(string, sep=" ")


def parse_fromto(string):
    fromto = np.fromstring(string, sep=" ")
    return fromto[:3], fromto[3:]


def normalize_range(value, lb, ub):
    return (value - lb) / (ub - lb) * 2 - 1


def denormalize_range(value, lb, ub):
    return (value + 1) * 0.5 * (ub - lb) + lb


def vec_to_polar(v):
    phi = math.atan2(v[1], v[0])
    theta = math.acos(v[2])
    return np.array([theta, phi])


def polar_to_vec(p):
    v = np.zeros(3)
    v[0] = math.sin(p[0]) * math.cos(p[1])
    v[1] = math.sin(p[0]) * math.sin(p[1])
    v[2] = math.cos(p[0])
    return v


def in_hull(hull, queries):
    tolerance = 1e-3
    if len(queries.shape) == 1:
        queries = queries[None, ]
    return np.all(
        np.add(np.dot(queries, hull.equations[:, :-1].T),
               hull.equations[:, -1]) <= tolerance,
        axis=1,
    )


def get_joint_geometries(
    smpl_verts,
    smpl_jts,
    skin_weights,
    joint_names,
    geom_dir,
    scale_dict={},
    suffix=None,
    verbose=False,
    min_num_vert=50,
):

    vert_to_joint = skin_weights.argmax(axis=1)
    hull_dict = {}

    # create joint geometries
    os.makedirs(geom_dir, exist_ok=True)
    for jind, jname in enumerate(joint_names):
        vind = np.where(vert_to_joint == jind)[0]
        if len(vind) == 0:
            print(f"{jname} has no vertices!")
            continue
        norm_verts = (smpl_verts[vind] - smpl_jts[jind]) * scale_dict.get(jname, 1)

        hull = ConvexHull(smpl_verts[vind])
        norm_hull = ConvexHull(norm_verts)
        hull_dict[jname] = {
            "norm_hull": norm_hull,
            "norm_verts": norm_verts,
            "verts": smpl_verts[vind],
            "hull": hull,
            "volume": hull.volume
        }
        center = norm_verts[hull.vertices].mean(axis=0)
        jgeom = mesh.Mesh(
            np.zeros(hull.simplices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(hull.simplices):
            for j in range(3):
                jgeom.vectors[i][j] = norm_verts[f[j], :]
            # check if the face's normal is facing outward
            normal = np.cross(
                jgeom.vectors[i][1] - jgeom.vectors[i][0],
                jgeom.vectors[i][2] - jgeom.vectors[i][0],
            )
            out_vec = jgeom.vectors[i].mean(axis=0) - center
            if np.dot(normal, out_vec) < 0:
                jgeom.vectors[i] = jgeom.vectors[i][[0, 2, 1]]  # flip the face
        if suffix is None:
            fname = f"{geom_dir}/{jname}.stl"
        else:
            fname = f"{geom_dir}/{jname}_{suffix}.stl"
        jgeom.save(fname)

        # mesh simplification with vtk
        # min_num_vert = 50
        min_num_vert = 50
        cur_num_vert = len(hull.vertices)
        reduction_rate = min(0.9, 1.0 - min_num_vert / cur_num_vert)

        quadric_mesh_decimation(fname, reduction_rate, verbose=verbose)

    return hull_dict


def get_geom_dict(
    smpl_verts,
    smpl_jts,
    skin_weights,
    joint_names,
    scale_dict={},
):

    vert_to_joint = skin_weights.argmax(axis=1)
    hull_dict = {}

    # create joint geometries
    for jind, jname in enumerate(joint_names):
        vind = np.where(vert_to_joint == jind)[0]
        if len(vind) == 0:
            print(f"{jname} has no vertices!")
            continue
        norm_verts = (smpl_verts[vind] - smpl_jts[jind]) * scale_dict.get(jname, 1)
        hull = ConvexHull(norm_verts)

        hull_dict[jname] = {
            "norm_hull": hull,
            "norm_verts": norm_verts,
            "verts": smpl_verts[vind],
            "volume": hull.volume
        }

    return hull_dict


def update_joint_limits(joint_range):
    joint_range["Head"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["Head"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["Head"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["Chest"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Chest"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Chest"][2] = np.array([-np.pi / 3, np.pi / 3])

    joint_range["Spine"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Spine"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Spine"][2] = np.array([-np.pi / 3, np.pi / 3])

    joint_range["Torso"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Torso"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Torso"][2] = np.array([-np.pi / 3, np.pi / 3])

    ##############################

    joint_range["L_Thorax"][0] = np.array([-np.pi, np.pi])
    joint_range["L_Thorax"][1] = np.array([-np.pi, np.pi])
    joint_range["L_Thorax"][2] = np.array([-np.pi, np.pi])

    joint_range["R_Thorax"][0] = np.array([-np.pi, np.pi])
    joint_range["R_Thorax"][1] = np.array([-np.pi, np.pi])
    joint_range["R_Thorax"][2] = np.array([-np.pi, np.pi])

    joint_range["L_Shoulder"][0] = np.array([-np.pi, np.pi])
    joint_range["L_Shoulder"][1] = np.array([-np.pi, np.pi])
    joint_range["L_Shoulder"][2] = np.array([-np.pi, np.pi])

    joint_range["R_Shoulder"][0] = np.array([-np.pi, np.pi])
    joint_range["R_Shoulder"][1] = np.array([-np.pi, np.pi])
    joint_range["R_Shoulder"][2] = np.array([-np.pi, np.pi])

    ##############################

    joint_range["L_Hip"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Hip"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Hip"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["R_Hip"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Hip"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Hip"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["L_Knee"][0] = np.array([-np.pi, np.pi])
    joint_range["L_Knee"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Knee"][2] = np.array([-np.pi / 32, np.pi / 32])

    joint_range["R_Knee"][0] = np.array([-np.pi, np.pi])
    joint_range["R_Knee"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Knee"][2] = np.array([-np.pi / 32, np.pi / 32])

    joint_range["L_Ankle"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Ankle"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["R_Ankle"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Ankle"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["L_Toe"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Toe"][1] = np.array([-np.pi / 4, np.pi / 4])
    joint_range["L_Toe"][2] = np.array([-np.pi / 4, np.pi / 4])

    joint_range["R_Toe"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Toe"][1] = np.array([-np.pi / 4, np.pi / 4])
    joint_range["R_Toe"][2] = np.array([-np.pi / 4, np.pi / 4])
    return joint_range


def update_joint_limits_upright(joint_range):

    joint_range["L_Knee"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Knee"][1] = np.array([0, np.pi])
    joint_range["L_Knee"][2] = np.array([-np.pi / 32, np.pi / 32])

    joint_range["R_Knee"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Knee"][1] = np.array([0, np.pi])
    joint_range["R_Knee"][2] = np.array([-np.pi / 32, np.pi / 32])

    joint_range["L_Ankle"][0] = np.array([-np.pi / 4, np.pi / 4])
    joint_range["L_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Ankle"][2] = np.array([-np.pi / 4, np.pi / 4])

    joint_range["R_Ankle"][0] = np.array([-np.pi / 4, np.pi / 4])
    joint_range["R_Ankle"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Ankle"][2] = np.array([-np.pi / 4, np.pi / 4])

    joint_range["L_Hip"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Hip"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["L_Hip"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["R_Hip"][0] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Hip"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["R_Hip"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["R_Elbow"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Elbow"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Elbow"][2] = np.array([0, np.pi])

    joint_range["L_Elbow"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Elbow"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Elbow"][2] = np.array([-np.pi, 0])


    joint_range["Spine"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Spine"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Spine"][2] = np.array([-np.pi / 3, np.pi / 3])

    joint_range["Torso"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Torso"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Torso"][2] = np.array([-np.pi / 3, np.pi / 3])

    joint_range["Chest"][0] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Chest"][1] = np.array([-np.pi / 3, np.pi / 3])
    joint_range["Chest"][2] = np.array([-np.pi / 3, np.pi / 3])

    joint_range["R_Thorax"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Thorax"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["R_Thorax"][2] = np.array([-np.pi / 32, np.pi / 32])

    joint_range["L_Thorax"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Thorax"][1] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["L_Thorax"][2] = np.array([-np.pi / 32, np.pi / 32])


    joint_range["Head"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["Head"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["Head"][2] = np.array([-np.pi / 2, np.pi / 2])

    joint_range["Neck"][0] = np.array([-np.pi / 32, np.pi / 32])
    joint_range["Neck"][1] = np.array([-np.pi / 2, np.pi / 2])
    joint_range["Neck"][2] = np.array([-np.pi / 2, np.pi / 2])


    # joint_range["L_Toe"][0] = np.array([-np.pi / 32, np.pi / 32])
    # joint_range["L_Toe"][1] = np.array([-np.pi / 2, np.pi / 2])
    # joint_range["L_Toe"][2] = np.array([-np.pi / 32, np.pi / 32])

    # joint_range["R_Toe"][0] = np.array([-np.pi / 32, np.pi / 32])
    # joint_range["R_Toe"][1] = np.array([-np.pi / 2, np.pi / 2])
    # joint_range["R_Toe"][2] = np.array([-np.pi / 32, np.pi / 32])
    return joint_range


class Joint:
    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib["name"]
        self.type = node.attrib["type"] if "type" in node.attrib else "free"

        if self.type == "hinge":
            self.range = np.deg2rad(
                parse_vec(node.attrib.get("range", "-360 360")))
        actu_node = (body.tree.getroot().find("actuator").find(
            f'motor[@joint="{self.name}"]'))
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None

        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.pos = parse_vec("0 0 0")
        if self.type == "hinge":
            self.axis = vec_to_polar(parse_vec(node.attrib["axis"]))

        if self.local_coord:
            self.pos += body.pos

        self.damping = (parse_vec(node.attrib["damping"])
                        if "damping" in node.attrib else np.array([0]))
        self.stiffness = (parse_vec(node.attrib["stiffness"])
                          if "stiffness" in node.attrib else np.array([0]))
        self.armature = (parse_vec(node.attrib["armature"])
                         if "armature" in node.attrib else np.array([0.01]))

        self.frictionloss = (parse_vec(node.attrib["frictionloss"]) if
                             "frictionloss" in node.attrib else np.array([0]))
        # import ipdb; ipdb.set_trace()
        # assert np.all(self.pos == body.pos)

    def __repr__(self):
        return "joint_" + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg.get("joint_params", {}))
        for name, specs in self.param_specs.items():
            if "lb" in specs and isinstance(specs["lb"], list):
                specs["lb"] = np.array(specs["lb"])
            if "ub" in specs and isinstance(specs["ub"], list):
                specs["ub"] = np.array(specs["ub"])

    def sync_node(self, rename=False, index=0):
        pos = self.pos - self.body.pos if self.local_coord else self.pos

        if rename:
            self.name = self.body.name + "_joint_" + str(index)
        self.node.attrib["name"] = self.name

        if self.type == "hinge":
            axis_vec = polar_to_vec(self.axis)
            self.node.attrib["axis"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in axis_vec])
            self.node.attrib["pos"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in pos])
            self.node.attrib["damping"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in self.damping])
            self.node.attrib["stiffness"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in self.stiffness])
            self.node.attrib["armature"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in self.armature])
        elif self.type == "free":
            pass

        if self.actuator is not None:
            self.actuator.sync_node()

    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if "axis" in self.param_specs:
            if self.type == "hinge":
                if get_name:
                    param_list += ["axis_theta", "axis_phi"]
                else:
                    axis = normalize_range(
                        self.axis,
                        np.array([0, -2 * np.pi]),
                        np.array([np.pi, 2 * np.pi]),
                    )
                    param_list.append(axis)
            elif pad_zeros:
                param_list.append(np.zeros(2))

        if self.actuator is not None:
            self.actuator.get_params(param_list, get_name)
        elif pad_zeros:
            param_list.append(
                np.zeros(3 if self.type == "free" else 1)
            )  # ZL currently a workaround for supporting 3D joints

        if "damping" in self.param_specs:
            if get_name:
                param_list.append("damping")
            else:
                if not self.param_inited and self.param_specs["damping"].get(
                        "rel", False):
                    self.param_specs["damping"]["lb"] += self.damping
                    self.param_specs["damping"]["ub"] += self.damping
                    self.param_specs["damping"]["lb"] = max(
                        self.param_specs["damping"]["lb"],
                        self.param_specs["damping"].get("min", -np.inf),
                    )
                    self.param_specs["damping"]["ub"] = min(
                        self.param_specs["damping"]["ub"],
                        self.param_specs["damping"].get("max", np.inf),
                    )
                damping = normalize_range(
                    self.damping,
                    self.param_specs["damping"]["lb"],
                    self.param_specs["damping"]["ub"],
                )
                param_list.append(damping.flatten())

        if "armature" in self.param_specs:
            if get_name:
                param_list.append("armature")
            else:
                if not self.param_inited and self.param_specs["armature"].get(
                        "rel", False):
                    self.param_specs["armature"]["lb"] += self.armature
                    self.param_specs["armature"]["ub"] += self.armature
                    self.param_specs["armature"]["lb"] = max(
                        self.param_specs["armature"]["lb"],
                        self.param_specs["armature"].get("min", -np.inf),
                    )
                    self.param_specs["armature"]["ub"] = min(
                        self.param_specs["armature"]["ub"],
                        self.param_specs["armature"].get("max", np.inf),
                    )

                armature = normalize_range(
                    self.armature,
                    self.param_specs["armature"]["lb"],
                    self.param_specs["armature"]["ub"],
                )
                param_list.append(armature.flatten())

        if "stiffness" in self.param_specs:
            if get_name:
                param_list.append("stiffness")
            else:
                if not self.param_inited and self.param_specs["stiffness"].get(
                        "rel", False):
                    self.param_specs["stiffness"]["lb"] += self.stiffness
                    self.param_specs["stiffness"]["ub"] += self.stiffness
                    self.param_specs["stiffness"]["lb"] = max(
                        self.param_specs["stiffness"]["lb"],
                        self.param_specs["stiffness"].get("min", -np.inf),
                    )
                    self.param_specs["stiffness"]["ub"] = min(
                        self.param_specs["stiffness"]["ub"],
                        self.param_specs["stiffness"].get("max", np.inf),
                    )
                stiffness = normalize_range(
                    self.stiffness,
                    self.param_specs["stiffness"]["lb"],
                    self.param_specs["stiffness"]["ub"],
                )
                param_list.append(stiffness.flatten())
        if "frictionloss" in self.param_specs:
            if get_name:
                param_list.append("frictionloss")
            else:
                if not self.param_inited and self.param_specs[
                        "frictionloss"].get("rel", False):
                    self.param_specs["frictionloss"]["lb"] += self.frictionloss
                    self.param_specs["frictionloss"]["ub"] += self.frictionloss
                    self.param_specs["frictionloss"]["lb"] = max(
                        self.param_specs["frictionloss"]["lb"],
                        self.param_specs["frictionloss"].get("min", -np.inf),
                    )
                    self.param_specs["frictionloss"]["ub"] = min(
                        self.param_specs["frictionloss"]["ub"],
                        self.param_specs["frictionloss"].get("max", np.inf),
                    )
                frictionloss = normalize_range(
                    self.frictionloss,
                    self.param_specs["frictionloss"]["lb"],
                    self.param_specs["frictionloss"]["ub"],
                )
                param_list.append(frictionloss.flatten())

        if not get_name:
            self.param_inited = True
            # import ipdb; ipdb.set_trace()

    def set_params(self, params, pad_zeros=False):
        if "axis" in self.param_specs:
            if self.type == "hinge":
                self.axis = denormalize_range(params[:2],
                                              np.array([0, -2 * np.pi]),
                                              np.array([np.pi, 2 * np.pi]))
                params = params[2:]
            elif pad_zeros:
                params = params[2:]

        if self.actuator is not None:
            params = self.actuator.set_params(params)
        elif pad_zeros:
            params = params[1:]

        # Order of this matters!!! Should always be damping, aramature, stiffness (the order they are read)

        if "damping" in self.param_specs:
            self.damping = denormalize_range(
                params[[0]],
                self.param_specs["damping"]["lb"],
                self.param_specs["damping"]["ub"],
            )
            params = params[1:]

        if "armature" in self.param_specs:
            self.armature = denormalize_range(
                params[[0]],
                self.param_specs["armature"]["lb"],
                self.param_specs["armature"]["ub"],
            )
            params = params[1:]

        if "stiffness" in self.param_specs:
            self.stiffness = denormalize_range(
                params[[0]],
                self.param_specs["stiffness"]["lb"],
                self.param_specs["stiffness"]["ub"],
            )
            params = params[1:]

        if "frictionloss" in self.param_specs:
            self.frictionloss = denormalize_range(
                params[[0]],
                self.param_specs["frictionloss"]["lb"],
                self.param_specs["frictionloss"]["ub"],
            )
            params = params[1:]

        return params


class Geom:
    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib.get("name", "")
        self.type = node.attrib["type"]
        self.density = (parse_vec(node.attrib["density"]) /
                        1000 if "density" in node.attrib else np.array([1]))
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        # self.size = (
        #     parse_vec(node.attrib["size"]) if "size" in node.attrib else np.array([0])
        # )
        self.size = (parse_vec(node.attrib["size"])
                     if "size" in node.attrib else np.array([1, 1, 1]))
        if self.type == "box":
            self.start = self.end = self.pos = parse_vec(node.attrib["pos"])
            self.pos_delta = np.array([0, 0, 0])
            self.rot = parse_vec(node.attrib["quat"])
        elif self.type == "sphere":
            self.pos_delta = np.array([0, 0, 0])
            self.start = self.end = self.pos = parse_vec(node.attrib["pos"])
        elif self.type == "capsule":
            self.start, self.end = parse_fromto(node.attrib["fromto"])
        elif self.type == "mesh":
            self.start, self.end = body.pos.copy(), body.pos.copy()

        if self.local_coord:
            self.start += body.pos
            self.end += body.pos

        if body.bone_start is None:
            self.bone_start = self.start.copy()
            body.bone_start = self.bone_start.copy()
        else:
            self.bone_start = body.bone_start.copy()

        self.ext_start = np.linalg.norm(
            self.bone_start - self.start)  ## Geom extension from bone start

    def __repr__(self):
        return "geom_" + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg.get("geom_params", {}))
        for name, specs in self.param_specs.items():
            if "lb" in specs and isinstance(specs["lb"], list):
                if self.type == "box":
                    specs["lb"] = np.array([specs["lb"]] * 3)
                elif self.type == "capsule":
                    specs["lb"] = np.array(specs["lb"])
            if "ub" in specs and isinstance(specs["ub"], list):
                if self.type == "box":
                    specs["lb"] = np.array([specs["lb"]] * 3)
                elif self.type == "capsule":
                    specs["lb"] = np.array(specs["lb"])

    def update_start(self):
        if self.type == "capsule":
            vec = self.bone_start - self.end
            self.start = self.bone_start + vec * (self.ext_start /
                                                  np.linalg.norm(vec))

    def sync_node(self):
        # self.node.attrib['name'] = self.name
        self.node.attrib.pop("name", None)
        if not self.size is None:
            self.node.attrib["size"] = " ".join(
                [f"{x:.6f}".rstrip("0").rstrip(".") for x in self.size])
        self.node.attrib["density"] = " ".join(
            [f"{x * 1000:.6f}".rstrip("0").rstrip(".") for x in self.density])

        # if self.type == "capsule":
        #     start = self.start - self.body.pos if self.local_coord else self.start
        #     end = self.end - self.body.pos if self.local_coord else self.end
        #     self.node.attrib["fromto"] = " ".join(
        #         [
        #             f"{x:.6f}".rstrip("0").rstrip(".")
        #             for x in np.concatenate([start, end])
        #         ]
        #     )
        # elif self.type == "box" or self.type == "sphere":
        #     # self.node.attrib["pos"] = " ".join(
        #     #     [f"{x:.6f}".rstrip("0").rstrip(".") for x in self.pos + self.pos_delta]
        #     # )
        #     import ipdb; ipdb.set_trace()
        #     pass

    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if "size" in self.param_specs:
            if get_name:
                param_list.append("size")
            else:
                if (self.type == "capsule" or self.type == "box"
                        or self.type == "sphere" or self.type == "mesh"):
                    if not self.param_inited and self.param_specs["size"].get(
                            "rel", False):
                        self.param_specs["size"]["lb"] += self.size
                        self.param_specs["size"]["ub"] += self.size
                        self.param_specs["size"]["lb"] = max(
                            self.param_specs["size"]["lb"],
                            self.param_specs["size"].get("min", -np.inf),
                        )
                        self.param_specs["size"]["ub"] = min(
                            self.param_specs["size"]["ub"],
                            self.param_specs["size"].get("max", np.inf),
                        )
                    size = normalize_range(
                        self.size,
                        self.param_specs["size"]["lb"],
                        self.param_specs["size"]["ub"],
                    )
                    param_list.append(size.flatten())
                    if pad_zeros and self.type == "capsule":
                        param_list.append(
                            np.zeros(2))  # capsule has needs to be 3 for GNN

                elif pad_zeros:
                    param_list.append(np.zeros(self.size.shape))

        if "ext_start" in self.param_specs:
            if get_name:
                param_list.append("ext_start")
            else:
                if (self.type == "capsule" or self.type == "box"
                        or self.type == "sphere"):
                    if not self.param_inited and self.param_specs[
                            "ext_start"].get("rel", False):
                        self.param_specs["ext_start"]["lb"] += self.ext_start
                        self.param_specs["ext_start"]["ub"] += self.ext_start
                        self.param_specs["ext_start"]["lb"] = max(
                            self.param_specs["ext_start"]["lb"],
                            self.param_specs["ext_start"].get("min", -np.inf),
                        )
                        self.param_specs["ext_start"]["ub"] = min(
                            self.param_specs["ext_start"]["ub"],
                            self.param_specs["ext_start"].get("max", np.inf),
                        )
                    ext_start = normalize_range(
                        self.ext_start,
                        self.param_specs["ext_start"]["lb"],
                        self.param_specs["ext_start"]["ub"],
                    )
                    param_list.append(ext_start.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(self.size.shape))

        if "density" in self.param_specs:
            if get_name:
                param_list.append("density")
            else:
                if not self.param_inited and self.param_specs["density"].get(
                        "rel", False):
                    self.param_specs["density"]["lb"] += self.density
                    self.param_specs["density"]["ub"] += self.density
                    self.param_specs["density"]["lb"] = max(
                        self.param_specs["density"]["lb"],
                        self.param_specs["density"].get("min", -np.inf),
                    )
                    self.param_specs["density"]["ub"] = min(
                        self.param_specs["density"]["ub"],
                        self.param_specs["density"].get("max", np.inf),
                    )

                density = normalize_range(
                    self.density,
                    self.param_specs["density"]["lb"],
                    self.param_specs["density"]["ub"],
                )

                param_list.append(density.flatten())
                # if pad_zeros:
                #     param_list.append(np.zeros(self.density.shape))

        if "pos_delta" in self.param_specs:
            if get_name:
                param_list.append("pos_delta")
            else:
                if self.type == "box" or self.type == "sphere":
                    if not self.param_inited and self.param_specs[
                            "pos_delta"].get("rel", False):
                        self.param_specs["pos_delta"]["lb"] += self.density
                        self.param_specs["pos_delta"]["ub"] += self.density
                        self.param_specs["pos_delta"]["lb"] = max(
                            self.param_specs["pos_delta"]["lb"],
                            self.param_specs["pos_delta"].get("min", -np.inf),
                        )
                        self.param_specs["pos_delta"]["ub"] = min(
                            self.param_specs["pos_delta"]["ub"],
                            self.param_specs["pos_delta"].get("max", np.inf),
                        )

                    pos_delta = normalize_range(
                        self.pos_delta,
                        self.param_specs["pos_delta"]["lb"],
                        self.param_specs["pos_delta"]["ub"],
                    )

                    param_list.append(pos_delta.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(3))
        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if "size" in self.param_specs:
            if (self.type == "capsule" or self.type == "box"
                    or self.type == "sphere" or self.type == "mesh"):
                if len(self.size) == 1:
                    self.size = denormalize_range(
                        params[[0]],
                        self.param_specs["size"]["lb"],
                        self.param_specs["size"]["ub"],
                    )
                    params = params[1:]
                elif len(self.size) == 3:
                    self.size = denormalize_range(
                        np.array(params[:3]),
                        self.param_specs["size"]["lb"],
                        self.param_specs["size"]["ub"],
                    )
                    params = params[3:]

            elif pad_zeros:
                params = params[1:]
        if "ext_start" in self.param_specs:
            if self.type == "capsule" or self.type == "box" or self.type == "sphere":
                self.ext_start = denormalize_range(
                    params[[0]],
                    self.param_specs["ext_start"]["lb"],
                    self.param_specs["ext_start"]["ub"],
                )
                params = params[1:]
            elif pad_zeros:
                params = params[1:]

        if "density" in self.param_specs:
            if (self.type == "capsule" or self.type == "box"
                    or self.type == "sphere" or self.type == "mesh"):
                self.density = denormalize_range(
                    params[[0]],
                    self.param_specs["density"]["lb"],
                    self.param_specs["density"]["ub"],
                )
                params = params[1:]
            elif pad_zeros:
                params = params[1:]

        if "pos_delta" in self.param_specs:
            if self.type == "box" or self.type == "sphere":
                self.pos_delta = denormalize_range(
                    np.array(params[:3]),
                    self.param_specs["pos_delta"]["lb"],
                    self.param_specs["pos_delta"]["ub"],
                )
                params = params[3:]
            elif pad_zeros:
                params = params[3:]

        return params


class Actuator:
    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        self.cfg = joint.cfg
        self.joint_name = node.attrib["joint"]
        self.name = self.joint_name
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib["gear"])

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg.get("actuator_params", {}))
        for name, specs in self.param_specs.items():
            if "lb" in specs and isinstance(specs["lb"], list):
                specs["lb"] = np.array(specs["lb"])
            if "ub" in specs and isinstance(specs["ub"], list):
                specs["ub"] = np.array(specs["ub"])

    def sync_node(self):
        self.node.attrib["gear"] = f"{self.gear:.6f}".rstrip("0").rstrip(".")
        self.name = self.joint.name
        self.node.attrib["name"] = self.name
        self.node.attrib["joint"] = self.joint.name

    def get_params(self, param_list, get_name=False):
        if "gear" in self.param_specs:
            if get_name:
                param_list.append("gear")
            else:
                if not self.param_inited and self.param_specs["gear"].get(
                        "rel", False):
                    self.param_specs["gear"]["lb"] += self.gear
                    self.param_specs["gear"]["ub"] += self.gear
                    self.param_specs["gear"]["lb"] = max(
                        self.param_specs["gear"]["lb"],
                        self.param_specs["gear"].get("min", -np.inf),
                    )
                    self.param_specs["gear"]["ub"] = min(
                        self.param_specs["gear"]["ub"],
                        self.param_specs["gear"].get("max", np.inf),
                    )
                gear = normalize_range(
                    self.gear,
                    self.param_specs["gear"]["lb"],
                    self.param_specs["gear"]["ub"],
                )
                param_list.append(np.array([gear]))

        if not get_name:
            self.param_inited = True

    def set_params(self, params):
        if "gear" in self.param_specs:
            self.gear = denormalize_range(
                params[0].item(),
                self.param_specs["gear"]["lb"],
                self.param_specs["gear"]["ub"],
            )
            params = params[1:]
        return params


class Body:
    def __init__(self, node, parent_body, robot, cfg, new_body=False):
        self.node = node
        self.parent = parent_body
        self.new_body = new_body
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0
        self.robot = robot
        self.cfg = cfg
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        self.name = (node.attrib["name"] if "name" in node.attrib else
                     self.parent.name + f"_child{len(self.parent.child)}")
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib["pos"])

        if self.local_coord and parent_body is not None:
            self.pos += parent_body.pos

        if cfg.get("init_root_from_geom", False):
            self.bone_start = None if parent_body is None else self.pos.copy()
        else:
            self.bone_start = self.pos.copy()
        self.joints = [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] + \
                      [Joint(x, self) for x in node.findall('joint[@type="free"]')] + \
                     [Joint(x, self) for x in node.findall('freejoint')]

        # self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')]
        supported_geoms = self.cfg.get("supported_geoms", ["capsule", "box"])
        self.geoms = [
            Geom(x, self) for geom_type in supported_geoms
            for x in node.findall(f'geom[@type="{geom_type}"]')
        ]
        # self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] + [Geom(x, self) for x in node.findall('geom[@type="sphere"]')] +  [Geom(x, self) for x in node.findall('geom[@type="box"]')]

        self.parse_param_specs()
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None

    def __repr__(self):
        return "body_" + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg.get("body_params", {}))
        for name, specs in self.param_specs.items():
            if "lb" in specs and isinstance(specs["lb"], list):
                specs["lb"] = np.array(specs["lb"])
            if "ub" in specs and isinstance(specs["ub"], list):
                specs["ub"] = np.array(specs["ub"])
            if name == "bone_ang":
                specs["lb"] = np.deg2rad(specs["lb"])
                specs["ub"] = np.deg2rad(specs["ub"])

    def reindex(self):
        if self.parent is None:
            self.index = "0"
        else:
            ind = self.parent.child.index(self) + 1
            pname = "" if self.parent.index == "0" else self.parent.index
            self.index = str(ind) + pname
            if self.new_body:
                self.name = self.index

    def init(self):
        if len(self.child) > 0:
            bone_ends = [x.bone_start for x in self.child]
        else:
            bone_ends = [x.end for x in self.geoms]
        if len(bone_ends) > 0:
            self.bone_end = np.mean(np.stack(bone_ends), axis=0)
            self.bone_offset = self.bone_end - self.bone_start

    def get_actuator_name(self):
        for joint in self.joints:
            if joint.actuator is not None:
                return joint.actuator.name

    def get_joint_range(self):
        assert len(self.joints) == 1
        return self.joints[0].range

    def sync_node(self):
        pos = (self.pos - self.parent.pos
               if self.local_coord and self.parent is not None else self.pos)
        self.node.attrib["name"] = self.name
        self.node.attrib["pos"] = " ".join(
            [f"{x:.6f}".rstrip("0").rstrip(".") for x in pos])
        for idx, joint in enumerate(self.joints):
            joint.sync_node(rename=self.new_body, index=idx)
        for geom in self.geoms:
            geom.sync_node()

    def sync_geom(self):
        for geom in self.geoms:
            geom.bone_start = self.bone_start.copy()
            # geom.end = self.bone_end.copy()
            # geom.update_start()

    def sync_joint(self):
        if self.parent is not None:
            for joint in self.joints:
                joint.pos = self.pos.copy()

    def rebuild(self):
        if self.parent is not None:
            # self.bone_start = self.parent.bone_end.copy()
            self.pos = self.bone_start.copy()
        if self.bone_offset is not None:
            self.bone_end = self.bone_start + self.bone_offset
        if self.parent is None and self.cfg.get("no_root_offset", False):
            self.bone_end = self.bone_start
        self.sync_geom()
        self.sync_joint()

    def get_params(self,
                   param_list,
                   get_name=False,
                   pad_zeros=False,
                   demap_params=False):
        if self.bone_offset is not None and "offset" in self.param_specs:
            if get_name:
                if self.param_specs["offset"]["type"] == "xz":
                    param_list += ["offset_x", "offset_z"]
                elif self.param_specs["offset"]["type"] == "xy":
                    param_list += ["offset_x", "offset_y"]
                else:
                    param_list += ["offset_x", "offset_y", "offset_z"]
            else:
                if self.param_specs["offset"]["type"] == "xz":
                    offset = self.bone_offset[[0, 2]]
                elif self.param_specs["offset"]["type"] == "xy":
                    offset = self.bone_offset[[0, 1]]
                else:
                    offset = self.bone_offset
                if not self.param_inited and self.param_specs["offset"].get(
                        "rel", False):
                    self.param_specs["offset"]["lb"] += offset
                    self.param_specs["offset"]["ub"] += offset
                    self.param_specs["offset"]["lb"] = np.maximum(
                        self.param_specs["offset"]["lb"],
                        self.param_specs["offset"].get(
                            "min", np.full_like(offset, -np.inf)),
                    )
                    self.param_specs["offset"]["ub"] = np.minimum(
                        self.param_specs["offset"]["ub"],
                        self.param_specs["offset"].get(
                            "max", np.full_like(offset, np.inf)),
                    )
                offset = normalize_range(
                    offset,
                    self.param_specs["offset"]["lb"],
                    self.param_specs["offset"]["ub"],
                )
                param_list.append(offset.flatten())

        if self.bone_offset is not None and "bone_len" in self.param_specs:
            if get_name:
                param_list += ["bone_len"]
            else:
                bone_len = np.linalg.norm(self.bone_offset)
                if not self.param_inited and self.param_specs["bone_len"].get(
                        "rel", False):
                    self.param_specs["bone_len"]["lb"] += bone_len
                    self.param_specs["bone_len"]["ub"] += bone_len
                    self.param_specs["bone_len"]["lb"] = max(
                        self.param_specs["bone_len"]["lb"],
                        self.param_specs["bone_len"].get("min", -np.inf),
                    )
                    self.param_specs["bone_len"]["ub"] = min(
                        self.param_specs["bone_len"]["ub"],
                        self.param_specs["bone_len"].get("max", np.inf),
                    )
                bone_len = normalize_range(
                    bone_len,
                    self.param_specs["bone_len"]["lb"],
                    self.param_specs["bone_len"]["ub"],
                )
                param_list.append(np.array([bone_len]))

        if self.bone_offset is not None and "bone_ang" in self.param_specs:
            if get_name:
                param_list += ["bone_ang"]
            else:
                bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])
                if not self.param_inited and self.param_specs["bone_ang"].get(
                        "rel", False):
                    self.param_specs["bone_ang"]["lb"] += bone_ang
                    self.param_specs["bone_ang"]["ub"] += bone_ang
                    self.param_specs["bone_ang"]["lb"] = max(
                        self.param_specs["bone_ang"]["lb"],
                        self.param_specs["bone_ang"].get("min", -np.inf),
                    )
                    self.param_specs["bone_ang"]["ub"] = min(
                        self.param_specs["bone_ang"]["ub"],
                        self.param_specs["bone_ang"].get("max", np.inf),
                    )
                bone_ang = normalize_range(
                    bone_ang,
                    self.param_specs["bone_ang"]["lb"],
                    self.param_specs["bone_ang"]["ub"],
                )
                param_list.append(np.array([bone_ang]))

        for joint in self.joints:
            joint.get_params(param_list, get_name, pad_zeros)

        for geom in self.geoms:
            geom.get_params(param_list, get_name, pad_zeros)

        if not get_name:
            self.param_inited = True

        if demap_params and not get_name and len(param_list) > 0:
            params = self.robot.demap_params(np.concatenate(param_list))
            return params

    def set_params(self, params, pad_zeros=False, map_params=False):
        if map_params:
            params = self.robot.map_params(params)
        if self.bone_offset is not None and "offset" in self.param_specs:
            if self.param_specs["offset"]["type"] in {"xz", "xy"}:
                offset = denormalize_range(
                    params[:2],
                    self.param_specs["offset"]["lb"],
                    self.param_specs["offset"]["ub"],
                )
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                if self.param_specs["offset"]["type"] == "xz":
                    self.bone_offset[[0, 2]] = offset
                elif self.param_specs["offset"]["type"] == "xy":
                    self.bone_offset[[0, 1]] = offset
                params = params[2:]
            else:
                offset = denormalize_range(
                    params[:3],
                    self.param_specs["offset"]["lb"],
                    self.param_specs["offset"]["ub"],
                )
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                self.bone_offset[:] = offset
                params = params[3:]

        if self.bone_offset is not None and "bone_len" in self.param_specs:
            bone_len = denormalize_range(
                params[0].item(),
                self.param_specs["bone_len"]["lb"],
                self.param_specs["bone_len"]["ub"],
            )
            bone_len = max(bone_len, 1e-4)
            params = params[1:]
        elif self.bone_offset is not None:
            bone_len = np.linalg.norm(self.bone_offset)

        if self.bone_offset is not None and "bone_ang" in self.param_specs:
            bone_ang = denormalize_range(
                params[0].item(),
                self.param_specs["bone_ang"]["lb"],
                self.param_specs["bone_ang"]["ub"],
            )
            params = params[1:]
        elif self.bone_offset is not None:
            bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])

        if "bone_len" in self.param_specs or "bone_ang" in self.param_specs:
            self.bone_offset = np.array([
                bone_len * math.cos(bone_ang), 0, bone_len * math.sin(bone_ang)
            ])

        for joint in self.joints:
            params = joint.set_params(params, pad_zeros)
        for geom in self.geoms:
            params = geom.set_params(params, pad_zeros)
        # rebuild bone, geom, joint
        self.rebuild()
        return params


class SMPL_Robot:
    def __init__(self, cfg, data_dir="data/smpl"):
        self.bodies = []
        self.weight = 0
        self.height = 0
        self.cfg = cfg
        self.model_dirs = []
        self.param_mapping = cfg.get("param_mapping", "clip")
        self.smpl_model = cfg.get("model", "smpl")
        self.mesh = cfg.get("mesh", False)
        self.replace_feet = cfg.get("replace_feet", True)
        self.gender = cfg.get("gender", "neutral")
        self.flatfoot = cfg.get("flatfoot", True)
        self.upright_start = cfg.get("upright_start", True)
        if self.upright_start:
            print("!!!! Using modified SMPL starting pose !!!!")
        self.remove_toe = cfg.get("remove_toe", False)
        self.freeze_hand = cfg.get("freeze_hand", True)
        self.big_ankle = cfg.get("big_ankle", False)
        self.box_body = cfg.get("box_body", False)
        self.real_weight = cfg.get("real_weight", False)
        self.real_weight_porpotion_capsules = cfg.get("real_weight_porpotion_capsules", False)
        self.real_weight_porpotion_boxes = cfg.get("real_weight_porpotion_boxes", False)
        self.rel_joint_lm = cfg.get("rel_joint_lm", True)  # Rolling this out worldwide!!
        
        os.makedirs("/tmp/smpl/", exist_ok=True)
        self.masterfoot = cfg.get("masterfoot", False)
        self.param_specs = self.cfg.get("body_params", {})
        self.hull_dict = {}
        self.beta = (torch.zeros(
            (1, 10)).float() if self.smpl_model == "smpl" else torch.zeros(
                (1, 16)).float())

        if self.smpl_model == "smpl":
            self.smpl_parser_n = SMPL_Parser(model_path=data_dir,
                                             gender="neutral")
            self.smpl_parser_m = SMPL_Parser(model_path=data_dir,
                                             gender="male")
            self.smpl_parser_f = SMPL_Parser(model_path=data_dir,
                                             gender="female")
        elif self.smpl_model == "smplh":
            self.smpl_parser_n = SMPLH_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLH_Parser(model_path=data_dir,
                                              gender="male",
                                              use_pca=False,
                                              create_transl=False)
            self.smpl_parser_f = SMPLH_Parser(model_path=data_dir,
                                              gender="female",
                                              use_pca=False,
                                              create_transl=False)
        elif self.smpl_model == "smplx":
            self.smpl_parser_n = SMPLX_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLX_Parser(model_path=data_dir,
                                              gender="male",
                                              use_pca=False,
                                              create_transl=False)
            self.smpl_parser_f = SMPLX_Parser(model_path=data_dir,
                                              gender="female",
                                              use_pca=False,
                                              create_transl=False)

        self.load_from_skeleton()

        atexit.register(self.remove_geoms)

    def remove_geoms(self):
        while len(self.model_dirs) > 0:
            geom_dir = self.model_dirs.pop(0)
            if osp.isdir(geom_dir):
                shutil.rmtree(geom_dir, ignore_errors=True)

    def get_joint_vertices(self,
                           pose_aa,
                           th_betas=None,
                           th_trans=None,
                           gender=[0]):
        if gender[0] == 0:
            smpl_parser = self.smpl_parser_n
        elif gender[0] == 1:
            smpl_parser = self.smpl_parser_m
        elif gender[0] == 2:
            smpl_parser = self.smpl_parser_f
        else:
            print(gender)
            raise Exception("Gender Not Supported!!")
        vertices, joints = smpl_parser.get_joints_verts(pose=pose_aa,
                                                        th_betas=th_betas,
                                                        th_trans=th_trans)
        return vertices, joints

    def load_from_skeleton(
        self,
        betas=None,
        v_template=None,
        gender=[0],
        objs_info=None,
        obj_pose=None,
        params=None,
    ):

        self.tree = None  # xml tree

        if gender[0] == 0:
            self.smpl_parser = smpl_parser = self.smpl_parser_n
        elif gender[0] == 1:
            self.smpl_parser = smpl_parser = self.smpl_parser_m
        elif gender[0] == 2:
            self.smpl_parser = smpl_parser = self.smpl_parser_f
        else:
            print(gender)
            raise Exception("Gender Not Supported!!")

        if betas is None and self.beta is None:
            betas = (torch.zeros(
                (1, 10)).float() if self.smpl_model == "smpl" else torch.zeros(
                    (1, 16)).float())
        else:
            if params is None:
                self.beta = betas if not betas is None else self.beta
            else:
                # If params is not none, we need to set the beta first
                betas = self.map_params(betas)
                self.beta = torch.from_numpy(
                    denormalize_range(
                        betas.numpy().squeeze(),
                        self.param_specs["beta"]["lb"],
                        self.param_specs["beta"]["ub"],
                    )[None, ])
        if flags.debug:
            print(self.beta)

        ## Clear up beta for smpl and smplh
        if self.smpl_model == "smpl" and self.beta.shape[1] == 16:
            self.beta = self.beta[:, :10]
            # print(f"Incorrect shape size for {self.model}!!!")
        elif self.smpl_model == "smplh" and self.beta.shape[1] == 10:
            self.beta = torch.hstack([self.beta, torch.zeros((1, 6)).float()])
            # print(f"Incorrect shape size for {self.model}!!!")

        # self.remove_geoms()
        size_dict = {}
        if self.mesh:
            self.model_dirs.append(f"/tmp/smpl/{uuid.uuid4()}")
            # self.model_dirs.append(f"phc/data/assets/mesh/smpl/{uuid.uuid4()}")

            self.skeleton = SkeletonMesh(self.model_dirs[-1])
            zero_pose = torch.zeros((1, 72))
            if self.upright_start:
                zero_pose[0, :3] = torch.tensor([1.2091996, 1.2091996, 1.2091996])
            (
                verts,
                joints,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                joint_axes,
                joint_dofs,
                joint_range,
                contype,
                conaffinity,
            ) = (smpl_parser.get_mesh_offsets(
                zero_pose=zero_pose, betas=self.beta, flatfoot=self.flatfoot)
                 if self.smpl_model != "smplx" else
                 smpl_parser.get_mesh_offsets(v_template=v_template))

            if self.rel_joint_lm:
                if self.upright_start:
                    joint_range = update_joint_limits_upright(joint_range)
                else:
                    joint_range = update_joint_limits(joint_range)

            self.height = np.max(verts[:, 1]) - np.min(verts[:, 1])

            if (len(self.get_params(get_name=True)) > 1
                    and not params is None):  # ZL: dank code, very dank code
                self.set_params(params)
                size_dict = self.get_size()
                size_dict = self.enforce_length_size(size_dict)

                # Gear based size
                # gear_dict = self.get_gear()
                # for k, v in size_dict.items():
                #     for idx, suffix in enumerate(["_x", "_y", "_z"]):
                #         if k + suffix in gear_dict:
                #             size_dict[k][idx] *= gear_dict[k + suffix]
            self.hull_dict = get_joint_geometries(
                verts,
                joints,
                skin_weights,
                joint_names,
                scale_dict=size_dict,
                geom_dir=f"{self.model_dirs[-1]}/geom",
            )
            self.skeleton.load_from_offsets(
                joint_offsets,
                joint_parents,
                joint_axes,
                joint_dofs,
                joint_range,
                hull_dict = self.hull_dict,
                sites={},
                scale=1,
                equalities={},
                exclude_contacts=[["Chest", "L_Shoulder"],
                                  ["Chest", "R_Shoulder"],
                                  ["Chest", "R_Thorax"], ["Chest", "L_Thorax"],
                                  ['L_Hip', 'Pelvis'], ['R_Hip', 'Pelvis'],
                                  ['Torso', 'Pelvis'], ['L_Knee', 'L_Hip'],
                                  ['R_Knee', 'R_Hip'], ['Spine', 'Torso'],
                                  ['L_Ankle', 'L_Knee'], ['R_Ankle', 'R_Knee'],
                                  ['Chest', 'Spine'], ['L_Toe', 'L_Ankle'],
                                  ['R_Toe', 'R_Ankle'], ['Neck', 'Chest'],
                                  ['L_Thorax', 'Chest'], ['R_Thorax', 'Chest'],
                                  ['Head', 'Neck'], ['L_Shoulder', 'L_Thorax'],
                                  ['R_Shoulder', 'R_Thorax'],
                                  ['L_Elbow', 'L_Shoulder'],
                                  ['R_Elbow', 'R_Shoulder'],
                                  ['L_Wrist',
                                   'L_Elbow'], ['R_Wrist', 'R_Elbow'],
                                  ['L_Hand', 'L_Wrist'], ['R_Hand',
                                                          'R_Wrist']],
                collision_groups=contype,
                conaffinity=conaffinity,
                simple_geom=False,
                real_weight = self.real_weight,
                replace_feet=self.replace_feet,
            )
        else:
            self.skeleton = Skeleton()
            if self.smpl_model == "smpl":
                zero_pose = torch.zeros((1, 72))
            else:
                zero_pose = torch.zeros((1, 156))
            if self.upright_start:
                zero_pose[0, :3] = torch.tensor(
                    [1.2091996, 1.2091996, 1.2091996])

            verts, joints, skin_weights, joint_names, joint_offsets, parents_dict, channels, joint_range = smpl_parser.get_offsets(
                betas=self.beta, zero_pose=zero_pose)

            self.height = torch.max(verts[:, 1]) - torch.min(verts[:, 1])
            self.hull_dict = get_geom_dict(verts,
                                           joints,
                                           skin_weights,
                                           joint_names,
                                           scale_dict=size_dict)
            channels = ["x", "y", "z"]  # ZL: need to fix
            if self.rel_joint_lm:
                if self.upright_start:
                    joint_range = update_joint_limits_upright(joint_range)
                else:
                    joint_range = update_joint_limits(joint_range)

            # print("enforcing symmetry")
            # for k, v in joint_offsets.items():
            #     if k.startswith("L_"):
            #         right_val = joint_offsets[k.replace("L_", "R_")].copy()
            #         right_val[1] = -right_val[1]
            #         v[:] = right_val
            #     elif k in ["Torso", "Spine", "Chest", "Neck", "Head"]:
            #         v[1] = 0

            self.skeleton.load_from_offsets(joint_offsets,
                                            parents_dict,
                                            1,
                                            joint_range,
                                            self.hull_dict, {},
                                            channels, {},
                                            upright_start=self.upright_start,
                                            remove_toe=self.remove_toe,
                                            freeze_hand = self.freeze_hand, 
                                            box_body = self.box_body, 
                                            big_ankle=self.big_ankle,
                                            real_weight_porpotion_capsules=self.real_weight_porpotion_capsules,
                                            real_weight_porpotion_boxes = self.real_weight_porpotion_boxes,
                                            real_weight = self.real_weight)
        self.bodies = []  ### Cleaning bodies list
        self.bone_length = np.array([np.linalg.norm(i) for i in joint_offsets.values()])
        parser = XMLParser(remove_blank_text=True)

        self.tree = parse(
            BytesIO(
                # self.skeleton.write_str(
                #     bump_buffer=self.smpl_model == "smplh" or self.smpl_model == "smplx"
                # )
                self.skeleton.write_str(bump_buffer=True)),
            parser=parser,
        )

        self.local_coord = (self.tree.getroot().find(
            ".//compiler").attrib["coordinate"] == "local")
        root = self.tree.getroot().find("worldbody").find("body")

        self.add_body(root, None)
        self.init_bodies()
        self.param_names = self.get_params(get_name=True)
        self.init_params = self.get_params()
        self.init_tree = deepcopy(self.tree)
        if self.masterfoot:
            # self.add_masterfoot_capsule()
            self.add_masterfoot_box()

        # all_root = self.tree.getroot()
        # contact_node = Element("contact", {})
        # SubElement(contact_node,"exclude",{"name": "add01", "body1": "L_Shoulr", "body2": "Chest"},)
        # SubElement(contact_node,"exclude",{"name": "add02", "body1": "R_Shoulder", "body2": "Chest"},)
        # all_root.append(contact_node)

    def in_body(self, body, point):
        return in_hull(self.hull_dict[body]["norm_hull"], point)

    def project_to_body(self, body, point):
        in_body = self.in_body(body, point)
        norm_points = self.hull_dict[body]["norm_verts"]
        if not in_body[0]:
            return norm_points[np.argmin(
                np.linalg.norm(norm_points - point, axis=1))]
        else:
            return point.squeeze()

    def get_gear(self):
        actuator_dict = {}
        for body in self.bodies:
            for joint in body.joints:
                if not joint.actuator is None:
                    actuator_dict[joint.actuator.name] = joint.actuator.gear
        return actuator_dict

    def get_size(self):
        size_dict = {}
        for body in self.bodies:
            for geom in body.geoms:
                size_dict[body.name] = geom.size
        return size_dict

    def enforce_length_size(self, size_dict):
        distal_dir = {
            "Pelvis": 1,
            "L_Hip": 1,
            "L_Knee": 1,
            "L_Ankle": [1, 2],
            "L_Toe": [1, 2],
            "R_Hip": 1,
            "R_Knee": 1,
            "R_Ankle": [1, 2],
            "R_Toe": [1, 2],
            "Torso": 1,
            "Spine": 1,
            "Chest": 1,
            "Neck": 1,
            "Head": 1,
            "L_Thorax": 0,
            "L_Shoulder": 0,
            "L_Elbow": 0,
            "L_Wrist": 0,
            "L_Hand": 0,
            "R_Thorax": 0,
            "R_Shoulder": 0,
            "R_Elbow": 0,
            "R_Wrist": 0,
            "R_Hand": 0,
        }
        for k, v in size_dict.items():
            subset = np.array(v[distal_dir[k]])
            subset[subset <= 1] = 1
            v[distal_dir[k]] = subset

        return size_dict

    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self, self.cfg, new_body=False)
        self.bodies.append(body)

        for body_node_c in body_node.findall("body"):
            self.add_body(body_node_c, body)

    def init_bodies(self):
        for body in self.bodies:
            body.init()
        self.sync_node()

    def sync_node(self):
        for body in self.bodies:
            # body.reindex()
            body.sync_node()

    def add_masterfoot_box(self):
        self.joint_range_master = {
            "L_Toe_x": f"-0.1 0.1",
            "L_Toe_y": f"-45.0 45",
            "L_Toe_z": f"-10 10",
            "R_Toe_x": f"-0.1 0.1",
            "R_Toe_y": f"-45 45",
            "R_Toe_z": f"-10 10",
            "L_Toe_1_joint_0": f"-0.1 0.1",
            "L_Toe_1_joint_1": f"-45 45",
            "L_Toe_1_joint_2": f"-10 10",
            "R_Toe_1_joint_0": f"-0.1 0.1",
            "R_Toe_1_joint_1": f"-45 45",
            "R_Toe_1_joint_2": f"-10 10",
            "L_Toe_1_1_joint_0": f"-0.1 0.1",
            "L_Toe_1_1_joint_1": f"-45 45",
            "L_Toe_1_1_joint_2": f"-10 10",
            "R_Toe_1_1_joint_0": f"-0.1 0.1",
            "R_Toe_1_1_joint_1": f"-45 45",
            "R_Toe_1_1_joint_2": f"-10 10",
            "L_Toe_2_joint_0": f"-0.1 0.1",
            "L_Toe_2_joint_1": f"-45 45",
            "L_Toe_2_joint_2": f"-10 10",
            "R_Toe_2_joint_0": f"-0.1 0.1",
            "R_Toe_2_joint_1": f"-45 45",
            "R_Toe_2_joint_2": f"-10 10",
        }
        body_index = [3, 7]

        yellow_bone_length = 0.02
        yellow_toe_back = -0.02
        for idx in body_index:
            ankle2clone = body_ankle = self.bodies[idx]
            ankle_node = body_ankle.node
            flip_y = -1 if idx == 3 else 1

            for element in ankle_node.getiterator(): # remvoe the geom of the ankle
                if element.tag == "geom":
                    ankle_node.remove(element)
                    break

            child_pos = body_ankle.child[0].pos - body_ankle.pos
            body_ankle.geoms = [] # remove Ankel's original geom
            toe_x, toe_y, toe_z = child_pos
            ankles_zs, ankle_xs = self.hull_dict[body_ankle.name]['norm_verts'][:, 2], self.hull_dict[body_ankle.name]['norm_verts'][:, 0]
            ankle_y_max, ankle_y_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 1].max(), self.hull_dict[body_ankle.name]['norm_verts'][:, 1].min()
            ankle_z_max, ankle_z_min = ankles_zs.max(), ankles_zs.min()
            toe_y_max, toe_y_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 1].max(), self.hull_dict[body_ankle.name]['norm_verts'][:, 1].min()
            ankle_z_max_x = self.hull_dict[body_ankle.name]['norm_verts'][ankles_zs.argmax(), 0]
            if idx == 7:
                ankle_y_tgt, ankle_y_green = ankle_y_max, ankle_y_min
            else:
                ankle_y_tgt, ankle_y_green = ankle_y_min, ankle_y_max

            ankle_size = np.abs(toe_z - ankles_zs.min())

            ankle_z_min_neg = ankles_zs[ankle_xs < 0].min()
            ankle_z_span = ankle_z_max - ankle_z_min_neg
            ankle_y_span = ankle_y_max - ankle_y_min
            ankle_frac = 10
            green_y = (toe_y + ankle_y_tgt)/2 - ankle_y_span * 2/5 * flip_y

            attributes = {
                "size": f"{ankle_size}",
                "type": "box",
                "pos":
                f"{(toe_x + yellow_toe_back)/2 - (toe_x - yellow_toe_back)/4} {(green_y + ankle_y_tgt)/2 } {ankle_z_min_neg + ankle_z_span * 9/ankle_frac /2}",
                "size":
                f"{(toe_x - yellow_toe_back)/4} {np.abs((ankle_y_tgt - green_y)/2)} {ankle_z_span * 9 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            attributes = {
                "type": "box",
                "pos":
                f"{(toe_x + yellow_toe_back)/2 + (toe_x - yellow_toe_back)/4} {(green_y + ankle_y_tgt)/2 } {ankle_z_min_neg + ankle_z_span * 6/ankle_frac /2}",
                "size":
                f"{(toe_x - yellow_toe_back)/4} {np.abs((ankle_y_tgt - green_y)/2)} {ankle_z_span * 6 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))


            ############################################################ Nodes for red toes
            body_toe = body_ankle.child[0]
            body_toe_node = body_toe.node
            for element in body_toe_node.getiterator(): # remvoe the geom of the ankle
                if element.tag == "geom":
                    body_toe_node.remove(element)
                elif element.tag == "joint":
                    master_range = self.cfg.get("master_range", 30)
                    element.attrib["range"] = self.joint_range_master.get(element.attrib["name"],  f"-{master_range} {master_range}")


            toe_x_max, toe_x_min = self.hull_dict[body_toe.name]['norm_verts'][:, 0].max(), self.hull_dict[body_toe.name]['norm_verts'][:, 0].min()
            # toe_y_max, toe_y_min = self.hull_dict[body_toe.name]['norm_verts'][:, 1].max(), self.hull_dict[body_toe.name]['norm_verts'][:, 1].min()
            # toe_z_max, toe_z_min = self.hull_dict[body_toe.name]['norm_verts'][:, 2].max(), self.hull_dict[body_toe.name]['norm_verts'][:, 2].min()

            attributes = {
                "type": "box",
                "pos":
                f"{(toe_x_max)/4} {(green_y + ankle_y_tgt)/2  - toe_y} {ankle_z_min_neg + ankle_z_span * 4/ankle_frac /2- toe_z}",
                "size":
                f"{(toe_x_max)/4} {np.abs((ankle_y_tgt - green_y)/2)} {ankle_z_span * 4 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(body_toe_node, "geom", attributes)
            body_toe.geoms.append(Geom(geom_node, body_toe))

            attributes = {
                "type": "box",
                "pos":
                f"{(toe_x_max) * 3/4 } {(green_y + ankle_y_tgt)/2  - toe_y} {ankle_z_min_neg + ankle_z_span * 2/ankle_frac /2- toe_z}",
                "size":
                f"{(toe_x_max)/4} {np.abs((ankle_y_tgt - green_y)/2)} {ankle_z_span * 2 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(body_toe_node, "geom", attributes)
            body_toe.geoms.append(Geom(geom_node, body_toe))

            # #################################### First additional toe (green one)
            green_z = ankle_z_max - ankle_size - ankle_z_span * 4 / ankle_frac
            toe_y_max, toe_y_min = self.hull_dict[body_toe.name]['norm_verts'][:, 1].max(), self.hull_dict[body_toe.name]['norm_verts'][:, 1].min()
            pos = np.array([yellow_bone_length, green_y, green_z])


            green_toe_body = self.add_body_joint_and_actuator(body_ankle, body_toe, pos, index_name = "_1")
            attributes = {
                "type": "box",
                "pos":
                f"{(toe_x - yellow_bone_length)/2 } {  (ankle_y_green - green_y)/2 } {ankle_z_min_neg + ankle_z_span * 5/ankle_frac /2 - green_z}",
                "size":
                f"{np.abs(toe_x - yellow_bone_length)/2} {np.abs(green_y - ankle_y_green)/2} {ankle_z_span * 5 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(green_toe_body.node, "geom", attributes)
            green_toe_body.geoms.append(Geom(geom_node, green_toe_body))

            # #################################### Second additional toe (white one)
            pos = np.array([toe_x - yellow_bone_length, 0, toe_z - green_z])
            white_toe_body = self.add_body_joint_and_actuator(green_toe_body,
                                                          green_toe_body,
                                                          pos,
                                                          index_name="_1")

            attributes = {
                "type": "box",
                "pos":
                f"{(toe_x_max  ) * 2/5} {  (ankle_y_green - green_y)/2 } {ankle_z_min_neg + ankle_z_span * 2 /ankle_frac /2- toe_z}",
                "size":
                f"{(toe_x_max ) * 2/5} {np.abs(green_y - ankle_y_green)/2} {ankle_z_span * 2 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(white_toe_body.node, "geom", attributes)
            white_toe_body.geoms.append(Geom(geom_node, white_toe_body))


            #################################### Third additional toe (white one)
            ankle_x_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 0].min()
            pos = np.array(
                [yellow_toe_back, (green_y + ankle_y_tgt) / 2 , toe_z])

            while_toe_body = self.add_body_joint_and_actuator(body_ankle, body_toe, pos, index_name = "_2")
            attributes = {
                "type": "box",
                "pos":
                f"{(ankle_x_min - yellow_toe_back)/2} {0} {ankle_z_min_neg + ankle_z_span * 7 /ankle_frac /2- toe_z}",
                "size":
                f"{np.abs(ankle_x_min - yellow_toe_back)/2} {np.abs((ankle_y_tgt - green_y)/2)} {ankle_z_span * 7 / ankle_frac /2}",
                "quat": f"0 0 0 1",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(while_toe_body.node, "geom", attributes)
            while_toe_body.geoms.append(Geom(geom_node, while_toe_body))




        self.init_tree = deepcopy(self.tree)


    def add_masterfoot_capsule(self):
        masterfoot_v = self.cfg.get("masterfoot_v", 0)
        body_index = [3, 7]
        self.joint_range_master = {"L_Toe_x": f"-0.1 0.1", "L_Toe_y": f"-45 45", "L_Toe_z": f"-10 10"}
        yellow_bone_length = 0.04
        yellow_toe_back = -0.02
        for idx in body_index:
            ankle2clone = body_ankle = self.bodies[idx]
            ankle_node = body_ankle.node
            flip_y = -1 if idx == 3 else 1

            for element in ankle_node.getiterator(): # remvoe the geom of the ankle
                if element.tag == "geom":
                    ankle_node.remove(element)
                    break
            child_pos = body_ankle.child[0].pos - body_ankle.pos
            body_ankle.geoms = [] # remove Ankel's original geom
            toe_x, toe_y, toe_z = child_pos

            ankle_y_max, ankle_y_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 1].max(), self.hull_dict[body_ankle.name]['norm_verts'][:, 1].min()
            ankle_z_max, ankle_z_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 2].max(), self.hull_dict[body_ankle.name]['norm_verts'][:, 2].min()
            toe_y_max, toe_y_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 1].max(), self.hull_dict[body_ankle.name]['norm_verts'][:, 1].min()
            ankle_z_max_x = self.hull_dict[body_ankle.name]['norm_verts'][self.hull_dict[body_ankle.name]['norm_verts'][:, 2].argmax(), 0]
            if idx == 7:
                ankle_y_tgt = ankle_y_max
            else:
                ankle_y_tgt = ankle_y_min

            ankle_size = np.abs(toe_z - self.hull_dict[body_ankle.name]['norm_verts'][:, 2].min())
            ankle_z_span = ankle_z_max - ankle_z_min
            ankle_frac = 10

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {toe_y } {ankle_z_max - ankle_size - ankle_z_span * 2/ankle_frac} {toe_x} {toe_y } {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {(toe_y + ankle_y_tgt)/2} {ankle_z_max - ankle_size- ankle_z_span * 1/ankle_frac} {toe_x} {(toe_y + ankle_y_tgt)/2} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {ankle_y_tgt} {ankle_z_max - ankle_size- ankle_z_span * 0/ankle_frac} {toe_x} {ankle_y_tgt} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            ########################################################################
            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {toe_y} {ankle_z_max - ankle_size- ankle_z_span * 2/ankle_frac} {yellow_toe_back} {(toe_y + ankle_y_tgt)/2} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {(toe_y + ankle_y_tgt)/2} {ankle_z_max - ankle_size- ankle_z_span * 1/ankle_frac} {yellow_toe_back} {(toe_y + ankle_y_tgt)/2} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{ankle_z_max_x} {ankle_y_tgt} {ankle_z_max - ankle_size- ankle_z_span * 0/ankle_frac} {yellow_toe_back} {(toe_y + ankle_y_tgt)/2} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))

            ############################################################
            green_z = ankle_z_max - ankle_size - ankle_z_span * 4/ankle_frac

            attributes = {
                "size": f"{ankle_size}",
                "type": "capsule",
                "fromto":
                f"{yellow_bone_length} {(toe_y + ankle_y_tgt)/2 - 0.05 * flip_y} {green_z} {yellow_toe_back} {(toe_y + ankle_y_tgt)/2} {toe_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(ankle_node, "geom", attributes)
            body_ankle.geoms.append(Geom(geom_node, body_ankle))


            ############################################################ Nodes for red toes
            body_toe = body_ankle.child[0]
            body_toe_node = body_toe.node
            for element in body_toe_node.getiterator(): # remvoe the geom of the toe
                if element.tag == "geom":
                    body_toe_node.remove(element)
                elif element.tag == "joint":
                    master_range = self.cfg.get("master_range", 30)
                    element.attrib["range"] = self.joint_range_master.get(element.attrib["name"],  f"-{master_range} {master_range}")
                    print(element.attrib["range"])

            toe_x_max = self.hull_dict[body_toe.name]['norm_verts'][:, 0].max()
            toe_z_min_abs = np.abs(self.hull_dict[body_toe.name]['norm_verts'][:, 2].min())

            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto": f"{0.0} {0} {0} {toe_x_max} {0} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(body_toe_node, "geom", attributes)
            body_toe.geoms.append(Geom(geom_node, body_toe))

            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0.0} {(toe_y + ankle_y_tgt)/2 - toe_y} {0} {toe_x_max} {(toe_y + ankle_y_tgt)/2- toe_y} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(body_toe_node, "geom", attributes)
            body_toe.geoms.append(Geom(geom_node, body_toe))

            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0.0} {ankle_y_tgt- toe_y} {0} {toe_x_max} {ankle_y_tgt- toe_y} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(body_toe_node, "geom", attributes)
            body_toe.geoms.append(Geom(geom_node, body_toe))


            #################################### First additional toe (green one)
            toe_y_max, toe_y_min = self.hull_dict[body_toe.name]['norm_verts'][:, 1].max(), self.hull_dict[body_toe.name]['norm_verts'][:, 1].min()
            if idx == 7:
                toe_y_tgt = toe_y_min + toe_z_min_abs
            else:
                toe_y_tgt = toe_y_max - toe_z_min_abs

            pos = np.array([yellow_bone_length, (toe_y + ankle_y_tgt)/2 - 0.05 * flip_y, green_z])
            green_toe_body = self.add_body_joint_and_actuator(body_ankle, body_toe, pos, index_name = "_1")
            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0} {0} {0} {toe_x - yellow_bone_length} {0} {toe_z - green_z }",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(green_toe_body.node, "geom", attributes)
            green_toe_body.geoms.append(Geom(geom_node, green_toe_body))
            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0} {toe_y_tgt + toe_z_min_abs * flip_y} {toe_z - green_z} {toe_x - yellow_bone_length} {toe_y_tgt + toe_z_min_abs* flip_y} {toe_z - green_z}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(green_toe_body.node, "geom", attributes)
            green_toe_body.geoms.append(Geom(geom_node, green_toe_body))

            #################################### Second additional toe (white one)
            pos = np.array([toe_x - yellow_bone_length, 0, toe_z - green_z])
            white_toe_body = self.add_body_joint_and_actuator(green_toe_body,
                                                          green_toe_body,
                                                          pos,
                                                          index_name="_1")

            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0} {0} {0} {np.abs((toe_x_max - toe_x) * 3/4) } {0} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(white_toe_body.node, "geom", attributes)
            white_toe_body.geoms.append(Geom(geom_node, white_toe_body))

            attributes = {
                "size": f"{toe_z_min_abs }",
                "type": "capsule",
                "fromto":
                f"{0} {toe_y_tgt + toe_z_min_abs * flip_y} {0} {np.abs((toe_x_max - toe_x) * 3/4) } {toe_y_tgt + toe_z_min_abs * flip_y} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(white_toe_body.node, "geom", attributes)
            white_toe_body.geoms.append(Geom(geom_node, white_toe_body))

            #################################### Third additional toe (white one)
            ankle_x_min = self.hull_dict[body_ankle.name]['norm_verts'][:, 0].min()
            pos = np.array([yellow_toe_back, (toe_y + ankle_y_tgt)/2, toe_z])

            while_toe_body = self.add_body_joint_and_actuator(body_ankle, body_toe, pos, index_name = "_2")
            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0} {toe_z_min_abs} {0} {ankle_x_min - yellow_toe_back + toe_z_min_abs} {toe_z_min_abs} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(while_toe_body.node, "geom", attributes)
            while_toe_body.geoms.append(Geom(geom_node, while_toe_body))


            attributes = {
                "size": f"{toe_z_min_abs}",
                "type": "capsule",
                "fromto":
                f"{0} {-toe_z_min_abs} {0} {ankle_x_min - yellow_toe_back + toe_z_min_abs} {-toe_z_min_abs} {0}",
                "contype": "0",
                "conaffinity": "1",
            }
            geom_node = SubElement(while_toe_body.node, "geom", attributes)
            while_toe_body.geoms.append(Geom(geom_node, while_toe_body))

        # attributes = {
        #     "size": f"{toe_z_min}",
        #     "type": "capsule",
        #     "fromto":
        #     f"{0} {toe_y_tgt + toe_z_min * flip_y} {0} {toe_x - yellow_bone_length} {toe_y_tgt + toe_z_min* flip_y} {0}",
        #     "contype": "0",
        #     "conaffinity": "1",
        # }
        # geom_node = SubElement(while_toe_body.node, "geom", attributes)
        # while_toe_body.geoms.append(Geom(geom_node, while_toe_body))


        self.init_tree = deepcopy(self.tree)

    def add_body_joint_and_actuator(self, parent_body, body, pos, index_name = "_1"):
        body_node =body.node
        new_node = deepcopy(body_node)
        new_node.attrib['pos'] = f"{pos[0]} {pos[1]} {pos[2]}"
        new_node.attrib['name'] = body.name + index_name

        actu_node = parent_body.tree.getroot().find("actuator")
        if len(parent_body.child) > 0:
            # Recursively finding the last child to insert
            last_child = parent_body.child[-1]

            while len(last_child.child) > 0:
                last_child = last_child.child[-1]

            actu_insert_index = (actu_node.index(
                actu_node.find(f'motor[@joint="{last_child.joints[-1].name}"]')) + 1)
        else:
            actu_insert_index = (actu_node.index(
                actu_node.find(
                    f'motor[@joint="{parent_body.joints[-1].name}"]')) + 1)

        for bnode in body_node.findall("body"):
            body_node.remove(bnode)

        child_body = Body(
            new_node, parent_body, self, self.cfg, new_body = True
        )  # This needs to called after finding the actu_insert_index
        for element in new_node.getiterator():
            if element.tag == "geom":
                new_node.remove(element)
            if element.tag == "joint":
                master_range = self.cfg.get("master_range", 30)
                element.attrib["range"] = f"-{master_range} {master_range}"


        for joint in child_body.joints:
            new_actu_node = deepcopy(actu_node.find(f'motor[@joint="{joint.name}"]'))
            joint.node.attrib["range"] = self.joint_range_master.get(joint.node.attrib["name"],  f"-{master_range} {master_range}")
            new_actu_node.attrib['name'] += index_name

            actu_node.insert(actu_insert_index, new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)
            actu_insert_index += 1

            child_body.bone_offset = parent_body.bone_offset.copy()
            child_body.param_specs = deepcopy(parent_body.param_specs)
            child_body.param_inited = True
            child_body.rebuild()
            child_body.sync_node()
            parent_body.node.append(new_node)
            self.bodies.append(child_body)
            self.sync_node()
        return child_body

    def add_child_to_body(self, body):
        if body == self.bodies[0]:
            body2clone = body.child[0]
        else:
            body2clone = body
        child_node = deepcopy(body2clone.node)

        actu_node = body.tree.getroot().find("actuator")

        if len(body.child) > 0:
            # Recursively finding the last child to insert
            last_child = body.child[-1]
            while len(last_child.child) > 0:
                last_child = last_child.child[-1]

            actu_insert_index = (actu_node.index(
                actu_node.find(
                    f'motor[@joint="{last_child.joints[-1].name}"]')) + 1)
        else:
            actu_insert_index = (actu_node.index(
                actu_node.find(f'motor[@joint="{body.joints[-1].name}"]')) + 1)

        for bnode in child_node.findall("body"):
            child_node.remove(bnode)

        ######## Special case for the the foot, template geom   ##############
        child_body = Body(
            child_node, body, self, self.cfg, new_body=True
        )  # This needs to called after finding the actu_insert_index

        start = " ".join([
            f"{x:.6f}".rstrip("0").rstrip(".")
            for x in body.pos + np.array([0.0, -0.05, 0.05])
        ])

        attributes = {
            "size": "0.020 0.1000 0.0100",
            "type": "box",
            "pos": start,
            "quat": "0.7071 -0.7071 0.0000 0.0000",
            "contype": "0",
            "conaffinity": "1",
        }

        for element in child_node.getiterator():
            if element.tag == "geom":
                child_node.remove(element)

        geom_node = SubElement(child_node, "geom", attributes)
        child_body.geoms = [Geom(geom_node, child_body)]
        ######## Special case for the the foot, template geometry   ##############

        for joint in child_body.joints:
            new_actu_node = deepcopy(
                actu_node.find(f'motor[@joint="{joint.name}"]'))
            actu_node.insert(actu_insert_index, new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)
            actu_insert_index += 1
        child_body.bone_offset = body.bone_offset.copy()
        child_body.param_specs = deepcopy(body.param_specs)
        child_body.param_inited = True
        child_body.rebuild()
        child_body.sync_node()
        body.node.append(child_node)
        self.bodies.append(child_body)
        self.sync_node()

    def remove_body(self, body):
        body.node.getparent().remove(body.node)
        body.parent.child.remove(body)
        self.bodies.remove(body)
        actu_node = body.tree.getroot().find("actuator")
        for joint in body.joints:
            actu_node.remove(joint.actuator.node)
        del body
        self.sync_node()

    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def export_vis_string(self,
                          num=2,
                          smpl_robot=None,
                          fname=None,
                          num_cones=0):
        tree = deepcopy(self.tree)
        if smpl_robot is None:
            vis_tree = deepcopy(self.init_tree)
        else:
            vis_tree = deepcopy(smpl_robot.tree)

        # Removing actuators from the tree
        remove_elements = ["actuator", "contact", "equality"]
        for elem in remove_elements:
            node = tree.getroot().find(elem)
            if node is None:
                # print(f"has no elem: {elem}")
                pass
            else:
                node.getparent().remove(node)

        option = tree.getroot().find("option")
        flag = SubElement(option, "flag", {"contact": "disable"})
        option.addnext(Element("size", {"njmax": "1000"}))

        worldbody = tree.getroot().find("worldbody")
        asset = tree.getroot().find("asset")
        vis_worldbody = vis_tree.getroot().find("worldbody")

        geom_body = vis_worldbody.find("geom")

        vis_meshes = vis_tree.getroot().find("asset").findall("mesh")

        for i in range(1, num):
            vis_meshes = deepcopy(vis_meshes)
            for mesh in vis_meshes:
                old_file = mesh.attrib["file"]
                mesh.attrib["file"] = mesh.attrib["file"].replace(
                    ".stl", f"_{i}.stl")
                shutil.copy(old_file, mesh.attrib["file"])
                asset.append(mesh)

        body = vis_worldbody.find("body")
        for i in range(1, num):
            new_body = deepcopy(body)
            new_body.attrib["name"] = "%d_%s" % (i, new_body.attrib["name"])
            new_body.find("geom").attrib["rgba"] = "0.7 0 0 1"

            for node in new_body.findall(".//body"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
                node.find("geom").attrib["rgba"] = "0.7 0 0 1"
            for node in new_body.findall(".//joint"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
            for node in new_body.findall(".//site"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
            for node in new_body.findall(".//geom"):
                node.attrib["mesh"] = "%s_%d" % (node.attrib["mesh"], i)
            worldbody.append(new_body)

        for i in range(num_cones):
            body_node = Element("body", {"pos": "0 0 0"})
            geom_node = SubElement(
                body_node,
                "geom",
                {
                    "mesh": "cone",
                    "type": "mesh",
                    "rgba": "0.0 0.8 1.0 1.0"
                },
            )
            worldbody.append(body_node)
        for i in range(num_cones):
            worldbody.append(
                Element(
                    "geom",
                    {
                        "fromto": "0.0 0.0 0.0 0.0 1.0 0.0",
                        "rgba": "0.0 0.8 1.0 1.0",
                        "type": "cylinder",
                        "size": "0.0420",
                    },
                ))

        if fname is not None:
            print("Writing to file: %s" % fname)
            tree.write(fname, pretty_print=True)
        vis_str = etree.tostring(tree, pretty_print=True)
        return vis_str

    def export_vis_string_self(self,
                               num=3,
                               smpl_robot=None,
                               fname=None,
                               num_cones=0):
        # colors = ["0.8 0.6 .4 1", "0.7 0 0 1", "0.0 0.0 0.7 1"] * num
        colors = [
            f"{np.random.random():.3f} {np.random.random():.3f} {np.random.random():.3f} 1"
            for _ in range(num)
        ]
        # Export multiple vis strings
        tree = deepcopy(self.tree)
        if smpl_robot is None:
            vis_tree = deepcopy(self.init_tree)
        else:
            vis_tree = deepcopy(smpl_robot.tree)

        # Removing actuators from the tree
        remove_elements = ["actuator", "contact", "equality"]
        for elem in remove_elements:
            node = tree.getroot().find(elem)
            if node is None:
                # print(f"has no elem: {elem}")
                pass
            else:
                node.getparent().remove(node)

        option = tree.getroot().find("option")
        flag = SubElement(option, "flag", {"contact": "disable"})
        option.addnext(Element("size", {"njmax": "1000"}))

        worldbody = tree.getroot().find("worldbody")
        asset = tree.getroot().find("asset")
        vis_worldbody = vis_tree.getroot().find("worldbody")

        geom_body = vis_worldbody.find("geom")

        vis_meshes = vis_tree.getroot().find("asset").findall("mesh")
        for i in range(1, num):
            cur_meshes = deepcopy(vis_meshes)
            for mesh in cur_meshes:
                old_file = mesh.attrib["file"]
                mesh.attrib["file"] = mesh.attrib["file"].replace(
                    ".stl", f"_{i}.stl")
                shutil.copy(old_file, mesh.attrib["file"])
                asset.append(mesh)

        body = vis_worldbody.find("body")
        for i in range(1, num):
            new_body = deepcopy(body)
            new_body.attrib["name"] = "%d_%s" % (i, new_body.attrib["name"])
            new_body.find("geom").attrib["rgba"] = colors[i]

            for node in new_body.findall(".//body"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
                node.find("geom").attrib["rgba"] = colors[i]
            for node in new_body.findall(".//joint"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
            for node in new_body.findall(".//site"):
                node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
            for node in new_body.findall(".//geom"):
                node.attrib["mesh"] = "%s_%d" % (node.attrib["mesh"], i)
            worldbody.append(new_body)

        if fname is not None:
            print("Writing to file: %s" % fname)
            tree.write(fname, pretty_print=True)
        vis_str = etree.tostring(tree, pretty_print=True)
        return vis_str

    def demap_params(self, params):
        if not np.all((params <= 1.0) & (params >= -1.0)):
            print(f"param out of bounds: {params}")
        params = np.clip(params, -1.0, 1.0)
        if self.param_mapping == "sin":
            params = np.arcsin(params) / (0.5 * np.pi)
        return params

    def get_params(self, get_name=False):
        param_list = []
        if self.beta is not None and "beta" in self.param_specs:
            if get_name:
                param_list += ["beta"]
            else:
                beta = normalize_range(
                    self.beta.numpy().squeeze(),
                    self.param_specs["beta"]["lb"],
                    self.param_specs["beta"]["ub"],
                )
                param_list.append(beta)

        for body in self.bodies:
            body.get_params(param_list, get_name)

        if not get_name and len(param_list) > 0:
            params = np.concatenate(param_list)
            params = self.demap_params(params)
        else:
            params = np.array(param_list)
        return params

    def map_params(self, params):
        if self.param_mapping == "clip":
            params = np.clip(params, -1.0, 1.0)
        elif self.param_mapping == "sin":
            params = np.sin(params * (0.5 * np.pi))
        return params

    def set_params(self, params):
        # clip params to range
        params = self.map_params(params)

        if "beta" in self.param_specs:
            self.beta = torch.from_numpy(
                denormalize_range(
                    params[0:10],
                    self.param_specs["beta"]["lb"],
                    self.param_specs["beta"]["ub"],
                )[None, ])
            params = params[10:]

        for body in self.bodies:
            params = body.set_params(params)
        assert len(params) == 0  # all parameters need to be consumed!

        self.sync_node()

    def rebuild(self):
        for body in self.bodies:
            body.rebuild()
            body.sync_node()

    def get_gnn_edges(self):
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        edges = np.stack(edges, axis=1)
        return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": False,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": True,
        "box_body": False,
        "master_range": 50,
        "model": "smpl",
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }
    smpl_robot = SMPL_Robot(robot_cfg)
    params_names = smpl_robot.get_params(get_name=True)

    betas = torch.zeros(1, 16)
    # betas = torch.tensor([[5, -0.0670, -0.1558,  0.3380,  0.2385, -1.9571,  1.0835,-0.7026, -0.7133, -2.0366,  1.2852, -1.0309,  0.1753,  0.3070,  3.1381, -3.3069]])
    gender = [0]
    t0 = time.time()
    params = smpl_robot.get_params()

    smpl_robot.load_from_skeleton(betas=betas, objs_info=None, gender=gender)
    print(smpl_robot.height)

    smpl_robot.write_xml(f"phc/data/assets/mjcf/smpl_humanoid.xml")
    m = mujoco.MjModel.from_xml_path(f"phc/data/assets/mjcf/smpl_humanoid.xml")
    d = mujoco.MjData(m)
    # model = load_model_from_path(f"phc/data/assets/mjcf/amp_humanoid.xml")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)