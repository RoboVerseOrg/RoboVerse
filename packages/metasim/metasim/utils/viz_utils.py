"""This module provides utility functions for visualizing 3D point clouds and meshes using Plotly."""

import numpy as np
import plotly.graph_objects as go


def plot_point_cloud(pts, **kwargs):
    """Plot point cloud with plotly.

    Args:
        pts (nd.ndarray): The point cloud to visualize, in shape [N_points, 3]
        **kwargs: Additional keyword arguments to pass to the plotly scatter object

    Returns:
        go.Scatter3d: The plotly scatter object
    """
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", **kwargs)


def plot_mesh(mesh, pos=None, rot=None, color="lightblue", opacity=1.0, name="mesh", **kwargs):
    """Plots a 3D mesh using Plotly.

    Parameters:
    mesh (trimesh.Trimesh): The mesh object to plot, containing vertices and faces.
    pos (np.ndarray, optional): A 3-element array specifying the position to translate the mesh. Defaults to None.
    rot (np.ndarray, optional): A 3x3 rotation matrix to rotate the mesh. Defaults to None.
    color (str, optional): The color of the mesh. Defaults to "lightblue".
    opacity (float, optional): The opacity of the mesh. Defaults to 1.0.
    name (str, optional): The name of the mesh in the plot legend. Defaults to "mesh".
    **kwargs: Additional keyword arguments to pass to the plotly scatter object

    Returns:
    plotly.graph_objs.Mesh3d: A Plotly Mesh3d object representing the mesh.
    """
    verts = mesh.vertices
    if rot is not None:
        verts = np.matmul(rot, verts.T).T
    if pos is not None:
        verts = verts + pos[None]
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        **kwargs,
    )
