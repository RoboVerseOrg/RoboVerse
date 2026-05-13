"""Blender-backend helpers shipped from the RoboVerse content layer.

Asset path resolution (``asset_paths``) and lightweight randomisation
samplers (``randomize``) used by the ``get_started/blender_demo.py``
example. The actual Blender scene mutators (HDRI / material / sun /
ground) live in ``metasim.sim.blender.utils`` because they call ``bpy``
directly and are part of the Blender backend's surface area.
"""
