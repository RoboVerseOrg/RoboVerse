import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

__version__ = "0.1.0"
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RoboVerse"
copyright = "2025, RoboVerse Developers"
author = "RoboVerse Developers"
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_subfigure",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_new_tab_link",  # open external link in new tab, see https://github.com/ftnext/sphinx-new-tab-link
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ["colon_fence", "dollarmath", "tasklist"]
# https://github.com/executablebooks/MyST-Parser/issues/519#issuecomment-1037239655
myst_heading_anchors = 4

templates_path = ["_templates"]
# exclude_patterns = ["user_guide/reference/_autosummary/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/RoboVerse86.22.svg"
html_favicon = "_static/logo.png"

json_url = "_static/version_switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
html_theme_options = {
    "show_nav_level": 1,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/RoboVerseOrg/RoboVerse",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Website",
            "url": "https://roboverseorg.github.io/",
            "icon": "fa-solid fa-globe",
        },
    ],
    "logo": {
        "image_dark": "_static/RoboVerse86.22.svg",
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
    "show_version_warning_banner": False,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "sidebarwidth": "150px",
}

html_context = {
    "display_github": True,
    "github_user": "RoboVerseOrg",
    "github_repo": "RoboVerse",
    "github_version": "main",
    "conf_py_path": "docs/source",
    "doc_path": "docs/source",
}
html_css_files = [
    "css/custom.css",
]
html_static_path = ["_static"]

### Autodoc configurations ###
autoclass_content = "class"
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "autosummary": True,
}
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autosummary_generate = True  # automatically generate rst files when there are no ones
autosummary_generate_overwrite = False  # do not overwrite existing rst files

########################################################
## Mock out modules that are not available on RTD
########################################################

autodoc_mock_imports = [
    ## IsaacLab
    "matplotlib",
    "scipy",
    "carb",
    "warp",
    "pxr",
    "omni",
    "omni.kit",
    "omni.log",
    "omni.usd",
    "omni.client",
    "omni.physx",
    "omni.physics",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "omni.isaac.core",
    "omni.isaac.kit",
    "omni.isaac.cloner",
    "omni.isaac.urdf",
    "omni.isaac.version",
    "omni.isaac.motion_generation",
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaacsim",
    "isaacsim.core.api",
    "isaacsim.core.cloner",
    "isaacsim.core.version",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.gui.components",
    "isaacsim.asset.importer.urdf",
    "isaacsim.asset.importer.mjcf",
    "omni.syntheticdata",
    "omni.timeline",
    "omni.ui",
    "gym",
    "skrl",
    "stable_baselines3",
    "rsl_rl",
    "rl_games",
    "ray",
    "h5py",
    "hid",
    "prettytable",
    "tqdm",
    "tensordict",
    "trimesh",
    "toml",
    ## Mujoco
    "mujoco",
    "mujoco_viewer",
    "dm_control",
    ## IsaacGym
    "isaacgym",
    ## PyBullet
    "pybullet",
    "pybullet_data",
    ## PyRep
    "pyrep",
    "rlbench",
    ## Genesis
    "genesis",
    ## Sapien
    "sapien",
    ## Blender
    "bpy",
    "mathutils",
    ## MetaSim dependencies
    "numpy",
    "torch",
    "torchvision",
    "imageio",
    "loguru",
    "gymnasium",
    "rootutils",
    "rich",
    "tyro",
    "tqdm",
    "huggingface_hub",
    "dill",
    "pytorch3d",
]


########################################################
## Skip members from autodoc
########################################################


def skip_member(app, what, name, obj, skip, options):
    # List the names of the functions you want to skip here
    exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
    if name in exclusions:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
