import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

__version__ = "0.1.17"

project = "MetaSim"
copyright = "2025, MetaSim Developers"
author = "MetaSim Developers"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_subfigure",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_new_tab_link",
]

myst_enable_extensions = ["colon_fence", "dollarmath", "tasklist"]
myst_heading_anchors = 4
templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/RoboVerse86.22.svg"
html_favicon = "_static/logo.png"

html_theme_options = {
    "show_nav_level": 1,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/RoboVerseOrg/MetaSim",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "RoboVerse",
            "url": "https://roboverse.wiki/",
            "icon": "fa-solid fa-globe",
        },
    ],
    "logo": {
        "image_dark": "_static/RoboVerse86.22.svg",
    },
    "navbar_center": ["navbar-nav"],
    "show_version_warning_banner": False,
    "sidebarwidth": "150px",
}

html_context = {
    "display_github": True,
    "github_user": "RoboVerseOrg",
    "github_repo": "MetaSim",
    "github_version": "main",
    "conf_py_path": "docs/source",
    "doc_path": "docs/source",
}
html_css_files = ["css/custom.css"]
html_show_copyright = True
html_show_sphinx = False
html_static_path = ["_static"]

autoclass_content = "class"
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "autosummary": True,
    "exclude-members": "__init__",
}
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autosummary_generate = True
autosummary_generate_overwrite = False

autodoc_mock_imports = [
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
    "mujoco",
    "mujoco_viewer",
    "dm_control",
    "isaacgym",
    "pybullet",
    "pybullet_data",
    "pyrep",
    "rlbench",
    "genesis",
    "sapien",
    "bpy",
    "mathutils",
    "MinkowskiEngine",
    "quaternion",
    "numpy",
    "numpy.quaternion",
    "torch",
    "torchvision",
    "imageio",
    "loguru",
    "gymnasium",
    "rich",
    "tyro",
    "huggingface_hub",
    "dill",
    "pytorch3d",
]


def skip_member(app, what, name, obj, skip, options):
    exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
    if name in exclusions:
        return True
    return None


_NAVBAR_LABELS = {
    "get_started": {"short": "Start", "full": "Getting Started"},
    "concept": {"short": "Concepts", "full": "Concepts"},
    "features": {"short": "Features", "full": "Features"},
    "developer_guide": {"short": "Develop", "full": "Development Guide"},
    "troubleshooting": {"short": "Help", "full": "Troubleshooting"},
    "API": {"short": "API", "full": "API"},
}


def _active_navbar_section(pagename):
    section = pagename.split("/", 1)[0]
    return section if section in _NAVBAR_LABELS else None


def _navbar_section_from_href(href, active_section):
    if href == "#":
        return active_section
    for section in _NAVBAR_LABELS:
        if href.endswith(f"{section}/index.html"):
            return section
    return None


def _replace_navbar_item_label(match, active_section):
    item_html = match.group(0)
    href_match = re.search(r'href="([^"]+)"', item_html)
    if href_match is None:
        return item_html

    section = _navbar_section_from_href(href_match.group(1), active_section)
    if section is None:
        return item_html

    label_kind = "full" if section == active_section else "short"
    label = _NAVBAR_LABELS[section][label_kind]
    return re.sub(r"(>)[^<>]+(</a>)", rf"\1{label}\2", item_html, count=1)


def _rewrite_navbar_labels(html, pagename):
    active_section = _active_navbar_section(pagename)

    def rewrite_navbar(match):
        navbar_html = match.group(0)
        return re.sub(
            r'<li class="nav-item[^"]*">.*?</li>',
            lambda item_match: _replace_navbar_item_label(item_match, active_section),
            navbar_html,
            flags=re.DOTALL,
        )

    return re.sub(
        r'<ul class="bd-navbar-elements navbar-nav">.*?</ul>',
        rewrite_navbar,
        html,
        flags=re.DOTALL,
    )


def normalize_navbar_labels(app, exception):
    if exception is not None or app.builder.name != "html":
        return

    outdir = os.path.abspath(app.outdir)
    for section in ("", *[f"{section}/" for section in _NAVBAR_LABELS]):
        html_path = os.path.join(outdir, section, "index.html")
        if not os.path.exists(html_path):
            continue

        pagename = "index" if section == "" else f"{section.rstrip('/')}/index"
        with open(html_path, encoding="utf-8") as f:
            html = f.read()

        rewritten = _rewrite_navbar_labels(html, pagename)
        if rewritten == html:
            continue

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(rewritten)


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("build-finished", normalize_navbar_labels)
