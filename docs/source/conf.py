import os
import re
import subprocess
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
html_show_copyright = True
html_show_sphinx = False
html_static_path = ["_static"]

### Autodoc configurations ###
autoclass_content = "class"
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "autosummary": True,
    "exclude-members": "__init__",
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
    ## GSNet
    "MinkowskiEngine",
    ## MetaSim dependencies
    "numpy",
    "torch",
    "torchvision",
    "imageio",
    "loguru",
    "gymnasium",
    "rich",
    "tyro",
    "tqdm",
    "huggingface_hub",
    "dill",
    "pytorch3d",
    # RoboVerse no longer vendors MetaSim; mock it for Pages docs builds that
    # install only docs/requirements.txt.
    "metasim",
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


def generate_task_markdown(app):
    script_path = os.path.join(app.srcdir, "dataset_benchmark", "tasks", "generate_task_docs.py")
    if not os.path.exists(script_path):
        # print(f"[Sphinx] ❌ generate_task_docs.py not found at {script_path}")
        return

    # print(f"[Sphinx] 🛠 Generating task markdown pages with: {script_path}")
    result = subprocess.run(
        [sys.executable, script_path], cwd=os.path.dirname(script_path), capture_output=True, text=True, check=False
    )


_NAVBAR_LABELS = {
    "metasim": {"short": "MetaSim", "full": "MetaSim User Guide"},
    "dataset_benchmark": {"short": "Dataset", "full": "Dataset and Benchmark"},
    "roboverse_learn": {"short": "Learn", "full": "RoboVerse Learn"},
    "API": {"short": "API", "full": "API"},
    "FAQ": {"short": "FAQ", "full": "Frequently Asked Questions"},
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
    app.connect("builder-inited", generate_task_markdown)
    app.connect("build-finished", normalize_navbar_labels)
