import ast
import glob
import os
import re
import shutil

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_CFG_ROOT = os.path.abspath(os.path.join(CUR_DIR, "../../../../roboverse_pack/tasks"))
OUTPUT_DIR = os.path.join(CUR_DIR, "tasks_md")
DEFAULT_DESC = "No description provided."

# Where rendered task/training media live on disk and how they are referenced
# from the published docs. We only emit a <video>/<img> tag when the file
# actually exists under STANDARD_OUTPUT_DIR, so a missing asset shows a
# "coming soon" note instead of a broken player.
STANDARD_OUTPUT_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../_static/standard_output"))
STANDARD_OUTPUT_URL = "/roboverse/_static/standard_output"

# Documented entrypoint that instantiates any registered task from its config
# and rolls it out (get_started/9_cfg_task.py). It takes the task name plus a
# simulator backend; the robot comes from the task's own scenario.
RUN_ENTRYPOINT = "get_started/9_cfg_task.py"
# Backend preference when a task declares which platforms it supports.
SIM_PREFERENCE = ["mujoco", "isaacsim", "sapien3", "isaacgym", "genesis", "newton"]

GROUPS = [
    "Maniskill",
    "RLBench",
    "Libero",
    "Calvin",
    "Graspnet",
    "Gapartnet",
    "Arnold",
    "Unidoormanip",
    "Simpler",
    "Robosuite",
    "Metaworld",
    "Rlafford",
]
PLATFORMS = ["isaacsim", "mujoco", "isaacgym", "sapien3", "genesis", "newton"]


def parse_docstring_metadata(docstring: str):
    if not docstring:
        return {}

    meta = {}
    sections = re.split(r"^###\s*", docstring, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.strip().splitlines()
        if not lines:
            continue
        key_line = lines[0].strip().rstrip(":")
        key = key_line.lower()
        content_lines = lines[1:]

        if not content_lines:
            continue

        is_list = all(line.strip().startswith("- ") for line in content_lines if line.strip())
        if is_list:
            values = [line.strip()[2:].strip() for line in content_lines if line.strip()]
        else:
            values = "\n".join(content_lines).strip()

        meta[key] = values

    # === Occur  ✅，otherwise ❓ ===
    if "platforms" in meta:
        listed_platforms = set(p.lower() for p in meta["platforms"])  # lowercase normalize
        platform_status = {}
        for p in PLATFORMS:
            platform_status[p] = "✅" if p in listed_platforms else "❓"
        meta["platforms"] = platform_status

    # badges to  map
    if "badges" in meta and isinstance(meta["badges"], list):
        meta["badges"] = {b.strip(): True for b in meta["badges"] if isinstance(b, str)}

    # Keep only the raw video filename if the docstring declares one; the
    # actual path resolution + on-disk existence check happen in generate_md.
    if "video_url" in meta and isinstance(meta["video_url"], str):
        meta["video_url"] = os.path.basename(meta["video_url"].strip())

    return meta


def render_badges(meta):
    badge_definitions = {
        "dense": ("dense-reward", "https://img.shields.io/badge/dense-yes-brightgreen.svg"),
        "sparse": ("sparse-reward", "https://img.shields.io/badge/sparse-yes-brightgreen.svg"),
        "demos": ("demos", "https://img.shields.io/badge/demos-yes-brightgreen.svg"),
    }
    badges = meta.get("badges", {})
    display_lines = []
    definition_lines = []

    for key, (label, badge_url) in badge_definitions.items():
        if badges.get(key, False):
            badge_id = f"{label}-badge"
            display_lines.append(f"![{label}][{badge_id}]")
            definition_lines.append(f"[{badge_id}]: {badge_url}")

    return "\n".join(display_lines + [""] + definition_lines) if display_lines else ""


def choose_task_id(task_ids: list, title: str) -> str:
    """Pick the registered id that best matches a page title.

    A task file may register several variants (e.g. ``maniskill.stack_cube`` and
    ``maniskill.stack_cube_dense``). Prefer the one whose last segment equals the
    page title, otherwise the shortest last segment (the base variant rather than
    a ``_dense`` / ``_rgb`` derivative).
    """
    title_l = title.strip().lower()
    exact = [t for t in task_ids if t.split(".")[-1].lower() == title_l]
    if exact:
        return exact[0]
    return min(task_ids, key=lambda t: (len(t.split(".")[-1]), t))


def pick_sim(meta: dict) -> str:
    """Choose a backend for the run command from the task's declared platforms."""
    platforms = meta.get("platforms")
    if isinstance(platforms, dict):
        for sim in SIM_PREFERENCE:
            if platforms.get(sim) == "✅":
                return sim
    return "mujoco"


def resolve_media(group: str, filename: str):
    """Return (web_url, exists) for a media file under standard_output/tasks/<group>/."""
    if not filename:
        return None, False
    disk_path = os.path.join(STANDARD_OUTPUT_DIR, "tasks", group, filename)
    web_url = f"{STANDARD_OUTPUT_URL}/tasks/{group}/{filename}"
    return web_url, os.path.isfile(disk_path)


def render_video_section(meta: dict) -> str:
    group = meta.get("group", "Unknown")
    title = meta.get("title", "")
    filename = meta.get("video_url") or (f"{title}.mp4" if title else "")
    web_url, exists = resolve_media(group, filename)
    if not exists:
        return "## Task Video\n\n_Task rollout video coming soon._\n"
    return f"""## Task Video

<div style="display: flex; justify-content: center; margin-bottom: 20px;">
    <div style="width: 100%; max-width: 512px; text-align: center;">
        <video width="100%" autoplay loop muted playsinline style="border-radius: 0px;">
            <source src="{web_url}" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"></p>
    </div>
</div>
"""


def render_training_section(meta: dict) -> str:
    """Render a training-curve image if one exists on disk, else a placeholder."""
    group = meta.get("group", "Unknown")
    title = meta.get("title", "")
    web_url, exists = (None, False)
    for ext in ("png", "jpg", "gif", "mp4"):
        web_url, exists = resolve_media(group, f"{title}_train.{ext}")
        if exists:
            break
    if not exists:
        return "## Training Visualization\n\n_Training curve coming soon._\n"
    if web_url.endswith(".mp4"):
        media = f'<video width="100%" autoplay loop muted playsinline><source src="{web_url}" type="video/mp4"></video>'
    else:
        media = f'<img src="{web_url}" width="100%" alt="training curve">'
    return f"""## Training Visualization

<div style="display: flex; justify-content: center; margin-bottom: 20px;">
    <div style="width: 100%; max-width: 512px; text-align: center;">
        {media}
    </div>
</div>
"""


def render_run_section(meta: dict) -> str:
    # Only emit a run command when we resolved a real registered task id, so we
    # never print a bogus `--task <helper-module>` for non-task files.
    task_id = meta.get("task_id")
    if not task_id:
        return ""
    sim = pick_sim(meta)
    return f"""## How to Run

Instantiate and roll out this task with the standard task entrypoint
(the robot is taken from the task's own scenario):

```bash
python {RUN_ENTRYPOINT} --task {task_id} --sim {sim}
```

Swap `--sim` for any backend the task supports (see the platform table in
[Task Groups](../task_groups.md)).
"""


def generate_md(tid: str, meta: dict) -> str:
    title = meta.get("title", tid)
    desc = meta.get("description", DEFAULT_DESC)

    def format_list_field(value):
        if isinstance(value, list):
            return "\n" + "\n".join([f"- {item}" for item in value])
        elif isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                import ast

                items = ast.literal_eval(value)
                if isinstance(items, list):
                    return "\n" + "\n".join([f"- {item}" for item in items])
            except Exception:
                pass
        return value

    randoms = format_list_field(meta.get("randomizations", "None."))
    success = format_list_field(meta.get("success", "None."))
    official_url = meta.get("official_url", "")
    badge_section = render_badges(meta)
    official_link = f"\n**[🔗 Official Task Page]({official_url})**\n" if official_url else ""

    return f"""# {title}

{badge_section}
{official_link}

**Task Description:** {desc}

**Randomizations:**{randoms}

**Success Conditions:**{success}

{render_run_section(meta)}
{render_video_section(meta)}
{render_training_section(meta)}"""


def discover_all_tasks():
    task_meta = {}
    for py_path in glob.glob(os.path.join(TASK_CFG_ROOT, "*", "*.py")):
        if os.path.basename(py_path).startswith("_"):
            continue  # skip __init__.py and private infra modules (_passthrough, _locator, ...)

        try:
            with open(py_path) as f:
                doc = f.read()
            tree = ast.parse(doc)

            # Find the first class registered via @register_task: it carries the
            # canonical task id (its first decorator arg) and usually the
            # metadata docstring. Fall back to the first *Cfg class for the
            # docstring if the registered class has none.
            task_ids = []
            deco_node = None
            cfg_node = None
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if cfg_node is None and node.name.endswith("Cfg"):
                    cfg_node = node
                for deco in node.decorator_list:
                    if (
                        isinstance(deco, ast.Call)
                        and getattr(deco.func, "id", "") == "register_task"
                        and deco.args
                        and isinstance(deco.args[0], ast.Constant)
                        and isinstance(deco.args[0].value, str)
                    ):
                        task_ids.append(deco.args[0].value)
                        if deco_node is None:
                            deco_node = node
                        break

            docstring = ""
            if deco_node is not None:
                docstring = ast.get_docstring(deco_node) or ""
            if not docstring and cfg_node is not None:
                docstring = ast.get_docstring(cfg_node) or ""

            meta = parse_docstring_metadata(docstring)
            meta["_task_ids"] = task_ids

            # title
            title = meta.get("title") or os.path.splitext(os.path.basename(py_path))[0].replace("_cfg", "")
            meta["title"] = title
            safe_title = re.sub(r"\W+", "_", title.strip().lower())
            meta["md_path"] = f"tasks_md/{safe_title}.md"

            # canonical run-command task id (prefer the variant matching the title)
            task_ids = meta.pop("_task_ids", [])
            if task_ids:
                meta["task_id"] = choose_task_id(task_ids, title)

            # group
            group_raw = os.path.basename(os.path.dirname(py_path))
            if group_raw.lower() == "rlbench":
                meta["group"] = "RLBench"
            else:
                meta["group"] = group_raw.capitalize()

            task_meta[safe_title] = meta
        except Exception as e:
            # print(f"❌ Failed to process {py_path}: {e}")
            pass
    return task_meta


def build_task_docs(TASK_REGISTRY):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for tid, meta in TASK_REGISTRY.items():
        path = os.path.join(OUTPUT_DIR, os.path.basename(meta["md_path"]))
        with open(path, "w") as f:
            f.write(generate_md(tid, meta))
        # print(f"✅ {path} written.")


def generate_task_groups_md(TASK_REGISTRY, output_path=None):
    if output_path is None:
        output_path = os.path.join(CUR_DIR, "task_groups.md")

    lines = ["# Task Group\n"]

    for i, group in enumerate(GROUPS):
        lines.append(f"## {group}\n")

        # HTML table
        lines.append(
            '<table style="table-layout: fixed; width: 100%; border-collapse: collapse; margin-bottom: 24px;">'
        )

        # Header
        lines.append(
            "<thead><tr>"
            "<th style='width: 30%; word-wrap: break-word; text-align: left; padding: 8px; border-bottom: 2px solid #ccc; font-size: 16px;'>Task / Robot</th>"
            + "".join([
                f"<th style='width: {int(70 / len(PLATFORMS))}%; text-align: center; padding: 8px; border-bottom: 2px solid #ccc; font-size: 16px;'>{plat}</th>"
                for plat in PLATFORMS
            ])
            + "</tr></thead>"
        )

        lines.append("<tbody>")

        group_tasks = [(tid, meta) for tid, meta in TASK_REGISTRY.items() if meta.get("group") == group]
        group_tasks.sort(key=lambda x: x[1].get("title", x[0]))

        for tid, meta in group_tasks:
            task_name = meta.get("title", tid)

            if len(task_name) > 25 and "_" in task_name:
                task_name = task_name.replace("_", "_<br>", 1)

            # md_path = meta.get("md_path", f"tasks_md/{tid}.md")

            # row = f"<td style='padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'><a href='{md_path}'>{task_name}</a></td>"
            html_path = meta.get("md_path", f"tasks_md/{tid}.md").replace(".md", ".html")
            row = f"<td style='padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'><a href='{html_path}'>{task_name}</a></td>"

            for plat in PLATFORMS:
                status = meta.get("platforms", {}).get(plat, "❓")
                row += f"<td style='text-align: center; padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'>{status}</td>"

            lines.append(f"<tr>{row}</tr>")

        lines.append("</tbody></table>")

        if i != len(GROUPS) - 1:
            lines.append("\n---\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    # print(f"✅ {output_path} generated.")


if __name__ == "__main__":
    TASK_REGISTRY = discover_all_tasks()
    build_task_docs(TASK_REGISTRY)
    generate_task_groups_md(TASK_REGISTRY)
