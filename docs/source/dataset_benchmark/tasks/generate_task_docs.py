import ast
import glob
import os
import re
import shutil

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_CFG_ROOT = os.path.abspath(os.path.join(CUR_DIR, "../../../../metasim/cfg/tasks"))
OUTPUT_DIR = os.path.join(CUR_DIR, "tasks_md")
VIDEO_BASE = "https://videos.example.com"
DEFAULT_DESC = "No description provided."

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
PLATFORMS = ["isaaclab", "mujoco", "isaacgym", "sapien3", "genesis"]


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

    # === 重点处理 platforms: 出现为 ✅，未出现为 ❓ ===
    if "platforms" in meta:
        listed_platforms = set(p.lower() for p in meta["platforms"])  # lowercase normalize
        platform_status = {}
        for p in PLATFORMS:
            platform_status[p] = "✅" if p in listed_platforms else "❓"
        meta["platforms"] = platform_status

    # badges 转布尔 map
    if "badges" in meta and isinstance(meta["badges"], list):
        meta["badges"] = {b.strip(): True for b in meta["badges"] if isinstance(b, str)}

    # 自动 video_url（如果没写）
    if "video_url" not in meta and "title" in meta and "group" in meta:
        meta["video_url"] = f"https://roboverse.wiki/_static/standard_output/tasks/{meta['group']}/{meta['title']}.mp4"

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
    video_url = meta.get("video_url") or f"{VIDEO_BASE}/{tid}.mp4"
    poster_url = meta.get("poster_url", "")
    official_url = meta.get("official_url", "")
    badge_section = render_badges(meta)
    official_link = f"\n**[🔗 Official Task Page]({official_url})**\n" if official_url else ""

    return f"""# {title}

{badge_section}
{official_link}

**Task Description:** {desc}

**Randomizations:**{randoms}

**Success Conditions:**{success}

<div style="display: flex; justify-content: center;">
  <video preload="none" controls width="100%" style="max-width: min(100%, 512px);">
    <source src="{video_url}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
"""


def discover_all_tasks():
    task_meta = {}
    for py_path in glob.glob(os.path.join(TASK_CFG_ROOT, "*", "*.py")):
        if os.path.basename(py_path).startswith("__"):
            continue  # 跳过 __init__.py

        try:
            with open(py_path) as f:
                doc = f.read()
            tree = ast.parse(doc)

            # docstring
            docstring = ""
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name.endswith("Cfg"):
                    docstring = ast.get_docstring(node)
                    break

            meta = parse_docstring_metadata(docstring or "")

            # title
            title = meta.get("title") or os.path.splitext(os.path.basename(py_path))[0].replace("_cfg", "")
            meta["title"] = title
            safe_title = re.sub(r"\W+", "_", title.strip().lower())
            meta["md_path"] = f"tasks_md/{safe_title}.md"

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

        # HTML 表格开头，添加样式美化
        lines.append(
            '<table style="table-layout: fixed; width: 100%; border-collapse: collapse; margin-bottom: 24px;">'
        )

        # 表头
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

        # 获取该 group 的所有任务
        group_tasks = [(tid, meta) for tid, meta in TASK_REGISTRY.items() if meta.get("group") == group]
        group_tasks.sort(key=lambda x: x[1].get("title", x[0]))

        for tid, meta in group_tasks:
            task_name = meta.get("title", tid)
            # 太长的 task_name 加换行
            if len(task_name) > 25 and "_" in task_name:
                task_name = task_name.replace("_", "_<br>", 1)

            # md_path = meta.get("md_path", f"tasks_md/{tid}.md")

            # 第一列：任务名链接
            # row = f"<td style='padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'><a href='{md_path}'>{task_name}</a></td>"
            html_path = meta.get("md_path", f"tasks_md/{tid}.md").replace(".md", ".html")
            row = f"<td style='padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'><a href='{html_path}'>{task_name}</a></td>"
            # 平台支持列
            for plat in PLATFORMS:
                status = meta.get("platforms", {}).get(plat, "❓")
                row += f"<td style='text-align: center; padding: 8px; font-size: 15px; border-bottom: 1px solid #eee;'>{status}</td>"

            lines.append(f"<tr>{row}</tr>")

        lines.append("</tbody></table>")

        if i != len(GROUPS) - 1:
            lines.append("\n---\n")

    # 写入 markdown 文件
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    # print(f"✅ {output_path} generated.")


if __name__ == "__main__":
    TASK_REGISTRY = discover_all_tasks()
    build_task_docs(TASK_REGISTRY)
    generate_task_groups_md(TASK_REGISTRY)
