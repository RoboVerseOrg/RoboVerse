#!/usr/bin/env bash
# publish_report.sh — RoboVerse → hub publish wrapper (direct-commit flow).
#
# 2026-05 起 hub 流程从 sync_to_github.py + mirror + symlink 换成「直接把真实
# 文件 commit 进 hub repo + git push」。本 wrapper 是该 flow 的项目侧入口。
#
# 这个脚本做的事:
#   1. (可选) 重跑 reports/roboverse_overview/_build_overview.py
#   2. git -C $HUB_DIR pull --rebase origin main   (拿别的机器的更新)
#   3. rsync 把本项目的 report dir 拷进 $HUB_DIR/projects/$PROJECT_SLUG/$REPORT_SLUG
#      + 把 overview HTML 引用的 mp4/png 资源 dir (mjlab/maniskill/robotwin runs/)
#      拷进 hub 对应位置 (overview HTML 通过 ../<slug>/runs/... 引用)
#   4. python 改 $HUB_DIR/manifest.json:只 filter 出本项目 slug 的旧 entry, 写
#      入新 entry。其他项目原封不动
#   5. git -C $HUB_DIR add manifest.json projects/$PROJECT_SLUG → 有 diff 就
#      commit + push (--no-push 跳过 push, 留 commit)
#
# 用法:
#   bash tools/publish_report.sh                          # 全跑 + push
#   bash tools/publish_report.sh --no-push                # 本地预览
#   bash tools/publish_report.sh --no-build               # 跳过 _build_overview.py 重生成
#   bash tools/publish_report.sh -m "comment"             # 自定义 commit msg
#
# 注意事项:
#   - 这个 wrapper 只动 $HUB_DIR/projects/$PROJECT_SLUG/ 和 manifest.json 里
#     自己的 entry。看到 diff 动了别的项目目录 = bug, 中止
#   - git push 失败 90% 是别人 push 了新东西。重跑就好 (开头会 pull --rebase)
#   - 不再用 sync_to_github.py / add_report.sh / publish.sh / mirror clone

set -euo pipefail

# ---------------------------------------------------------------------- CONFIG
HUB_DIR="${RESEARCH_REPORT_HUB:-/home/ghr/projects/research_report}"
PROJECT_SLUG="roboverse"
REPORT_SLUG="roboverse_overview"

REPORT_TITLE="⭐ RoboVerse 项目主报告 (维护 + 集成 + 路线, 26 页)"
REPORT_DATE="$(date +%Y-%m-%d)"
REPORT_TAGS=("总览" "维护" "集成" "roadmap" "中文" "多页" "mjlab" "maniskill" "robotwin" "g1" "rl")
REPORT_SUMMARY="26 页项目主报告。维护 (架构 / 状态 / 积压 / 修复 / 技术债 / 兼容 / 测试 / 2 次 audit) + 集成 (mjlab 含 G1 真 walking + 12 任务物理对齐 + policy-rollout side-by-side, ManiSkill 4 任务 + recon, RoboTwin ALOHA + 50 任务清单) + 横向主题 (parity / RL policy / 痛点) + 路线图 (P0-P3, 0.3→1.0)。"

PROJECT_NAME="RoboVerse"
PROJECT_DESC="RoboVerse + MetaSim 机器人仿真框架的 audit 和 task-coverage reports"
PROJECT_COLOR="#e8806a"

REPORTS_ROOT="/home/ghr/projects/RoboVerse/reports"
SOURCE_REPORT_DIR="$REPORTS_ROOT/$REPORT_SLUG"

# Overview HTML 引用兄弟目录的 mp4 (via `../<sub>/runs/...`); hub 需要这些 mp4
# 真实存在在 projects/$PROJECT_SLUG/<sub>/runs/。只拷 runs/, 不拷 HTML (老
# sub-report HTML 已经合并进 overview)。
ASSET_SUBDIRS=(
    "mjlab_integration/runs"
    "maniskill_integration/runs"
    "robotwin_integration/runs"
    "mplan_render/runs"
    "mplan_render/curves"
)

# -------------------------------------------------------------------- ARGS
DO_BUILD=1
DO_PUSH=1
MSG=""
REVIEW_ENTRY=""
REVIEW_LINK=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) DO_BUILD=0; shift;;
        --no-push) DO_PUSH=0; shift;;
        -m|--message) MSG="$2"; shift 2;;
        -m=*|--message=*) MSG="${1#*=}"; shift;;
        --review) REVIEW_ENTRY="$2"; shift 2;;
        --review=*) REVIEW_ENTRY="${1#*=}"; shift;;
        --review-link) REVIEW_LINK="$2"; shift 2;;
        --review-link=*) REVIEW_LINK="${1#*=}"; shift;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

[ -z "$MSG" ] && MSG="$PROJECT_SLUG: $(date +%Y-%m-%d_%H:%M) refresh"
[ -z "$REVIEW_ENTRY" ] && REVIEW_ENTRY="$MSG"
[ -z "$REVIEW_LINK" ] && REVIEW_LINK="index.html"

# The review table lives in roboverse_overview/index.html. Links to *sibling*
# reports (mjlab_integration/, maniskill_integration/, …) must go up one level
# (../) to resolve on the deployed hub. Auto-prefix if the caller passed a bare
# sibling-report path so the links aren't dead on the website.
case "$REVIEW_LINK" in
    mjlab_integration/*|maniskill_integration/*|robotwin_integration/*|menagerie_integration/*|mplan_render/*)
        REVIEW_LINK="../$REVIEW_LINK" ;;
esac

# Prepend a "Need to review" row to the index's review table BEFORE building.
REVIEW_NEW_ROW="    <tr><td><strong>$(date +"%Y-%m-%d %H:%M")</strong></td><td>${REVIEW_ENTRY}</td><td><a href=\"${REVIEW_LINK}\">link</a></td></tr>"
python3 - <<PY
import re
from pathlib import Path
fp = Path("${SOURCE_REPORT_DIR:-/home/ghr/projects/RoboVerse/reports/roboverse_overview}/_build_overview.py")
src = fp.read_text()
marker = '<tbody>\n    <tr><td><strong>'  # first row inside review table
new_row = '''${REVIEW_NEW_ROW}
'''
if marker not in src:
    print("[publish_report] review-marker not found in builder; skipping prepend")
else:
    # Insert immediately AFTER the <tbody> line so newest is on top
    new_src = src.replace('<tbody>\n    <tr><td><strong>2026',
                          '<tbody>\n' + new_row + '    <tr><td><strong>2026',
                          1)
    if new_src != src:
        fp.write_text(new_src)
        print("[publish_report] prepended review row")
    else:
        print("[publish_report] review row already up-to-date (no change)")
PY

# -------------------------------------------------------------------- CHECKS
if [ ! -d "$HUB_DIR/.git" ]; then
    echo "ERROR: hub dir is not a git repo: $HUB_DIR" >&2
    echo "       set RESEARCH_REPORT_HUB or clone the hub first" >&2
    exit 1
fi
if [ ! -d "$SOURCE_REPORT_DIR" ]; then
    echo "ERROR: source report dir missing: $SOURCE_REPORT_DIR" >&2
    exit 1
fi
if [ ! -f "$SOURCE_REPORT_DIR/index.html" ]; then
    echo "ERROR: source report dir has no index.html: $SOURCE_REPORT_DIR" >&2
    exit 1
fi

# -------------------------------------------------------------------- BUILD
if [ "$DO_BUILD" = "1" ]; then
    for builder in "$SOURCE_REPORT_DIR"/_build*.py; do
        [ -f "$builder" ] || continue
        echo "==> rebuild via $(basename "$builder")"
        (cd "$SOURCE_REPORT_DIR" && python3 "$(basename "$builder")")
    done
fi

# -------------------------------------------------------------------- PULL HUB
echo "==> git pull --rebase $HUB_DIR"
git -C "$HUB_DIR" pull --rebase --autostash origin main

# -------------------------------------------------------------------- RSYNC REPORT
PROJECT_HUB_DIR="$HUB_DIR/projects/$PROJECT_SLUG"
REPORT_HUB_DIR="$PROJECT_HUB_DIR/$REPORT_SLUG"
mkdir -p "$PROJECT_HUB_DIR"

# Make sure the path isn't a leftover symlink from the old flow.
if [ -L "$REPORT_HUB_DIR" ]; then
    echo "==> removing leftover symlink: $REPORT_HUB_DIR"
    rm "$REPORT_HUB_DIR"
fi

echo "==> rsync $SOURCE_REPORT_DIR/  →  $REPORT_HUB_DIR/"
mkdir -p "$REPORT_HUB_DIR"
# --copy-links: dereference any symlinks inside the report (defensive; overview
# no longer has internal symlinks but old runs may have one).
# --exclude='*.pyc' / __pycache__: don't ship junk.
rsync -a --delete --delete-excluded \
    --copy-links \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='_cache/' \
    --exclude='_build*.py' \
    --exclude='*.npz' \
    --exclude='*.mjb' \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='*.pth' \
    "$SOURCE_REPORT_DIR/" "$REPORT_HUB_DIR/"

for sub in "${ASSET_SUBDIRS[@]}"; do
    src="$REPORTS_ROOT/$sub"
    dst="$PROJECT_HUB_DIR/$sub"
    if [ ! -d "$src" ]; then
        echo "WARN: asset subdir missing: $src — skipping"
        continue
    fi
    # Old flow had a sibling symlink (e.g. projects/roboverse/mjlab_integration -> ...).
    # In the new flow we put real files; remove the symlink first if it exists.
    parent_link="$PROJECT_HUB_DIR/$(dirname "$sub")"
    if [ -L "$parent_link" ]; then
        echo "==> removing leftover symlink: $parent_link"
        rm "$parent_link"
    fi
    echo "==> rsync $src/  →  $dst/"
    mkdir -p "$dst"
    rsync -a --delete --delete-excluded --copy-links \
        --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='*.npz' --exclude='*.mjb' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.pth' \
        "$src/" "$dst/"
done

# -------------------------------------------------------------------- MANIFEST
echo "==> patch manifest.json (slug=$PROJECT_SLUG, report=$REPORT_SLUG)"
# Pass config via env so we don't have to quote-shell-escape into python.
export PUB_HUB_DIR="$HUB_DIR"
export PUB_PROJECT_SLUG="$PROJECT_SLUG"
export PUB_PROJECT_NAME="$PROJECT_NAME"
export PUB_PROJECT_DESC="$PROJECT_DESC"
export PUB_PROJECT_COLOR="$PROJECT_COLOR"
export PUB_REPORT_SLUG="$REPORT_SLUG"
export PUB_REPORT_TITLE="$REPORT_TITLE"
export PUB_REPORT_DATE="$REPORT_DATE"
export PUB_REPORT_SUMMARY="$REPORT_SUMMARY"
export PUB_REPORT_TAGS_CSV="$(IFS=,; echo "${REPORT_TAGS[*]}")"

python3 - <<'PY'
import json, os
from pathlib import Path

hub = Path(os.environ["PUB_HUB_DIR"])
slug = os.environ["PUB_PROJECT_SLUG"]
report_slug = os.environ["PUB_REPORT_SLUG"]

m = json.loads((hub / "manifest.json").read_text())
m.setdefault("projects", [])

# Filter out our project; we always replace, never merge.
m["projects"] = [p for p in m["projects"] if p.get("slug") != slug]

entry = {
    "slug": slug,
    "name": os.environ["PUB_PROJECT_NAME"],
    "description": os.environ["PUB_PROJECT_DESC"],
    "color": os.environ["PUB_PROJECT_COLOR"],
    "reports": [{
        "slug": report_slug,
        "title": os.environ["PUB_REPORT_TITLE"],
        "date": os.environ["PUB_REPORT_DATE"],
        "tags": [t for t in os.environ["PUB_REPORT_TAGS_CSV"].split(",") if t],
        "summary": os.environ["PUB_REPORT_SUMMARY"],
    }],
}
m["projects"].append(entry)

# Stable ordering by slug — keeps diffs readable across runs.
m["projects"].sort(key=lambda p: p.get("slug", ""))

out = hub / "manifest.json"
out.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
print(f"  manifest now has {len(m['projects'])} projects; this one's reports = "
      f"{[r['slug'] for r in entry['reports']]}")
PY

# -------------------------------------------------------------------- COMMIT/PUSH
cd "$HUB_DIR"
git add manifest.json "projects/$PROJECT_SLUG"

# Sanity: this commit should NOT touch anything outside our scope.
DIRTY_OUTSIDE=$(git diff --cached --name-only |
    grep -vE "^manifest.json$|^projects/$PROJECT_SLUG(/|$)" || true)
if [ -n "$DIRTY_OUTSIDE" ]; then
    echo "ERROR: staged changes outside our scope — aborting:" >&2
    echo "$DIRTY_OUTSIDE" | sed 's/^/  /' >&2
    git restore --staged .
    exit 1
fi

if git diff --cached --quiet; then
    echo "==> no changes to commit"
elif [ "$DO_PUSH" = "1" ]; then
    git -c commit.gpgsign=false commit -m "$MSG"
    echo "==> push"
    git push origin main
    echo "✓ pushed — Cloudflare deploys in 1-2 min"
else
    git -c commit.gpgsign=false commit -m "$MSG"
    echo "✓ committed locally (--no-push)"
    echo "  push with: cd $HUB_DIR && git push origin main"
fi
