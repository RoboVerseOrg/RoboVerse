#!/usr/bin/env bash
# publish_reports.sh — RoboVerse → hub publish wrapper
#
# 这个脚本把所有 RoboVerse-side report 一次性 (a) 重新生成、(b) 注册到 hub
# (add_report.sh, idempotent)、(c) sync + git push 到 Cloudflare Pages。
#
# Hub 约定 (权威说明在 /home/ghr/projects/reports/README.md):
#   - project-slug: roboverse  (固定, 别改, URL 哈希用)
#   - report dir 标准位置: /home/ghr/projects/RoboVerse/reports/<sub-slug>/
#   - 子 report 当前 7 个:
#       task_audit_2026_05_13     — 维护 / task audit baseline
#       branch_audit_2026_05_16   — 维护 / branch + backcompat audit
#       maniskill_sapien_repro_2026_05_16  — 集成 / maniskill recon (前置)
#       maniskill_integration     — 集成 / ManiSkill (中文)
#       mjlab_integration         — 集成 / mjlab + G1 RL (中文, 多页)
#       robotwin_integration      — 集成 / RoboTwin (中文)
#       roboverse_overview        — ⭐ 总览 (统一索引 6 子报告)
#
# 用法:
#   bash tools/publish_reports.sh                       # 全部重生成 + sync + push
#   bash tools/publish_reports.sh --no-push             # 本地预览, 不 push
#   bash tools/publish_reports.sh --no-build            # 跳过 _build*.py 重生成
#   bash tools/publish_reports.sh --only mjlab_integration  # 只刷一个 sub-report
#   bash tools/publish_reports.sh -m "comment"          # 自定义 commit msg
#
# 跑完做了什么:
#   1. 重跑各 sub-report 的 _build*.py (mjlab_integration / roboverse_overview)
#   2. 对每个子 report 调 publish.sh <slug> <dir> "<title>" 重新 register manifest
#      (idempotent: 已存在的条目会被 update; metadata 不变)
#   3. publish.sh 内部 sync_to_github.py 扫所有 HTML 引用资源, 大文件走 LFS,
#      最后 git push (除非 --no-push)
#
# 关于 outputs symlink:
#   本项目的所有 report 都用相对路径引用 ./runs/<task>/*.mp4 (在 report dir 内),
#   不跨目录, 所以 *不需要* 额外建 outputs symlink。
#   如果将来引入 ../outputs 跨目录引用, 需要补:
#     ln -sfn /home/ghr/projects/RoboVerse/outputs \
#             /home/ghr/projects/reports/projects/roboverse/outputs

set -euo pipefail

PROJECT_SLUG="roboverse"
REPORTS_ROOT="/home/ghr/projects/RoboVerse/reports"
HUB="/home/ghr/projects/reports"
PUBLISH="$HUB/publish.sh"

if [ ! -x "$PUBLISH" ]; then
    echo "ERROR: hub publish script not found at $PUBLISH" >&2
    exit 1
fi

DO_BUILD=1
DO_PUSH=1
ONLY=""
MSG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) DO_BUILD=0; shift;;
        --no-push) DO_PUSH=0; shift;;
        --only) ONLY="$2"; shift 2;;
        --only=*) ONLY="${1#*=}"; shift;;
        -m|--message) MSG="$2"; shift 2;;
        -m=*|--message=*) MSG="${1#*=}"; shift;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# Sub-report registry: each row is "slug | title | tags | summary".
# Keep tags/summary current with what's actually in the report; this is the
# source of truth for the hub manifest entries owned by `roboverse`.
SUBREPORTS=(
    "roboverse_overview|⭐ RoboVerse 总览 (6 子报告统一索引, 中文多页)|总览,维护,集成,中文,多页|14 页 vlm_robot 风格汇总: 维护 / 新仿真器集成 / 横向主题 / 时间线。突出 G1 真 walking + 10+ MetaSim 修复 + 跨报告痛点汇总。"
    "mjlab_integration|mjlab → MetaSim/MuJoCo (含 G1 RL demo, 多页中文)|mjlab,mujoco,g1,rl,humanoid,中文,多页|14 页报告。G1 walking 本地训 + 12 任务物理对齐 ≤1e-9 + 7 个 MetaSim 修复 (5 分支, 11 回归测试)。"
    "maniskill_integration|ManiSkill → MetaSim/Sapien 集成 (中文报告)|maniskill,sapien,parity,中文|4 桌面任务 harness + maniskill.pick_cube_v1 几何 port (3/10 项) + 10 项 spec 缺口分析。"
    "robotwin_integration|RoboTwin → MetaSim/Sapien 集成 (中文报告)|robotwin,sapien,中文|ALOHA-AgileX 38 DoF 加载 + 渲染 + 50 任务清单 + 3 集成路线方案。痛点: sapien 3.0.0b1 / mplib 0.2.1 / curobo 依赖冲突。"
)

for row in "${SUBREPORTS[@]}"; do
    slug="${row%%|*}"; rest="${row#*|}"
    title="${rest%%|*}"; rest="${rest#*|}"
    tags="${rest%%|*}"
    summary="${rest#*|}"

    if [ -n "$ONLY" ] && [ "$ONLY" != "$slug" ]; then
        continue
    fi

    report_dir="$REPORTS_ROOT/$slug"
    if [ ! -d "$report_dir" ]; then
        echo "WARN: skipping $slug — report dir missing: $report_dir" >&2
        continue
    fi

    # Re-run any HTML generator script in the report dir (idempotent).
    if [ "$DO_BUILD" = "1" ]; then
        for builder in "$report_dir"/_build*.py; do
            [ -f "$builder" ] || continue
            echo "==> [$slug] rebuild via $(basename "$builder")"
            (cd "$report_dir" && python3 "$(basename "$builder")")
        done
    fi

    # Register / refresh hub entry (add_report.sh is idempotent).
    echo "==> [$slug] register in hub manifest"
    "$HUB/add_report.sh" "$PROJECT_SLUG" "$report_dir" "$title" \
        --tags="$tags" --summary="$summary" \
        | tail -n 5
done

# Final sync + push (publish.sh handles the LFS + git mechanics).
SYNC_ARGS=()
if [ "$DO_PUSH" = "0" ]; then SYNC_ARGS+=("--no-push"); fi
if [ -n "$MSG" ]; then SYNC_ARGS+=("-m" "$MSG"); fi
[ ${#SYNC_ARGS[@]} -eq 0 ] && SYNC_ARGS=("-m" "roboverse: $(date +%Y-%m-%d_%H:%M) refresh")

echo ""
echo "==> sync hub + deploy"
"$PUBLISH" "${SYNC_ARGS[@]}"
