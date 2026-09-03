#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-"$repo_root/public"}"
# MetaSim lives in this repository (packages/metasim); METASIM_DIR overrides for out-of-tree builds
# and METASIM_REPO/METASIM_REF are kept for building an older, pre-monorepo MetaSim.
metasim_repo="${METASIM_REPO:-}"
metasim_ref="${METASIM_REF:-main}"
metasim_dir="${METASIM_DIR:-$repo_root/packages/metasim}"
cname="${ROBOVERSE_WIKI_CNAME:-roboverse.wiki}"
python_bin="${PYTHON:-python}"

if [[ -z "$output_dir" || "$output_dir" == "/" ]]; then
    echo "Refusing to clean unsafe output directory: ${output_dir:-<empty>}" >&2
    exit 2
fi

tmp_dir=""
cleanup() {
    if [[ -n "$tmp_dir" && -d "$tmp_dir" ]]; then
        rm -rf "$tmp_dir"
    fi
}
trap cleanup EXIT

if [[ -n "$metasim_repo" ]]; then
    tmp_dir="$(mktemp -d)"
    metasim_dir="$tmp_dir/MetaSim"
    git clone --depth 1 --branch "$metasim_ref" "$metasim_repo" "$metasim_dir"
fi

if [[ ! -d "$metasim_dir/docs/source" ]]; then
    echo "MetaSim docs source not found at $metasim_dir/docs/source" >&2
    exit 3
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"

# RoboVerse docs are the homepage — built directly at the site root so
# visiting roboverse.wiki/ lands on the Sphinx RoboVerse intro, matching
# the pre-#764 deployment layout. MetaSim docs live under /metasim/.
"$python_bin" -m sphinx -b html "$repo_root/docs/source" "$output_dir"
"$python_bin" -m sphinx -b html "$metasim_dir/docs/source" "$output_dir/metasim"

if [[ -f "$metasim_dir/docs/source/images/tea.jpg" ]]; then
    mkdir -p "$output_dir/metasim/_images"
    cp "$metasim_dir/docs/source/images/tea.jpg" "$output_dir/metasim/_images/tea.jpg"
fi

printf "%s\n" "$cname" > "$output_dir/CNAME"
touch "$output_dir/.nojekyll"
