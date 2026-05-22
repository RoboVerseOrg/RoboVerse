"""Upload the staged mjlab MJCF assets to the RoboVerseOrg/roboverse_data HF dataset.

Run AFTER `huggingface-cli login` (needs write access to RoboVerseOrg/roboverse_data):

    python tools/upload_mjlab_assets_to_hf.py            # dry-run: list what would upload
    python tools/upload_mjlab_assets_to_hf.py --commit   # actually upload

Uploads `roboverse_data/robots/mjlab/` -> dataset path `robots/mjlab/`, mirroring
mjlab's src/mjlab/ tree so the locator resolves files by a pure prefix swap.
"""

from __future__ import annotations

import argparse
import os

REPO_ID = "RoboVerseOrg/roboverse_data"
LOCAL_DIR = "roboverse_data/robots/mjlab"
DEST_IN_REPO = "robots/mjlab"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually upload (default: dry-run)")
    args = ap.parse_args()

    if not os.path.isdir(LOCAL_DIR):
        raise SystemExit(f"Staged dir not found: {LOCAL_DIR} (run the staging step first)")

    files = sorted(os.path.relpath(os.path.join(r, f), LOCAL_DIR) for r, _, fs in os.walk(LOCAL_DIR) for f in fs)
    total_mb = sum(os.path.getsize(os.path.join(LOCAL_DIR, f)) for f in files) / 1e6
    print(f"{len(files)} files, {total_mb:.1f} MB -> {REPO_ID}:{DEST_IN_REPO}/")

    if not args.commit:
        for f in files[:10]:
            print("  ", f)
        if len(files) > 10:
            print(f"   ... (+{len(files) - 10} more)")
        print("\nDry-run. Re-run with --commit to upload.")
        return

    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise SystemExit("Not logged in. Run `hf auth login` first.")
    api = HfApi(token=token)

    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=LOCAL_DIR,
        path_in_repo=DEST_IN_REPO,
        commit_message="Add mjlab MJCF assets (cartpole/go1/g1/yam) for RoboVerse mjlab tasks",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}/tree/main/{DEST_IN_REPO}")


if __name__ == "__main__":
    main()
