"""Automated 1-Command Pose Catalog Synchronization.

Usage:
    python -m tools.sync_poses

Drop any new pose images (.jpg, .jpeg, .png, .webp) into data/poses/,
run this script, and they are automatically:
  1. Detected & scanned for human pose landmarks using MediaPipe.
  2. Angles extracted for all 6 core limbs (elbows, shoulders, hips, knees).
  3. Key distinguishing joints auto-detected (anti-cheat gate).
  4. Formatted and added to data/poses_index.json.
  5. Instantly playable in Station B (Pose Blazer).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_core.blazer_indexer import BlazerIndexer
from pose_core.blazer_matcher import load_blazer_catalog, save_blazer_catalog


def sync_poses(
    poses_dir: str | Path = "data/poses",
    catalog_path: str | Path = "data/poses_index.json",
    force_all: bool = False,
) -> int:
    """Scan poses_dir for images and sync with catalog."""
    poses_path = Path(poses_dir)
    cat_path = Path(catalog_path)

    if not poses_path.is_dir():
        print(f"[!] Directory not found: {poses_path}")
        return 0

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    candidate_files = sorted([
        f for f in poses_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

    print(f"[*] Found {len(candidate_files)} image files in {poses_path}/")
    catalog = load_blazer_catalog(cat_path) if cat_path.is_file() else []
    indexed_map = {p.id: p for p in catalog}

    new_files = []
    for f in candidate_files:
        pid = f.stem.lower().replace(" ", "_").replace("-", "_")
        if force_all or pid not in indexed_map:
            new_files.append((f, pid))

    if not new_files:
        print(f"[✓] All {len(candidate_files)} images are already synced in {cat_path} ({len(catalog)} active poses).")
        return 0

    print(f"[*] Processing {len(new_files)} new or updated pose images...")
    indexer = BlazerIndexer()
    added_count = 0
    updated_catalog = list(catalog)

    for f, pid in new_files:
        pname = f.stem.replace("_", " ").replace("-", " ").title()
        print(f"  -> Indexing '{pname}' ({f.name})...", end="", flush=True)
        try:
            template = indexer.index_image_file(
                img_path=f,
                pose_id=pid,
                pose_name=pname,
                output_dir=poses_path,
            )
            # Replace existing or append
            idx = next((i for i, p in enumerate(updated_catalog) if p.id == pid), -1)
            if idx >= 0:
                updated_catalog[idx] = template
            else:
                updated_catalog.append(template)
            added_count += 1
            print(f" [OK] (Key joints: {', '.join(template.key_joints) or 'none'})")
        except Exception as e:
            print(f" [FAILED] {e}")

    indexer.close()

    if added_count > 0:
        save_blazer_catalog(updated_catalog, cat_path)
        print(f"\n[✓] Successfully synced {added_count} poses! Total in catalog: {len(updated_catalog)}")
    else:
        print("\n[!] No new poses could be indexed.")

    return added_count


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Auto-sync data/poses/ with Pose Blazer catalog")
    parser.add_argument("--dir", default="data/poses", help="Path to pose images directory")
    parser.add_argument("--catalog", default="data/poses_index.json", help="Path to catalog JSON")
    parser.add_argument("--force", action="store_true", help="Force re-indexing of all images")
    args = parser.parse_args()

    sync_poses(poses_dir=args.dir, catalog_path=args.catalog, force_all=args.force)


if __name__ == "__main__":
    main()
