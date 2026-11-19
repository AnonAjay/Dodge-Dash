"""CLI Tool for Pose Blazer Catalog Indexing.

Usage:
  # Initialize or reset the 6 starter poses:
  python tools/index_poses.py --init-defaults

  # Index a single new pose photo:
  python tools/index_poses.py --add path/to/my_pose.jpg --id hero_pose --name "Hero Pose" --difficulty hard

  # Scan a folder of new images and add any unindexed poses:
  python tools/index_poses.py --scan data/raw_poses/

  # List all currently indexed poses and their key joints:
  python tools/index_poses.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Insert site-packages if available
VENV_PKGS = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_PKGS.is_dir() and str(VENV_PKGS) not in sys.path:
    sys.path.insert(0, str(VENV_PKGS))

from pose_core.blazer_indexer import BlazerIndexer, generate_starter_catalog
from pose_core.blazer_matcher import (
    load_blazer_catalog,
    save_blazer_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose Blazer Image Ingestion & Indexer")
    parser.add_argument("--init-defaults", action="store_true", help="Generate 6 starter poses and default catalog")
    parser.add_argument("--list", action="store_true", help="List all currently indexed poses in catalog")
    parser.add_argument("--add", type=str, default="", help="Path to image file to index")
    parser.add_argument("--id", type=str, default="", help="Custom pose ID (defaults to filename)")
    parser.add_argument("--name", type=str, default="", help="Display name of the pose")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium", help="Pose difficulty")
    parser.add_argument("--description", type=str, default="", help="Description or instructions for player")
    parser.add_argument("--scan", type=str, default="", help="Directory of images to batch index")
    parser.add_argument("--catalog", type=str, default="data/poses_index.json", help="Path to catalog JSON")
    parser.add_argument("--poses-dir", type=str, default="data/poses", help="Directory where game cards are saved")
    args = parser.parse_args()

    if args.init_defaults:
        print("Generating starter poses catalog and cards...")
        templates = generate_starter_catalog(catalog_path=args.catalog, poses_dir=args.poses_dir)
        print(f"Successfully generated {len(templates)} starter poses in '{args.catalog}' and cards in '{args.poses_dir}/'")
        return

    if args.list:
        catalog = load_blazer_catalog(args.catalog)
        if not catalog:
            print(f"Catalog at '{args.catalog}' is empty or does not exist.")
            print("Run 'python tools/index_poses.py --init-defaults' to seed default poses.")
            return

        print(f"\nPose Blazer Catalog ({len(catalog)} poses):")
        print("-" * 75)
        print(f"{'ID':<16} | {'NAME':<18} | {'DIFF':<6} | {'KEY JOINTS'}")
        print("-" * 75)
        for p in catalog:
            key_j = ", ".join(p.key_joints) if p.key_joints else "all"
            print(f"{p.id:<16} | {p.name:<18} | {p.difficulty.upper():<6} | {key_j}")
        print("-" * 75)
        return

    indexer = BlazerIndexer()

    if args.add:
        img_p = Path(args.add)
        print(f"Indexing single image: {img_p} ...")
        tmpl = indexer.index_image_file(
            img_path=img_p,
            pose_id=args.id if args.id else None,
            pose_name=args.name if args.name else None,
            difficulty=args.difficulty,
            description=args.description,
            output_dir=args.poses_dir,
        )

        catalog = load_blazer_catalog(args.catalog)
        # Replace if existing or append
        existing_idx = next((i for i, p in enumerate(catalog) if p.id == tmpl.id), -1)
        if existing_idx >= 0:
            catalog[existing_idx] = tmpl
            print(f"Updated existing pose '{tmpl.id}' in catalog.")
        else:
            catalog.append(tmpl)
            print(f"Appended new pose '{tmpl.id}' to catalog.")

        save_blazer_catalog(catalog, args.catalog)
        print(f"Saved catalog to '{args.catalog}'. Key joints: {tmpl.key_joints}")
        indexer.close()
        return

    if args.scan:
        scan_dir = Path(args.scan)
        if not scan_dir.is_dir():
            print(f"Scan directory not found: {scan_dir}")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        img_files = [f for f in scan_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"Scanning '{scan_dir}': found {len(img_files)} candidate images.")

        catalog = load_blazer_catalog(args.catalog)
        indexed_ids = {p.id for p in catalog}

        added_count = 0
        for f in img_files:
            pid = f.stem.lower().replace(" ", "_").replace("-", "_")
            if pid in indexed_ids:
                print(f"Skipping already indexed pose: {pid}")
                continue

            try:
                print(f"Indexing new pose from {f.name} ...")
                tmpl = indexer.index_image_file(
                    img_path=f,
                    pose_id=pid,
                    output_dir=args.poses_dir,
                )
                catalog.append(tmpl)
                indexed_ids.add(pid)
                added_count += 1
            except Exception as e:
                print(f"Error indexing {f.name}: {e}")

        save_blazer_catalog(catalog, args.catalog)
        print(f"Scan complete. Added {added_count} new poses. Total poses in catalog: {len(catalog)}")
        indexer.close()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
