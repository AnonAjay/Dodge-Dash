import os
import shutil
from pathlib import Path
import cv2

from pose_core.blazer_indexer import BlazerIndexer
from pose_core.blazer_matcher import load_blazer_catalog, save_blazer_catalog

NEW_POSES_MAP = {
    "062f814892140b0031be5656a49e8953.jpg": {
        "id": "megumi_summon",
        "name": "Megumi Summon",
        "difficulty": "medium",
        "description": "Summon divine dogs with clasped hands and wide stance.",
    },
    "0bc4072a6bd629533a46bdb39fcc2ca8.jpg": {
        "id": "mj_smooth_criminal",
        "name": "MJ Smooth Criminal",
        "difficulty": "hard",
        "description": "Michael Jackson iconic anti-gravity lean forward.",
    },
    "195f2a67d83f17e88c6b5302261f388f.jpg": {
        "id": "dance_glide",
        "name": "Dance Glide",
        "difficulty": "medium",
        "description": "Contemporary dance balance with one leg lifted back.",
    },
    "22176749cc43d610c2e6ad3bd62a81bd.jpg": {
        "id": "disco_dancer",
        "name": "Disco Dancer",
        "difficulty": "medium",
        "description": "Retro disco stance with right arm pointing to the ceiling.",
    },
    "4b244eef8c2b01948470c0d266735579.jpg": {
        "id": "chill_stand",
        "name": "Chill Stand",
        "difficulty": "easy",
        "description": "Relaxed standing pose with both hands resting behind head.",
    },
    "7d65e2f33805f528b0651b38d9c2ba17.jpg": {
        "id": "model_pose",
        "name": "Model Pose",
        "difficulty": "medium",
        "description": "High-fashion street pose with hand on hip and lifted knee.",
    },
    "83e857f4866270a4c6bf364e62c54c41.jpg": {
        "id": "crab_squat",
        "name": "Crab Squat",
        "difficulty": "hard",
        "description": "Deep low squat with hands flexed outward in crab style.",
    },
    "8633803cc754d3e8c5a1bac52e08ca78.jpg": {
        "id": "tree_pose_fun",
        "name": "Tree Pose Fun",
        "difficulty": "medium",
        "description": "Playful tree balance with arms arched overhead.",
    },
    "8aadf81f9dfde79c7061a44b2dc92f46.jpg": {
        "id": "marionette_dance",
        "name": "Marionette Dance",
        "difficulty": "medium",
        "description": "Puppet dance stance with elbows out and knee popped up.",
    },
    "92409e067538a96327460933f38e6b6a.jpg": {
        "id": "will_smith_presenting",
        "name": "Will Smith Presenting",
        "difficulty": "medium",
        "description": "Meme presentation pose gesturing proudly to the side.",
    },
    "Pose_5.jpg": {
        "id": "rdj_stand",
        "name": "RDJ Stand",
        "difficulty": "easy",
        "description": "Sharp executive posture with hands poised low at sides.",
    },
    "ad0bc360bcab1f1246657cf77049db4e.jpg": {
        "id": "srk_arms_open",
        "name": "SRK Arms Open",
        "difficulty": "medium",
        "description": "Shah Rukh Khan iconic wide open arms romantic pose.",
    },
    "b68a043d778e7b7b6dc077cffb4d8455.jpg": {
        "id": "rockstar_disco",
        "name": "Rockstar Disco",
        "difficulty": "medium",
        "description": "Electric rockstar pose with arm raised and hip cocked.",
    },
    "ce9a9c4f5d1b6d274ed400f9aedceeea.jpg": {
        "id": "mr_bean_walk",
        "name": "Mr Bean Walk",
        "difficulty": "medium",
        "description": "Mr. Bean classic quirky walking stance with flat hand gesture.",
    },
    "download (1).png": {
        "id": "gojo_standing",
        "name": "Gojo Standing",
        "difficulty": "easy",
        "description": "Satoru Gojo confident upright stance ready for combat.",
    },
    "yoga Pose.jpg": {
        "id": "yoga_tree_pose",
        "name": "Yoga Tree Pose",
        "difficulty": "hard",
        "description": "Vrikshasana: full one-leg balance with prayer hands overhead.",
    },
}

def run_compression_and_cleanup():
    catalog_path = "data/poses_index.json"
    poses_dir = Path("data/poses")
    
    print("Loading existing catalog...")
    catalog = load_blazer_catalog(catalog_path)
    existing_by_id = {p.id: p for p in catalog}
    print(f"Catalog currently has {len(catalog)} poses.")

    indexer = BlazerIndexer()

    # Step 1: Process each new pose image
    for filename, meta in NEW_POSES_MAP.items():
        src_path = poses_dir / filename
        if not src_path.is_file():
            print(f"Warning: {filename} not found!")
            continue

        print(f"Indexing & compressing: {filename} -> {meta['id']}.jpg ...")
        tmpl = indexer.index_image_file(
            img_path=src_path,
            pose_id=meta["id"],
            pose_name=meta["name"],
            difficulty=meta["difficulty"],
            description=meta["description"],
            output_dir=poses_dir,
        )
        existing_by_id[tmpl.id] = tmpl
        print(f"  OK: {tmpl.name} (Key joints: {tmpl.key_joints})")

    indexer.close()

    # Step 2: Save updated catalog
    updated_catalog = list(existing_by_id.values())
    save_blazer_catalog(updated_catalog, catalog_path)
    print(f"\nUpdated catalog saved with {len(updated_catalog)} poses to {catalog_path}")

    # Step 3: Delete uncompressed / redundant files
    # Keep only the compressed .jpg files corresponding to catalog poses
    keep_filenames = {f"{p.id}.jpg" for p in updated_catalog}
    print(f"\nFiles to retain ({len(keep_filenames)}):")
    for k in sorted(keep_filenames):
        print(f"  - {k}")

    deleted_count = 0
    for f in poses_dir.iterdir():
        if f.is_file():
            if f.name not in keep_filenames:
                print(f"Deleting uncompressed/redundant file: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
                f.unlink()
                deleted_count += 1

    print(f"\nCleaned up {deleted_count} files.")
    
    # Step 4: Verification of remaining files
    remaining = sorted([f for f in poses_dir.iterdir() if f.is_file()])
    print(f"\nRemaining files in {poses_dir} ({len(remaining)} total):")
    total_size = 0
    for f in remaining:
        sz = f.stat().st_size
        total_size += sz
        print(f"  {f.name:25s} | {sz / 1024:5.1f} KB")
    print(f"Total folder size: {total_size / 1024:.1f} KB ({total_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    run_compression_and_cleanup()
