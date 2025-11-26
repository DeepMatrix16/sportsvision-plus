"""
SoccerNet Tracking Data Parser

Converts SoccerNet "Tracking" data (MOT format) into YOLO format for training.

SoccerNet Tracking uses MOT (Multiple Object Tracking) format:
- gt/gt.txt: frame_id, track_id, x, y, w, h, conf, class_id, visibility
- img1/: Frame images (000001.jpg, etc.)
- seqinfo.ini: Sequence metadata
- gameinfo.ini: Track ID to class name mapping

YOLO Class IDs:
    0: Ball
    1: Player
    2: Referee  
    3: Goalkeeper

YOLO Format: [class_id, x_center/img_w, y_center/img_h, w/img_w, h/img_h]
"""

import os
import re
import shutil
import configparser
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


# Default image dimensions
DEFAULT_IMG_WIDTH = 1920
DEFAULT_IMG_HEIGHT = 1080


def parse_class_from_tracklet_info(tracklet_info: str) -> Optional[int]:
    """
    Parse class ID from SoccerNet gameinfo.ini tracklet description.
    
    Examples:
        "player team left;10" -> 1 (Player)
        "ball;1" -> 0 (Ball)
        "referee;main" -> 2 (Referee)
        "goalkeeper team right;X" -> 3 (Goalkeeper)
        "goalkeepers team left;y" -> 3 (Goalkeeper)
    
    Args:
        tracklet_info: Tracklet description string from gameinfo.ini
        
    Returns:
        YOLO class ID (0=Ball, 1=Player, 2=Referee, 3=Goalkeeper) or None
    """
    info_lower = tracklet_info.lower().strip()
    
    # Ball
    if info_lower.startswith("ball"):
        return 0
    
    # Goalkeeper (check before player since "goalkeeper" contains no "player" substring issues)
    if "goalkeeper" in info_lower or "goalie" in info_lower:
        return 3
    
    # Player
    if "player" in info_lower:
        return 1
    
    # Referee
    if "referee" in info_lower or "ref" in info_lower:
        return 2
    
    return None


def parse_gameinfo_ini(ini_path: Path) -> Dict[int, int]:
    """
    Parse gameinfo.ini to get track_id -> class_id mapping.
    
    Args:
        ini_path: Path to gameinfo.ini file
        
    Returns:
        Dictionary mapping track_id to YOLO class_id
    """
    track_to_class = {}
    
    if not ini_path.exists():
        return track_to_class
    
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    
    if 'Sequence' not in config:
        return track_to_class
    
    # Parse trackletID_X entries
    for key, value in config['Sequence'].items():
        if key.startswith('trackletid_'):
            # Extract track ID number from key (e.g., "trackletid_1" -> 1)
            try:
                track_id = int(key.replace('trackletid_', ''))
            except ValueError:
                continue
            
            # Parse class from value
            class_id = parse_class_from_tracklet_info(value)
            if class_id is not None:
                track_to_class[track_id] = class_id
    
    return track_to_class


def parse_seqinfo_ini(ini_path: Path) -> Tuple[int, int, int]:
    """
    Parse seqinfo.ini to get image dimensions and sequence length.
    
    Args:
        ini_path: Path to seqinfo.ini file
        
    Returns:
        Tuple of (width, height, sequence_length)
    """
    width = DEFAULT_IMG_WIDTH
    height = DEFAULT_IMG_HEIGHT
    seq_length = 0
    
    if not ini_path.exists():
        return width, height, seq_length
    
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    
    if 'Sequence' in config:
        width = int(config['Sequence'].get('imwidth', DEFAULT_IMG_WIDTH))
        height = int(config['Sequence'].get('imheight', DEFAULT_IMG_HEIGHT))
        seq_length = int(config['Sequence'].get('seqlength', 0))
    
    return width, height, seq_length


def parse_gt_txt(gt_path: Path, track_to_class: Dict[int, int]) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Parse gt.txt file in MOT format.
    
    MOT format: frame_id, track_id, x, y, w, h, conf, class, visibility
    
    Args:
        gt_path: Path to gt.txt file
        track_to_class: Mapping from track_id to class_id
        
    Returns:
        Dictionary mapping frame_id to list of (class_id, x, y, w, h) tuples
    """
    frame_annotations = defaultdict(list)
    
    if not gt_path.exists():
        return frame_annotations
    
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except (ValueError, IndexError):
                continue
            
            # Get class from track_to_class mapping
            class_id = track_to_class.get(track_id)
            if class_id is None:
                # If no mapping, try to infer from position in file or skip
                continue
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            frame_annotations[frame_id].append((class_id, x, y, w, h))
    
    return frame_annotations


def convert_bbox_to_yolo(
    x: float, y: float, w: float, h: float,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x, y, w, h] (top-left corner) to YOLO format.
    
    YOLO format: [x_center/img_w, y_center/img_h, w/img_w, h/img_h]
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    return x_center, y_center, w_norm, h_norm


def process_sequence(
    seq_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    sample_rate: int = 1
) -> Dict[str, int]:
    """
    Process a single SoccerNet sequence folder.
    
    Args:
        seq_dir: Path to sequence directory (e.g., SNMOT-060)
        output_images_dir: Output directory for images
        output_labels_dir: Output directory for labels
        sample_rate: Process every Nth frame
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "frames": 0,
        "annotations": 0,
        "balls": 0,
        "players": 0,
        "referees": 0,
        "goalkeepers": 0
    }
    
    seq_name = seq_dir.name
    
    # Parse configuration files
    gameinfo_path = seq_dir / "gameinfo.ini"
    seqinfo_path = seq_dir / "seqinfo.ini"
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    
    # Get track to class mapping
    track_to_class = parse_gameinfo_ini(gameinfo_path)
    if not track_to_class:
        print(f"  Warning: No track mapping found in {seq_name}")
        return stats
    
    # Get image dimensions
    img_width, img_height, seq_length = parse_seqinfo_ini(seqinfo_path)
    
    # Parse ground truth
    frame_annotations = parse_gt_txt(gt_path, track_to_class)
    if not frame_annotations:
        print(f"  Warning: No annotations found in {seq_name}")
        return stats
    
    # Process each frame
    for frame_id, annotations in frame_annotations.items():
        # Apply sample rate
        if (frame_id - 1) % sample_rate != 0:
            continue
        
        # Source image path
        img_filename = f"{frame_id:06d}.jpg"
        src_img_path = img_dir / img_filename
        
        if not src_img_path.exists():
            continue
        
        # Output paths
        output_filename = f"{seq_name}_{frame_id:06d}"
        dst_img_path = output_images_dir / f"{output_filename}.jpg"
        dst_label_path = output_labels_dir / f"{output_filename}.txt"
        
        # Copy image
        shutil.copy2(src_img_path, dst_img_path)
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for class_id, x, y, w, h in annotations:
            x_c, y_c, w_n, h_n = convert_bbox_to_yolo(x, y, w, h, img_width, img_height)
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            
            # Update stats
            if class_id == 0:
                stats["balls"] += 1
            elif class_id == 1:
                stats["players"] += 1
            elif class_id == 2:
                stats["referees"] += 1
            elif class_id == 3:
                stats["goalkeepers"] += 1
            
            stats["annotations"] += 1
        
        # Write label file
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        stats["frames"] += 1
    
    return stats


def extract_zip_files(tracking_dir: Path) -> None:
    """
    Extract any zip files in the tracking directory.
    """
    zip_files = list(tracking_dir.glob("*.zip"))
    
    for zip_file in zip_files:
        extract_dir = tracking_dir / zip_file.stem
        if not extract_dir.exists():
            print(f"Extracting {zip_file.name}...")
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  Extracted to {extract_dir}")


def parse_soccernet_tracking(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 1,
    extract_zips: bool = True
) -> Dict[str, int]:
    """
    Main function to parse SoccerNet Tracking dataset into YOLO format.
    
    Args:
        input_dir: Path to SoccerNet tracking data (e.g., "data/soccernet")
        output_dir: Path to output directory (e.g., "data/processed")
        sample_rate: Process every Nth frame (default 1 = all frames)
        extract_zips: Whether to extract zip files automatically
        
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_frames": 0,
        "total_annotations": 0,
        "balls": 0,
        "players": 0,
        "referees": 0,
        "goalkeepers": 0,
        "sequences_processed": 0
    }
    
    # Find tracking directory
    tracking_dir = input_path / "tracking"
    if not tracking_dir.exists():
        tracking_dir = input_path  # Maybe data is directly in input_dir
    
    # Extract zip files if needed
    if extract_zips:
        extract_zip_files(tracking_dir)
    
    # Find all sequence directories (SNMOT-XXX folders)
    sequence_dirs = []
    
    # Look for SNMOT-* directories recursively
    for snmot_dir in tracking_dir.rglob("SNMOT-*"):
        if snmot_dir.is_dir() and (snmot_dir / "gt").exists():
            sequence_dirs.append(snmot_dir)
    
    if not sequence_dirs:
        print(f"No SoccerNet sequences found in {input_dir}")
        print("Expected structure: tracking/[train|test]/SNMOT-XXX/")
        print("\nSearching for any gt.txt files...")
        
        gt_files = list(input_path.rglob("gt.txt"))
        if gt_files:
            print(f"Found {len(gt_files)} gt.txt files:")
            for gf in gt_files[:5]:
                print(f"  {gf}")
        return stats
    
    print(f"Found {len(sequence_dirs)} sequences to process")
    print(f"Sample rate: every {sample_rate} frame(s)")
    print("-" * 50)
    
    # Process each sequence
    for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
        seq_stats = process_sequence(
            seq_dir=seq_dir,
            output_images_dir=images_dir,
            output_labels_dir=labels_dir,
            sample_rate=sample_rate
        )
        
        stats["total_frames"] += seq_stats["frames"]
        stats["total_annotations"] += seq_stats["annotations"]
        stats["balls"] += seq_stats["balls"]
        stats["players"] += seq_stats["players"]
        stats["referees"] += seq_stats["referees"]
        stats["goalkeepers"] += seq_stats["goalkeepers"]
        stats["sequences_processed"] += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("PARSING COMPLETE")
    print("=" * 50)
    print(f"Sequences processed: {stats['sequences_processed']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"  - Balls: {stats['balls']}")
    print(f"  - Players: {stats['players']}")
    print(f"  - Referees: {stats['referees']}")
    print(f"  - Goalkeepers: {stats['goalkeepers']}")
    print(f"\nOutput saved to: {output_path}")
    
    return stats


def create_yolo_dataset_yaml(
    output_dir: str,
    yaml_path: str = "dataset.yaml"
) -> str:
    """
    Create YOLO dataset configuration YAML file.
    """
    output_path = Path(output_dir)
    
    # Check if train/val split exists
    train_dir = output_path / "train" / "images"
    val_dir = output_path / "val" / "images"
    
    if train_dir.exists() and val_dir.exists():
        yaml_content = f"""# SportsVision+ Dataset Configuration
# Auto-generated by soccernet_parser.py

path: {output_path.absolute()}
train: train/images
val: val/images

# Classes
names:
  0: ball
  1: player
  2: referee
  3: goalkeeper

# Number of classes
nc: 4
"""
    else:
        yaml_content = f"""# SportsVision+ Dataset Configuration
# Auto-generated by soccernet_parser.py

path: {output_path.absolute()}
train: images
val: images

# Classes
names:
  0: ball
  1: player
  2: referee
  3: goalkeeper

# Number of classes
nc: 4
"""
    
    yaml_file = output_path / yaml_path
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created: {yaml_file}")
    return str(yaml_file)


def split_dataset(
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> None:
    """
    Split dataset into train and validation sets.
    """
    import random
    random.seed(seed)
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    # Create train/val directories
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(image_files)
    
    if not image_files:
        print("No images found to split!")
        return
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"\nSplitting dataset...")
    print(f"  Total images: {len(image_files)}")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    
    # Copy files
    for img_file in tqdm(train_files, desc="Creating train set"):
        label_file = labels_dir / f"{img_file.stem}.txt"
        shutil.copy2(img_file, train_images / img_file.name)
        if label_file.exists():
            shutil.copy2(label_file, train_labels / label_file.name)
    
    for img_file in tqdm(val_files, desc="Creating val set"):
        label_file = labels_dir / f"{img_file.stem}.txt"
        shutil.copy2(img_file, val_images / img_file.name)
        if label_file.exists():
            shutil.copy2(label_file, val_labels / label_file.name)
    
    print(f"\nDataset split complete!")


# ============================================
# CLI Entry Point
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet Tracking data to YOLO format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/soccernet",
        help="Input directory containing SoccerNet tracking data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed",
        help="Output directory for YOLO format data"
    )
    parser.add_argument(
        "--sample-rate", "-s",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split dataset into train/val after parsing"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip automatic zip extraction"
    )
    
    args = parser.parse_args()
    
    # Run parser
    stats = parse_soccernet_tracking(
        input_dir=args.input,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        extract_zips=not args.no_extract
    )
    
    # Create YAML config
    if stats["total_frames"] > 0:
        # Split if requested
        if args.split:
            split_dataset(args.output, train_ratio=args.train_ratio)
        
        create_yolo_dataset_yaml(args.output)
    else:
        print("\nNo data was processed. Please check your input directory structure.")
