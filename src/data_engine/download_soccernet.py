"""
SoccerNet Data Downloader

Downloads the SoccerNet Tracking dataset for training the YOLO model.

Usage:
    python download_soccernet.py --output data/soccernet

Requirements:
    pip install SoccerNet --upgrade
"""

import os
import argparse
from pathlib import Path


def download_soccernet_tracking(
    output_dir: str = "data/soccernet",
    splits: list = None,
    dataset_version: str = "tracking"  # "tracking" or "tracking-2023"
) -> None:
    """
    Download SoccerNet Tracking dataset.
    
    Args:
        output_dir: Directory to save downloaded data
        splits: List of splits to download ["train", "test", "challenge"]
        dataset_version: Which tracking dataset ("tracking" or "tracking-2023")
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("SoccerNet not installed. Installing...")
        os.system("pip install SoccerNet --upgrade")
        from SoccerNet.Downloader import SoccerNetDownloader
    
    if splits is None:
        splits = ["train", "test", "challenge"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading SoccerNet {dataset_version} to: {output_path.absolute()}")
    print(f"Splits: {splits}")
    print("-" * 50)
    
    # Initialize downloader
    downloader = SoccerNetDownloader(LocalDirectory=str(output_path))
    
    # Download tracking data
    print(f"\nDownloading {dataset_version} data and labels...")
    downloader.downloadDataTask(task=dataset_version, split=splits)
    
    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE!")
    print("=" * 50)
    print(f"\nData saved to: {output_path.absolute()}")
    print("\nNext steps:")
    print("  1. Run the parser to convert to YOLO format:")
    print("     python src/data_engine/soccernet_parser.py -i data/soccernet -o data/processed")
    print("  2. Train YOLO model:")
    print("     yolo detect train model=yolo11n.pt data=data/processed/dataset.yaml epochs=20")


def list_available_data(output_dir: str) -> None:
    """
    List what data has been downloaded.
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Directory does not exist: {output_path}")
        return
    
    print(f"\nContents of {output_path}:")
    print("-" * 50)
    
    for item in output_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(output_path)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SoccerNet Tracking dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/soccernet",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--splits", "-s",
        nargs="+",
        default=["train", "test", "challenge"],
        help="Splits to download (default: train test challenge)"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        choices=["tracking", "tracking-2023"],
        default="tracking",
        help="Dataset version (default: tracking)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List downloaded data instead of downloading"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_data(args.output)
    else:
        download_soccernet_tracking(
            output_dir=args.output,
            splits=args.splits,
            dataset_version=args.version
        )
