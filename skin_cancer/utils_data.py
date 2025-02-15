from pathlib import Path
import shutil
import random
import pandas as pd

def split_dataset(source_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2, image_ext: str = "*.jpg"):
    """
    Splits a skin cancer dataset into training, validation, and calibration sets using GroundTruth CSV.

    Parameters:
        source_dir (str): Path to the dataset containing 'images' folder and 'GroundTruth.csv'.
        train_ratio (float): Percentage of images used for training (default: 0.7).
        val_ratio (float): Percentage of images used for validation (default: 0.2).
        image_ext (str): Image file extension to search for (default: "*.jpg").
    """

    source_ground_truth = Path(source_dir) / "GroundTruth.csv"
    source_path = Path(source_dir) / "images"
    train_path = Path(source_dir) / "train"
    val_path = Path(source_dir) / "val"
    calib_path = Path(source_dir) / "calibration"

    # Ensure split ratios sum to 1
    calib_ratio = 1.0 - (train_ratio + val_ratio)
    assert 0 < calib_ratio < 1, "Invalid split ratios! Ensure train + val + calib = 1."

    # Load ground truth labels
    df = pd.read_csv(source_ground_truth)

    # Convert one-hot encoded labels to category names
    label_columns = df.columns[1:]  # Ignore first column ('image')
    df["label"] = df[label_columns].idxmax(axis=1)  # Get the class name

    # Create directories
    for category in label_columns:
        (train_path / category).mkdir(parents=True, exist_ok=True)
        (val_path / category).mkdir(parents=True, exist_ok=True)
        (calib_path / category).mkdir(parents=True, exist_ok=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split dataset
    train_split = int(len(df) * train_ratio)
    val_split = train_split + int(len(df) * val_ratio)

    train_df = df.iloc[:train_split]
    val_df = df.iloc[train_split:val_split]
    calib_df = df.iloc[val_split:]

    # Function to copy files
    def copy_files(subset_df, target_dir):
        for _, row in subset_df.iterrows():
            image_file = source_path / f"{row['image']}.jpg"
            if image_file.exists():
                shutil.copy(image_file, target_dir / row["label"] / f"{row['image']}.jpg")

    # Copy images to respective folders
    copy_files(train_df, train_path)
    copy_files(val_df, val_path)
    copy_files(calib_df, calib_path)

    # Print summary
    print(f"Dataset successfully split into training, validation, and calibration sets.")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Calibration: {len(calib_df)}")
