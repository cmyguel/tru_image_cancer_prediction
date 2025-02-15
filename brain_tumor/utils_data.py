from pathlib import Path
import shutil
import random

def split_dataset(source_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2, image_ext: str = "*.jpg"):
    """
    Splits a dataset into training, validation, and calibration sets.

    Parameters:
        source_dir (str): Path to the original dataset containing 'original' subdirectory.
        train_ratio (float): Percentage of images used for training (default: 0.7).
        val_ratio (float): Percentage of images used for validation (default: 0.2).
        image_ext (str): Image file extension to search for (default: "*.jpg").
    """
    source_path = Path(source_dir) / "original"
    train_path = Path(source_dir) / "train"
    val_path = Path(source_dir) / "val"
    calib_path = Path(source_dir) / "calibration"  # Calibration set

    categories = [category.name for category in source_path.iterdir() if category.is_dir()]

    # Ensure split ratios sum to 1
    calib_ratio = 1.0 - (train_ratio + val_ratio)
    assert 0 < calib_ratio < 1, "Invalid split ratios! Ensure train + val < 1."

    # Create train, val, and calibration directories
    for category in categories:
        (train_path / category).mkdir(parents=True, exist_ok=True)
        (val_path / category).mkdir(parents=True, exist_ok=True)
        (calib_path / category).mkdir(parents=True, exist_ok=True)

    # Process each category
    for category in categories:
        image_files = list((source_path / category).glob(image_ext))
        random.shuffle(image_files)

        train_split = int(len(image_files) * train_ratio)
        val_split = train_split + int(len(image_files) * val_ratio)

        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        calib_files = image_files[val_split:]  # Remaining for calibration

        # Copy images to respective folders
        for file in train_files:
            shutil.copy(file, train_path / category / file.name)
        for file in val_files:
            shutil.copy(file, val_path / category / file.name)
        for file in calib_files:
            shutil.copy(file, calib_path / category / file.name)

        print(f"Category '{category}': {len(train_files)} train, {len(val_files)} val, {len(calib_files)} calibration")

    print("Dataset successfully split into training, validation, and calibration sets.")