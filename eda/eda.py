import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import cv2
from PIL import Image
import re
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

def load_data():
    """Load training data and perform initial exploration"""
    df = pd.read_csv(TRAIN_CSV)
    print(f"Training data shape: {df.shape}")

    # Display first few rows of the dataset
    print("\nFirst few rows of the training data:")
    display(df.head())

    # Check for missing values
    print("\nMissing values in the dataset:")
    display(df.isnull().sum())

    return df

def explore_dataset_structure(df):
    """Explore the structure of the dataset"""
    print("Dataset columns:", df.columns.tolist())

    # Extract case, day, and slice from the id column
    # Format: case{case}_day{day}_slice_{slice}
    if 'id' in df.columns:
        df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
        df['case'] = df['case'].astype(int)
        df['day'] = df['day'].astype(int)
        df['slice'] = df['slice'].astype(int)

    # Extract unique cases, days, and slices
    cases = df['case'].unique()
    days = df['day'].unique()

    print(f"\nNumber of unique cases: {len(cases)}")
    print(f"Number of unique days: {len(days)}")

    # Create a case-day count
    case_day_df = df[['case', 'day']].drop_duplicates()
    case_counts = case_day_df['case'].value_counts()

    print("\nDistribution of days per case:")
    display(case_counts.describe())

    # Plot distribution of days per case
    plt.figure(figsize=(12, 6))
    sns.histplot(case_counts, kde=True)
    plt.title('Distribution of Days per Case')
    plt.xlabel('Number of Days')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Explore class distribution
    class_counts = df['class'].value_counts()
    print("\nClass distribution:")
    display(class_counts)

    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.show()

    return cases, days

def decode_rle(rle, shape=(266, 266)):
    """Decode RLE to mask"""
    if pd.isna(rle):
        return np.zeros(shape)

    # Check if the string is empty
    if not rle or rle.strip() == '':
        return np.zeros(shape)

    try:
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        return img.reshape(shape)
    except Exception as e:
        print(f"Error decoding RLE: {str(e)}, RLE: {rle}")
        return np.zeros(shape)

def extract_metadata_from_filepath(file_path):
    """Extract metadata from image filepath"""
    # Example path: .../case101/case101_day20/scans/slice_0001_266_266_1.50_1.50.png
    parts = file_path.split('/')

    case_folder = parts[-4]  # case101
    day_folder = parts[-3]   # case101_day20
    slice_file = parts[-1]   # slice_0001_266_266_1.50_1.50.png

    # Extract case number
    case_match = re.search(r'case(\d+)', case_folder)
    case_id = int(case_match.group(1)) if case_match else None

    # Extract day number
    day_match = re.search(r'day(\d+)', day_folder)
    day = int(day_match.group(1)) if day_match else None

    # Extract slice number
    slice_match = re.search(r'slice_(\d+)', slice_file)
    slice_id = int(slice_match.group(1)) if slice_match else None

    # Extract dimensions and pixel spacing
    dimensions_match = re.search(r'(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)', slice_file)
    if dimensions_match:
        width = int(dimensions_match.group(1))
        height = int(dimensions_match.group(2))
        x_spacing = float(dimensions_match.group(3))
        y_spacing = float(dimensions_match.group(4))
    else:
        width, height, x_spacing, y_spacing = None, None, None, None

    return {
        'case_id': case_id,
        'day': day,
        'slice_id': slice_id,
        'width': width,
        'height': height,
        'x_spacing': x_spacing,
        'y_spacing': y_spacing
    }

def visualize_images_and_masks(df, num_samples=10):
    """Visualize random images and their masks"""
    # Get unique (case, day, slice) combinations
    id_groups = df.groupby(['case', 'day', 'slice'])
    ids = list(id_groups.groups.keys())

    # Select random samples
    np.random.seed(42)
    sample_ids = np.random.choice(len(ids), min(num_samples, len(ids)), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))

    for i, idx in enumerate(sample_ids):
        case, day, slice_id = ids[idx]

        # Filter data for this (case, day, slice)
        sample = df[(df['case'] == case) & (df['day'] == day) & (df['slice'] == slice_id)]

        # Construct image path
        image_path = f"{TRAIN_DIR}/case{case}/case{case}_day{day}/scans/slice_{slice_id:04d}_266_266_1.50_1.50.png"

        try:
            # Load image (16-bit grayscale)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                continue

            # Scale image for display (16-bit to 8-bit)
            image_scaled = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            # Display image
            axes[i, 0].imshow(image_scaled, cmap='gray')
            axes[i, 0].set_title(f"Case {case}, Day {day}, Slice {slice_id}")
            axes[i, 0].axis('off')

            # Colors for masks
            colors = {'large_bowel': [1, 0, 0], 'small_bowel': [0, 1, 0], 'stomach': [0, 0, 1]}

            # Create combined colored mask
            colored_mask = np.zeros((266, 266, 3))

            # Display individual class masks
            for j, (_, row) in enumerate(sample.iterrows(), 1):
                if j <= 3:  # We have space for 3 masks
                    class_name = row['class']
                    rle = row['segmentation']

                    # Decode RLE to mask
                    mask = decode_rle(rle)

                    # Display individual mask
                    axes[i, j].imshow(mask, cmap='gray')
                    axes[i, j].set_title(f"{class_name} Mask")
                    axes[i, j].axis('off')

                    # Add to colored mask
                    if not pd.isna(rle):
                        for c in range(3):
                            colored_mask[:, :, c] = np.maximum(colored_mask[:, :, c], mask * colors[class_name][c])
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

    plt.tight_layout()
    plt.show()

def analyze_image_properties():
    """Analyze image properties like dimensions and pixel spacing"""
    sample_files = []

    # Get sample files
    for root, _, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.endswith('.png'):
                sample_files.append(os.path.join(root, file))
                if len(sample_files) >= 100:  # Limit to 100 files for efficiency
                    break
        if len(sample_files) >= 100:
            break

    # Extract metadata
    metadata = []
    for file_path in sample_files:
        meta = extract_metadata_from_filepath(file_path)
        metadata.append(meta)

    meta_df = pd.DataFrame(metadata)

    print("Image metadata statistics:")
    display(meta_df.describe())

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(meta_df['width'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Image Width')

    sns.histplot(meta_df['height'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Image Height')

    sns.histplot(meta_df['x_spacing'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of X Spacing')

    sns.histplot(meta_df['y_spacing'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Y Spacing')

    plt.tight_layout()
    plt.show()

def analyze_masks(df):
    """Analyze mask properties"""
    # Calculate non-empty masks
    df['has_mask'] = ~df['segmentation'].isna()

    # Group by class and calculate percentage with masks
    mask_by_class = df.groupby('class')['has_mask'].mean() * 100

    print("Percentage of non-empty masks by class:")
    display(mask_by_class)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mask_by_class.index, y=mask_by_class.values)
    plt.title('Percentage of Non-empty Masks by Class')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate mask sizes (for non-empty masks)
    def get_mask_size(rle):
        if pd.isna(rle):
            return 0
        s = rle.split()
        lengths = np.asarray(s[1:][::2], dtype=int)
        return lengths.sum()

    df['mask_size'] = df['segmentation'].apply(get_mask_size)

    # Analyze mask sizes by class
    mask_sizes = df[df['mask_size'] > 0].groupby('class')['mask_size'].describe()

    print("\nMask size statistics by class:")
    display(mask_sizes)

    # Plot mask size distributions
    plt.figure(figsize=(12, 6))
    for class_name in df['class'].unique():
        sns.kdeplot(df[df['class'] == class_name]['mask_size'], label=class_name)

    plt.title('Distribution of Mask Sizes by Class')
    plt.xlabel('Mask Size (pixels)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main_eda():
    """Main function to run the EDA"""
    print("=" * 50)
    print("Starting Exploratory Data Analysis for GI Tract Image Segmentation")
    print("=" * 50)

    # Load data
    df = load_data()

    # Explore dataset structure
    cases, days = explore_dataset_structure(df)

    # Analyze image properties
    analyze_image_properties()

    # Analyze masks
    analyze_masks(df)

    # Visualize images and masks
    print("\nVisualizing sample images and their masks:")
    visualize_images_and_masks(df, num_samples=5)

    print("\nEDA completed!")

if __name__ == "__main__":
    main_eda()
