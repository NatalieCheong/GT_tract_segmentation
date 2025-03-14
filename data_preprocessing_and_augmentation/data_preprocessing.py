import os
import gc
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import albumentations as A
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

# Global variables for data loaders
train_loader = None
val_loader = None

def rle_decode(rle, shape=(266, 266)):
    """
    Decode RLE encoded mask

    Args:
        rle: Run-length encoded mask string
        shape: Output mask shape

    Returns:
        Decoded mask as numpy array
    """
    if pd.isna(rle):
        return np.zeros(shape, dtype=np.uint8)

    # Check if the string is empty
    if not rle or rle.strip() == '':
        return np.zeros(shape, dtype=np.uint8)

    try:
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            if lo < img.shape[0] and hi <= img.shape[0]:
                img[lo:hi] = 1

        return img.reshape(shape)
    except Exception as e:
        print(f"Error decoding RLE: {str(e)}, RLE: {rle}")
        return np.zeros(shape, dtype=np.uint8)

def rle_encode(img):
    """
    Encode mask as RLE

    Args:
        img: Binary mask as numpy array

    Returns:
        RLE encoded string
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def load_image(path):
    """
    Load 16-bit grayscale image and normalize

    Args:
        path: Path to image file

    Returns:
        Normalized image as numpy array
    """
    try:
        # Load 16-bit image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Warning: Failed to load image {path}")
            return np.zeros((266, 266), dtype=np.float32)

        # Clip and normalize to [0, 1]
        min_val = np.percentile(img, 1)
        max_val = np.percentile(img, 99)
        img = np.clip(img, min_val, max_val)
        img = ((img - min_val) / (max_val - min_val)).astype(np.float32)

        return img
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        return np.zeros((266, 266), dtype=np.float32)

def prepare_data(df):
    """
    Prepare and organize the dataset for training

    Args:
        df: DataFrame containing the dataset

    Returns:
        Processed DataFrame with grouped mask information
    """
    # If necessary columns don't exist, extract them from id
    if 'case' not in df.columns or 'day' not in df.columns or 'slice' not in df.columns:
        df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
        df['case'] = df['case'].astype(int)
        df['day'] = df['day'].astype(int)
        df['slice'] = df['slice'].astype(int)

    # Group by case, day, slice and aggregate masks for each class
    # Fixed: using a list instead of a tuple for column selection
    grouped_df = df.groupby(['case', 'day', 'slice'])[['class', 'segmentation']].apply(
        lambda x: x.set_index('class')['segmentation'].to_dict()
    ).reset_index()

    grouped_df.columns = ['case', 'day', 'slice', 'masks']

    # Add image paths
    grouped_df['image_path'] = grouped_df.apply(
        lambda row: f"{TRAIN_DIR}/case{row['case']}/case{row['case']}_day{row['day']}/scans/slice_{row['slice']:04d}_266_266_1.50_1.50.png",
        axis=1
    )

    # Add case_day for grouping in cross-validation
    grouped_df['case_day'] = grouped_df['case'].astype(str) + '_' + grouped_df['day'].astype(str)

    # Verify image paths exist
    grouped_df['exists'] = grouped_df['image_path'].apply(os.path.exists)
    if not grouped_df['exists'].all():
        print(f"Warning: {(~grouped_df['exists']).sum()} image paths do not exist!")
        grouped_df = grouped_df[grouped_df['exists']].reset_index(drop=True)

    # Remove the 'exists' column to save memory
    grouped_df = grouped_df.drop(columns=['exists'])

    # Run garbage collection
    gc.collect()

    return grouped_df

def create_folds(df, n_splits=5):
    """
    Create cross-validation folds based on case_day groups

    Args:
        df: Processed DataFrame
        n_splits: Number of folds

    Returns:
        DataFrame with fold column added
    """
    # Create group k-fold based on case_day
    gkf = GroupKFold(n_splits=n_splits)

    # Add fold column
    df['fold'] = -1

    # Assign folds
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df['case_day'])):
        df.loc[val_idx, 'fold'] = fold

    # Run garbage collection
    gc.collect()

    return df

class GITractDataset(Dataset):
    """
    Dataset class for GI Tract Image Segmentation
    """
    def __init__(self, df, transforms=None, test=False):
        """
        Initialize dataset

        Args:
            df: DataFrame with dataset information
            transforms: Albumentations transformations
            test: Whether this is a test dataset
        """
        self.df = df
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get dataset item"""
        row = self.df.iloc[idx]

        # Load image
        image_path = row['image_path']
        image = load_image(image_path)

        # Create mask: channels are [large_bowel, small_bowel, stomach]
        mask = np.zeros((266, 266, 3), dtype=np.float32)

        if not self.test:
            if 'large_bowel' in row['masks'] and not pd.isna(row['masks'].get('large_bowel')):
                mask[:, :, 0] = rle_decode(row['masks']['large_bowel'])
            if 'small_bowel' in row['masks'] and not pd.isna(row['masks'].get('small_bowel')):
                mask[:, :, 1] = rle_decode(row['masks']['small_bowel'])
            if 'stomach' in row['masks'] and not pd.isna(row['masks'].get('stomach')):
                mask[:, :, 2] = rle_decode(row['masks']['stomach'])

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim for grayscale
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        return {
            'image': image,
            'mask': mask,
            'case': row['case'],
            'day': row['day'],
            'slice': row['slice']
        }

def get_transforms(phase):
    """
    Get augmentation transforms

    Args:
        phase: 'train' or 'valid'

    Returns:
        Albumentations transforms
    """
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            ], p=0.5),
        ], p=1.0)
    else:
        return A.Compose([
            # No augmentation for validation phase
        ], p=1.0)

def create_data_loaders(df, fold=0, batch_size=16):
    """
    Create training and validation data loaders

    Args:
        df: DataFrame with dataset information
        fold: Validation fold
        batch_size: Batch size

    Returns:
        train_loader, val_loader
    """
    global train_loader, val_loader

    # Create train/validation split
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    # Create datasets
    train_dataset = GITractDataset(train_df, transforms=get_transforms('train'))
    val_dataset = GITractDataset(val_df, transforms=get_transforms('valid'))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train loader: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"Val loader: {len(val_loader)} batches, {len(val_dataset)} samples")

    return train_loader, val_loader

def visualize_processed_data(df, n_samples=10):
    """
    Visualize processed images and their masks

    Args:
        df: Processed DataFrame
        n_samples: Number of samples to visualize
    """
    # Select random samples
    random.seed(42)
    sample_indices = random.sample(range(len(df)), min(n_samples, len(df)))

    # Create dataset with validation transforms
    dataset = GITractDataset(df.iloc[sample_indices], transforms=get_transforms('valid'))

    # Create figure
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))

    # Class names for display
    class_names = ['large_bowel', 'small_bowel', 'stomach']

    for i in range(n_samples):
        data = dataset[i]

        # Extract data
        image = data['image'].squeeze().numpy()
        masks = data['mask'].numpy()

        # Display original image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"Case {data['case']}, Day {data['day']}, Slice {data['slice']}")
        axes[i, 0].axis('off')

        # Display individual masks
        for j in range(3):
            axes[i, j+1].imshow(masks[j], cmap='viridis')
            axes[i, j+1].set_title(f"{class_names[j]} Mask")
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()

    # Display augmented training examples
    train_transforms = get_transforms('train')

    fig, axes = plt.subplots(5, 4, figsize=(20, 25))

    # Select one sample for augmentation examples
    sample_idx = sample_indices[0]
    sample_row = df.iloc[sample_idx]

    # Load original image and mask
    image = load_image(sample_row['image_path'])

    mask = np.zeros((266, 266, 3), dtype=np.float32)
    if 'large_bowel' in sample_row['masks'] and not pd.isna(sample_row['masks'].get('large_bowel')):
        mask[:, :, 0] = rle_decode(sample_row['masks']['large_bowel'])
    if 'small_bowel' in sample_row['masks'] and not pd.isna(sample_row['masks'].get('small_bowel')):
        mask[:, :, 1] = rle_decode(sample_row['masks']['small_bowel'])
    if 'stomach' in sample_row['masks'] and not pd.isna(sample_row['masks'].get('stomach')):
        mask[:, :, 2] = rle_decode(sample_row['masks']['stomach'])

    # Original image and combined mask
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    combined_mask = np.zeros((266, 266, 3), dtype=np.float32)
    combined_mask[:, :, 0] = mask[:, :, 0]  # Red for large bowel
    combined_mask[:, :, 1] = mask[:, :, 1]  # Green for small bowel
    combined_mask[:, :, 2] = mask[:, :, 2]  # Blue for stomach

    axes[0, 1].imshow(combined_mask)
    axes[0, 1].set_title("Original Combined Mask")
    axes[0, 1].axis('off')

    # Apply different augmentations
    for i in range(1, 5):
        augmented = train_transforms(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        axes[i, 0].imshow(aug_image, cmap='gray')
        axes[i, 0].set_title(f"Augmented Image {i}")
        axes[i, 0].axis('off')

        # Display individual augmented masks
        for j in range(3):
            axes[i, j+1].imshow(aug_mask[:, :, j], cmap='viridis')
            axes[i, j+1].set_title(f"Augmented {class_names[j]} Mask")
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()

    # Free up memory
    plt.close('all')
    gc.collect()

def main_preprocessing():
    """Main function for data preprocessing"""
    print("=" * 50)
    print("Data Preprocessing for GI Tract Image Segmentation")
    print("=" * 50)

    # Load data
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded data with {len(df)} rows")

    # Extract case, day, and slice from the id column
    # Format: case{case}_day{day}_slice_{slice}
    if 'id' in df.columns:
        df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
        df['case'] = df['case'].astype(int)
        df['day'] = df['day'].astype(int)
        df['slice'] = df['slice'].astype(int)

    # Free up memory
    gc.collect()

    # Prepare data
    print("\nPreparing data...")
    processed_df = prepare_data(df)
    print(f"Processed data with {len(processed_df)} rows")

    # Free up memory - we don't need the original dataframe anymore
    del df
    gc.collect()

    # Create folds
    print("\nCreating cross-validation folds...")
    folded_df = create_folds(processed_df, n_splits=5)

    # Free up memory - processed_df is now replaced by folded_df
    del processed_df
    gc.collect()

    # Print fold distribution
    fold_counts = folded_df['fold'].value_counts().sort_index()
    print("\nFold distribution:")
    for fold, count in fold_counts.items():
        print(f"Fold {fold}: {count} samples")

    # Count distribution of classes
    class_counts = {
        'large_bowel': sum('large_bowel' in masks and not pd.isna(masks.get('large_bowel', np.nan))
                          for masks in folded_df['masks']),
        'small_bowel': sum('small_bowel' in masks and not pd.isna(masks.get('small_bowel', np.nan))
                          for masks in folded_df['masks']),
        'stomach': sum('stomach' in masks and not pd.isna(masks.get('stomach', np.nan))
                      for masks in folded_df['masks'])
    }

    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples ({count/len(folded_df)*100:.2f}%)")

    # Visualize processed data
    print("\nVisualizing processed data:")
    visualize_processed_data(folded_df, n_samples=10)

    # Create data loaders for fold 0
    print("\nCreating data loaders for fold 0:")
    train_loader, val_loader = create_data_loaders(folded_df, fold=0, batch_size=16)

    print("\nData preprocessing completed!")

    # Additional memory usage information
    print("\nMemory usage information:")
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

    # Return the dataset ready for training, but only if needed
    return folded_df, train_loader, val_loader

if __name__ == "__main__":
    # Only create the full dataset when running this script directly
    folded_df, train_loader, val_loader = main_preprocessing()

    # Keep the data loaders and folded_df for the model training
    print("\nDataset and loaders are ready for model training.")
