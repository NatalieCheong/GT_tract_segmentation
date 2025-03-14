import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm.notebook import tqdm
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

# Color maps for different organs
ORGAN_COLORS = {
    'large_bowel': 'coral',
    'small_bowel': 'springgreen',
    'stomach': 'dodgerblue'
}

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
            if lo < img.shape[0] and hi <= img.shape[0]:  # Add bounds check
                img[lo:hi] = 1

        return img.reshape(shape)
    except Exception as e:
        print(f"Error decoding RLE: {str(e)}, RLE: {rle}")
        return np.zeros(shape)

def load_image(path):
    """Load and normalize image"""
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not load image from {path}")
            return np.zeros((266, 266), dtype=np.uint8)

        # Normalize to 0-255 for display
        if img.max() > 0:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        return img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return np.zeros((266, 266), dtype=np.uint8)

def create_enhanced_3d_visualizations(masks_3d, case_id, day_id, organ_class):
    """Create enhanced 3D visualizations for a set of masks"""
    if len(masks_3d) < 3:
        print("Not enough slices for 3D visualization")
        return

    try:
        # Create a downsampled version for faster visualization
        downsampled_masks = []
        for mask in masks_3d:
            # Downsample to 64x64
            mask_small = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)
            downsampled_masks.append(mask_small)

        # Create 3D array
        mask_volume = np.stack(downsampled_masks, axis=0)

        # Get organ-specific color
        color = ORGAN_COLORS.get(organ_class, 'red')

        # VISUALIZATION 1: Enhanced 3D Surface Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Find all points where mask is positive
        z, y, x = np.where(mask_volume > 0)

        if len(z) > 0:  # Only plot if we have points
            # Create a scatter plot with alpha based on density
            scatter = ax.scatter(
                z, x, y,  # Note the reordering to make the visualization more intuitive
                c=z,      # Color by slice (depth)
                cmap=cm.coolwarm,
                marker='o',
                s=20,     # Point size
                alpha=0.7,
                edgecolors='none'
            )

            # Add labels and title
            ax.set_title(f"3D {organ_class.replace('_', ' ').title()} Visualization\nCase {case_id}, Day {day_id}",
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Slice Number (Depth)', fontsize=12, labelpad=10)
            ax.set_ylabel('Width', fontsize=12, labelpad=10)
            ax.set_zlabel('Height', fontsize=12, labelpad=10)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=8)
            cbar.set_label('Slice Depth', fontsize=12)

            # Adjust the view angle for better visualization
            ax.view_init(elev=30, azim=45)

            # Add a text explanation
            fig.text(0.02, 0.02,
                    f"This visualization shows the 3D structure of the {organ_class.replace('_', ' ')}.\n"
                    "Each point represents a pixel where the organ is present.\n"
                    "Color indicates the slice depth (blue → red = shallow → deep).",
                    fontsize=11, wrap=True)
        else:
            ax.text(0.5, 0.5, 0.5, "No data points to visualize",
                   ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plt.show()

        # VISUALIZATION 2: Multi-slice 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get dimensions
        n_slices, height, width = mask_volume.shape

        # Create grid for each slice
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Plot each slice with its mask
        for i, mask in enumerate(downsampled_masks):
            # Skip empty slices
            if not np.any(mask):
                continue

            # Plot the slice surface
            z = np.ones_like(mask) * i
            ax.plot_surface(
                z, x, y,  # Slice depth, X coordinate, Y coordinate
                rstride=2, cstride=2,  # Stride for better performance
                facecolors=cm.viridis(mask),  # Color based on mask values
                alpha=0.5,  # Transparency
                shade=True,
                edgecolor='none'
            )

            # Add contour lines for each slice
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 5:  # Only plot if there are enough points
                    contour = contour.squeeze()
                    z_contour = np.ones(contour.shape[0]) * i
                    ax.plot(z_contour, contour[:, 0], contour[:, 1], color=color, linewidth=2)

        # Add explanatory text
        ax.set_title(f"3D Multi-slice View of {organ_class.replace('_', ' ').title()}\nCase {case_id}, Day {day_id}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Slice Number (Depth)', fontsize=12, labelpad=10)
        ax.set_ylabel('Width', fontsize=12, labelpad=10)
        ax.set_zlabel('Height', fontsize=12, labelpad=10)

        # Add explanation
        fig.text(0.02, 0.02,
                f"This visualization shows each slice as a semi-transparent surface.\n"
                f"The {color} lines outline the {organ_class.replace('_', ' ')} boundaries in each slice.\n"
                "By stacking the slices, you can see how the organ's shape changes through the scan.",
                fontsize=11, wrap=True)

        # Adjust view
        ax.view_init(elev=35, azim=45)

        plt.tight_layout()
        plt.show()

        # VISUALIZATION 3: Volume Rendering
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create 3D volume with voxels
        mask_volume_padded = np.pad(mask_volume, pad_width=1, mode='constant')

        # Create a colormap with organ-specific color and transparency
        rgba_colors = np.zeros(mask_volume_padded.shape + (4,))

        # Set color and alpha for the organ
        if organ_class == 'large_bowel':
            rgba_colors[mask_volume_padded > 0] = [1, 0.4, 0.4, 0.6]  # Semi-transparent red
        elif organ_class == 'small_bowel':
            rgba_colors[mask_volume_padded > 0] = [0.4, 1, 0.4, 0.6]  # Semi-transparent green
        else:  # stomach
            rgba_colors[mask_volume_padded > 0] = [0.4, 0.4, 1, 0.6]  # Semi-transparent blue

        # Create voxel visualization
        ax.voxels(mask_volume_padded, facecolors=rgba_colors, edgecolors='k', linewidth=0.1)

        # Add title and labels
        ax.set_title(f"3D Volume Rendering of {organ_class.replace('_', ' ').title()}\nCase {case_id}, Day {day_id}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Slice Number (Depth)', fontsize=12, labelpad=10)
        ax.set_ylabel('Width', fontsize=12, labelpad=10)
        ax.set_zlabel('Height', fontsize=12, labelpad=10)

        # Add explanatory text
        fig.text(0.02, 0.02,
                f"This visualization shows the 3D volume of the {organ_class.replace('_', ' ')}.\n"
                "Each cube (voxel) represents a 3D pixel where the organ is present.\n"
                "This helps understand the overall 3D shape of the organ across multiple slices.",
                fontsize=11, wrap=True)

        # Adjust view
        ax.view_init(elev=30, azim=30)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating 3D visualization: {str(e)}")

def visualize_image_set():
    """Visualize a set of images and masks from specific cases"""
    # Load the CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total rows in CSV: {len(df)}")

    # Extract case, day, slice from the id
    df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
    df['case'] = df['case'].astype(int)
    df['day'] = df['day'].astype(int)
    df['slice'] = df['slice'].astype(int)

    # Find cases with good mask coverage
    case_mask_counts = df[~df['segmentation'].isna()].groupby(['case', 'day', 'class']).size().reset_index(name='mask_count')
    good_cases = case_mask_counts[case_mask_counts['mask_count'] > 5].sort_values('mask_count', ascending=False)

    print("Cases with good mask coverage:")
    display(good_cases.head(10))

    # Choose the top case for each class
    selected_cases = []
    for organ_class in ['large_bowel', 'small_bowel', 'stomach']:
        class_cases = good_cases[good_cases['class'] == organ_class]
        if len(class_cases) > 0:
            selected_cases.append(class_cases.iloc[0])

    # If no good cases found, use default cases
    if len(selected_cases) == 0:
        print("No cases with good mask coverage found. Using default examples.")
        # These are placeholder values
        selected_cases = [
            {'case': 123, 'day': 20, 'class': 'large_bowel'},
            {'case': 126, 'day': 30, 'class': 'small_bowel'},
            {'case': 148, 'day': 53, 'class': 'stomach'}
        ]

    # For each selected case
    for case_info in selected_cases:
        case_id = case_info['case']
        day_id = case_info['day']
        organ_class = case_info['class']

        print(f"\nVisualizing {organ_class} for Case {case_id}, Day {day_id}")

        # Get slices with masks for this case/day/class
        case_slices = df[(df['case'] == case_id) &
                          (df['day'] == day_id) &
                          (df['class'] == organ_class) &
                          (~df['segmentation'].isna())]['slice'].unique()

        if len(case_slices) == 0:
            print(f"No slices with masks found for this case/day/class.")
            continue

        case_slices = sorted(case_slices)
        print(f"Found {len(case_slices)} slices with masks.")

        # Select a subset of slices for visualization
        if len(case_slices) > 5:
            selected_slices = np.linspace(0, len(case_slices)-1, 5, dtype=int)
            visualize_slices = [case_slices[i] for i in selected_slices]
        else:
            visualize_slices = case_slices

        # Create a figure
        fig, axes = plt.subplots(2, len(visualize_slices), figsize=(4*len(visualize_slices), 8))

        # For each slice
        masks_3d = []
        for i, slice_id in enumerate(visualize_slices):
            # Construct image path
            img_path = f"{TRAIN_DIR}/case{case_id}/case{case_id}_day{day_id}/scans/slice_{slice_id:04d}_266_266_1.50_1.50.png"

            # Load image
            img = load_image(img_path)

            # Display image
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Slice {slice_id}")
            axes[0, i].axis('off')

            # Get mask RLE
            mask_row = df[(df['case'] == case_id) &
                          (df['day'] == day_id) &
                          (df['slice'] == slice_id) &
                          (df['class'] == organ_class)]

            if len(mask_row) > 0 and not pd.isna(mask_row.iloc[0]['segmentation']):
                rle = mask_row.iloc[0]['segmentation']
                mask = decode_rle(rle)
                masks_3d.append(mask)
            else:
                mask = np.zeros((266, 266), dtype=np.uint8)
                masks_3d.append(mask)

            # Display mask
            axes[1, i].imshow(mask, cmap='viridis')
            axes[1, i].set_title(f"{organ_class} Mask")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

        # Create enhanced 3D visualizations
        if len(masks_3d) >= 3:
            create_enhanced_3d_visualizations(masks_3d, case_id, day_id, organ_class)

def visualize_single_cases():
    """Visualize a single slice with different organs"""
    # Load the CSV
    df = pd.read_csv(TRAIN_CSV)

    # Extract case, day, slice from the id
    df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
    df['case'] = df['case'].astype(int)
    df['day'] = df['day'].astype(int)
    df['slice'] = df['slice'].astype(int)

    # Find slices with all three organs
    slice_counts = df[~df['segmentation'].isna()].groupby(['case', 'day', 'slice']).size()
    good_slices = slice_counts[slice_counts == 3].reset_index()

    if len(good_slices) == 0:
        print("No slices with all three organs found.")
        # Try to find slices with at least two organs
        good_slices = slice_counts[slice_counts >= 2].reset_index().head(1)
        if len(good_slices) == 0:
            print("No slices with multiple organs found either.")
            return

    print(f"Found {len(good_slices)} slices with all three organs.")

    # Select a random good slice
    selected_slice = good_slices.iloc[0]
    case_id = selected_slice['case']
    day_id = selected_slice['day']
    slice_id = selected_slice['slice']

    print(f"Visualizing Case {case_id}, Day {day_id}, Slice {slice_id}")

    # Construct image path
    img_path = f"{TRAIN_DIR}/case{case_id}/case{case_id}_day{day_id}/scans/slice_{slice_id:04d}_266_266_1.50_1.50.png"

    # Load image
    img = load_image(img_path)

    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Display original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # Create a combined mask
    combined_mask = np.zeros((266, 266, 3), dtype=np.float32)

    # For each organ, get and display mask
    i = 1
    for organ_class in ['large_bowel', 'small_bowel', 'stomach']:
        # Get mask RLE
        mask_row = df[(df['case'] == case_id) &
                      (df['day'] == day_id) &
                      (df['slice'] == slice_id) &
                      (df['class'] == organ_class)]

        if len(mask_row) > 0 and not pd.isna(mask_row.iloc[0]['segmentation']):
            rle = mask_row.iloc[0]['segmentation']
            mask = decode_rle(rle)

            # Add to combined mask with organ-specific color
            if organ_class == 'large_bowel':
                combined_mask[:, :, 0] += mask  # Red channel
            elif organ_class == 'small_bowel':
                combined_mask[:, :, 1] += mask  # Green channel
            else:  # stomach
                combined_mask[:, :, 2] += mask  # Blue channel

            # Display individual mask
            cmap = plt.cm.get_cmap('viridis')
            axes[i].imshow(mask, cmap=cmap)
            axes[i].set_title(f"{organ_class.replace('_', ' ').title()}", fontsize=14)
            axes[i].axis('off')
            i += 1
        else:
            axes[i].text(0.5, 0.5, f"No {organ_class} mask",
                         ha='center', va='center', fontsize=14)
            axes[i].axis('off')
            i += 1

    # Normalize combined mask
    max_val = combined_mask.max()
    if max_val > 0:
        combined_mask = combined_mask / max_val

    # Display combined mask
    axes[4].imshow(combined_mask)
    axes[4].set_title("Combined Masks (RGB)", fontsize=14)

    # Add legend for combined mask
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=10, label='Large Bowel'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='green', markersize=10, label='Small Bowel'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', markersize=10, label='Stomach')
    ]
    axes[4].legend(handles=legend_elements, loc='lower right')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

def main_visualization():
    """Main function for enhanced visualization"""
    print("=" * 50)
    print("Enhanced GI Tract Image Visualization")
    print("=" * 50)

    # Visualize individual cases
    print("\nVisualizing individual slices with multiple organs:")
    visualize_single_cases()

    # Visualize image sets
    print("\nVisualizing image sets and 3D structures for each organ:")
    visualize_image_set()

    print("\nVisualization completed!")

if __name__ == "__main__":
    main_visualization()
