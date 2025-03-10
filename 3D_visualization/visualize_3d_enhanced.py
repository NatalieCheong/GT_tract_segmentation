import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

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
            if lo < img.shape[0] and hi <= img.shape[0]:
                img[lo:hi] = 1

        return img.reshape(shape)
    except Exception as e:
        print(f"Error decoding RLE: {str(e)}")
        return np.zeros(shape)

def get_organ_masks_for_case(df, case_id, day_id, organ_class, max_slices=15):
    """Get organ masks for a specific case, day, and class"""
    # Get slices with masks for this case/day/class
    mask_slices = df[(df['case'] == case_id) &
                    (df['day'] == day_id) &
                    (df['class'] == organ_class) &
                    (~df['segmentation'].isna())]['slice'].unique()

    if len(mask_slices) == 0:
        print(f"No masks found for Case {case_id}, Day {day_id}, Class {organ_class}")
        return [], []

    mask_slices = sorted(mask_slices)
    print(f"Found {len(mask_slices)} slices with {organ_class} masks.")

    # If too many slices, select a subset
    if len(mask_slices) > max_slices:
        # Choose evenly spaced slices
        indices = np.linspace(0, len(mask_slices)-1, max_slices, dtype=int)
        mask_slices = [mask_slices[i] for i in indices]

    # Get masks for selected slices
    masks = []
    slice_nums = []

    for slice_id in mask_slices:
        # Get RLE
        mask_row = df[(df['case'] == case_id) &
                     (df['day'] == day_id) &
                     (df['slice'] == slice_id) &
                     (df['class'] == organ_class)]

        if len(mask_row) > 0 and not pd.isna(mask_row.iloc[0]['segmentation']):
            rle = mask_row.iloc[0]['segmentation']
            mask = decode_rle(rle)

            # If mask has content, add it
            if np.sum(mask) > 0:
                masks.append(mask)
                slice_nums.append(slice_id)

    return masks, slice_nums

def visualize_3d_bowel():
    """Create 3D visualizations focusing on bowel structures"""
    print("Loading data for 3D bowel visualization...")

    # Load the CSV
    df = pd.read_csv(TRAIN_CSV)

    # Extract case, day, slice from the id
    df[['case', 'day', 'slice']] = df['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
    df['case'] = df['case'].astype(int)
    df['day'] = df['day'].astype(int)
    df['slice'] = df['slice'].astype(int)

    # Find good cases for each bowel class
    bowel_classes = ['large_bowel', 'small_bowel']

    # Get mask counts for each case/day/class
    case_mask_counts = df[~df['segmentation'].isna()].groupby(['case', 'day', 'class']).size().reset_index(name='mask_count')
    good_cases = case_mask_counts[case_mask_counts['mask_count'] > 10].sort_values('mask_count', ascending=False)

    # Filter for bowel classes
    good_bowel_cases = good_cases[good_cases['class'].isin(bowel_classes)]

    # Display some example cases
    print("Top cases with good bowel mask coverage:")
    display(good_bowel_cases.head(6))

    # Create 3D visualizations for the top cases of each bowel type
    visualized_classes = {}

    for organ_class in bowel_classes:
        class_cases = good_bowel_cases[good_bowel_cases['class'] == organ_class]

        if len(class_cases) == 0:
            print(f"No good cases found for {organ_class}")
            continue

        # Take the top 3 cases or less if fewer are available
        top_cases = class_cases.head(min(3, len(class_cases)))

        # Process each case
        for _, case_info in top_cases.iterrows():
            case_id = case_info['case']
            day_id = case_info['day']

            # Skip if we've already visualized this class 3 times
            class_key = organ_class
            visualized_classes[class_key] = visualized_classes.get(class_key, 0) + 1
            if visualized_classes[class_key] > 3:
                continue

            print(f"\nCreating 3D visualization for {organ_class}, Case {case_id}, Day {day_id}")

            # Get masks for this case
            masks, slice_nums = get_organ_masks_for_case(df, case_id, day_id, organ_class)

            if len(masks) < 3:
                print(f"Not enough masks with content for 3D visualization")
                continue

            # Create 3D visualization
            create_3d_bowel_visualization(masks, slice_nums, case_id, day_id, organ_class)

def create_3d_bowel_visualization(masks, slice_nums, case_id, day_id, organ_class):
    """Create 3D visualization for bowel structures"""
    if len(masks) < 3:
        print("Not enough masks for 3D visualization")
        return

    try:
        # Downsample masks for better performance
        downsampled_masks = []
        for mask in masks:
            # Reduce to 64x64 resolution
            mask_small = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)
            downsampled_masks.append(mask_small)

        # Create color mappings for different organs - using more vibrant colormaps
        if organ_class == 'large_bowel':
            main_color = 'red'
            cmap = plt.cm.plasma  # More vibrant and visible colormap
        else:  # small_bowel
            main_color = 'green'
            cmap = plt.cm.viridis  # More vibrant and visible colormap

        # VISUALIZATION 1: 3D Surface Plot with Slice Depth
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a mesh for each slice
        for i, (mask, slice_id) in enumerate(zip(downsampled_masks, slice_nums)):
            # Get mask coordinates
            y, x = np.where(mask > 0)

            if len(x) > 0 and len(y) > 0:
                # Create z-coordinates (all same value for a slice)
                z = np.ones_like(x) * i

                # Plot as scatter with more vivid colors and larger point size
                ax.scatter(x, y, z, c=z, cmap=cmap, s=50, alpha=0.9, edgecolors='k', linewidths=0.3)

                # Add a text label for the slice number
                if i % 5 == 0:  # Label every 5th slice to avoid clutter
                    ax.text(x.mean(), y.mean(), i, f"Slice {slice_id}",
                           fontsize=8, ha='center', va='center')

        # Set title and labels
        ax.set_title(f"3D Visualization of {organ_class.replace('_', ' ').title()}\nCase {case_id}, Day {day_id}",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Slice Depth', fontsize=12)

        # Set a good viewing angle
        ax.view_init(elev=30, azim=45)

        # Add explanation
        fig.text(0.05, 0.02,
                f"This visualization shows the 3D structure of the {organ_class.replace('_', ' ')}.\n"
                f"Each point represents a pixel where the {organ_class.replace('_', ' ')} is present.\n"
                f"Vibrant colors indicate slice depth, with different colors representing different depths.",
                fontsize=11)

        # Add colorbar for depth reference
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.linspace(0, len(downsampled_masks), 10))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.1)
        cbar.set_label('Slice Depth', fontsize=12)

        plt.tight_layout()
        plt.show()

        # VISUALIZATION 2: 3D Isosurface Visualization with improved colors
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D volume
        volume = np.zeros((len(downsampled_masks), 64, 64))
        for i, mask in enumerate(downsampled_masks):
            volume[i] = mask

        # Determine colors for contours based on organ class
        if organ_class == 'large_bowel':
            contour_color = 'crimson'  # More vivid red
        else:  # small_bowel
            contour_color = 'limegreen'  # More vivid green

        # Find contours for each slice and plot
        for i, mask in enumerate(downsampled_masks):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) > 3:  # Only plot if enough points
                    # Reshape contour and create z-coordinates
                    contour = contour.squeeze()
                    z_vals = np.ones(contour.shape[0]) * i

                    # Plot contour with thicker, more visible lines
                    ax.plot(contour[:, 0], contour[:, 1], z_vals,
                           color=contour_color, linewidth=3, alpha=0.9)

                    # Fill contour with a semi-transparent surface
                    if len(contour) > 3:
                        try:
                            # Create a hull/mesh for the contour
                            from matplotlib.tri import Triangulation

                            # Add center point to improve triangulation
                            center_x, center_y = np.mean(contour[:, 0]), np.mean(contour[:, 1])
                            contour_with_center = np.vstack([contour, [center_x, center_y]])
                            z_with_center = np.append(z_vals, i)

                            # Create triangulation
                            triang = Triangulation(contour_with_center[:, 0], contour_with_center[:, 1])

                            # Plot triangulated surface with more vivid colors
                            if organ_class == 'large_bowel':
                                surface_color = 'indianred'  # More vivid red for surfaces
                            else:
                                surface_color = 'mediumseagreen'  # More vivid green for surfaces

                            ax.plot_trisurf(contour_with_center[:, 0], contour_with_center[:, 1], z_with_center,
                                          triangles=triang.triangles, color=surface_color, alpha=0.3)
                        except Exception as e:
                            # If triangulation fails, just continue without the surface
                            pass

        # Set title and labels
        ax.set_title(f"3D Contour Visualization of {organ_class.replace('_', ' ').title()}\nCase {case_id}, Day {day_id}",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Slice Depth', fontsize=12)

        # Set a good viewing angle
        ax.view_init(elev=30, azim=135)

        # Add explanation
        fig.text(0.05, 0.02,
                f"This visualization shows the 3D contours of the {organ_class.replace('_', ' ')}.\n"
                f"Each slice is represented by its boundary, with semi-transparent surfaces connecting them.\n"
                f"This helps visualize the changing shape of the organ through the scan.",
                fontsize=11)

        plt.tight_layout()
        plt.show()

        # VISUALIZATION 3: 3D Volume with Transparent Voxels - enhanced colors
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Add padding to the volume to avoid edge artifacts
        volume_padded = np.pad(volume, pad_width=1, mode='constant')

        # Create custom colormap with transparency and more vivid colors
        if organ_class == 'large_bowel':
            rgba_colors = np.zeros(volume_padded.shape + (4,))
            # Bright red with varying transparency
            for i in range(volume_padded.shape[0]):
                alpha = 0.4 + 0.5 * (i / volume_padded.shape[0])  # Increasing alpha by depth
                mask = volume_padded[i] > 0
                rgba_colors[i, mask, 0] = 1.0  # Full red
                rgba_colors[i, mask, 1] = 0.2  # Slight green to enhance visibility
                rgba_colors[i, mask, 2] = 0.2  # Slight blue to enhance visibility
                rgba_colors[i, mask, 3] = alpha  # Alpha
        else:  # small_bowel
            rgba_colors = np.zeros(volume_padded.shape + (4,))
            # Bright green with varying transparency
            for i in range(volume_padded.shape[0]):
                alpha = 0.4 + 0.5 * (i / volume_padded.shape[0])  # Increasing alpha by depth
                mask = volume_padded[i] > 0
                rgba_colors[i, mask, 0] = 0.2  # Slight red to enhance visibility
                rgba_colors[i, mask, 1] = 1.0  # Full green
                rgba_colors[i, mask, 2] = 0.2  # Slight blue to enhance visibility
                rgba_colors[i, mask, 3] = alpha  # Alpha

        # Create voxel visualization with more visible edges
        ax.voxels(volume_padded > 0, facecolors=rgba_colors, edgecolor='k', linewidth=0.3)

        # Set title and labels
        ax.set_title(f"3D Volume Visualization of {organ_class.replace('_', ' ').title()}\nCase {case_id}, Day {day_id}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Slice Depth', fontsize=12)

        # Set a good viewing angle
        ax.view_init(elev=30, azim=30)

        # Add explanation
        fig.text(0.05, 0.02,
                f"This visualization shows the 3D volume of the {organ_class.replace('_', ' ')}.\n"
                f"Each cube represents a voxel (3D pixel) where the organ is present.\n"
                f"Transparency increases with depth to show the internal structure.",
                fontsize=11)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating 3D bowel visualization: {str(e)}")

def main_bowel_visualization():
    """Main function for 3D bowel visualization"""
    print("=" * 50)
    print("3D Bowel Structure Visualization")
    print("=" * 50)

    # Create 3D visualizations focusing on bowel structures
    visualize_3d_bowel()

    print("\nBowel visualization completed!")

if __name__ == "__main__":
    main_bowel_visualization()
