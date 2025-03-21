import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import random
from tqdm.notebook import tqdm
from model import UNetWithResNetBackbone, create_model, calculate_combined_metric
from data_preprocessing import GITractDataset, get_transforms
from train import validate

def visualize_predictions(model, data_loader, num_samples=5, device='cpu'):
    """
    Visualize model predictions on random samples from the data loader

    Args:
        model: Trained segmentation model
        data_loader: DataLoader with samples to visualize
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    # Set model to evaluation mode
    model.eval()

    # Get random samples from the data loader
    all_samples = []
    for batch in data_loader:
        all_samples.extend([(batch['image'][i], batch['mask'][i],
                           batch['case'][i], batch['day'][i], batch['slice'][i])
                          for i in range(len(batch['image']))])
        if len(all_samples) >= 50:  # Collect a good pool to sample from
            break

    # Select random samples
    random.seed(42)  # For reproducibility
    selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))

    # Process each sample
    with torch.no_grad():
        for i, (image, mask, case, day, slice_num) in enumerate(selected_samples):
            # Add batch dimension
            image = image.unsqueeze(0).to(device)

            # Forward pass
            prediction = model(image)

            # Apply sigmoid to get probability maps
            prediction = torch.sigmoid(prediction)

            # Convert tensors to numpy arrays
            image = image.squeeze().cpu().numpy()
            mask = mask.cpu().numpy()
            prediction = prediction.squeeze().cpu().numpy()

            # Create figure for visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # Display original image
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title(f"Case {case}, Day {day}, Slice {slice_num}")
            axes[0, 0].axis('off')

            # Display ground truth masks
            class_names = ['Large Bowel', 'Small Bowel', 'Stomach']
            colors = ['red', 'green', 'blue']

            # Combined ground truth mask
            combined_mask = np.zeros((266, 266, 3))
            for c in range(3):
                axes[0, c+1].imshow(mask[c], cmap='viridis')
                axes[0, c+1].set_title(f"GT: {class_names[c]}")
                axes[0, c+1].axis('off')

                # Add to combined mask
                combined_mask[:, :, c] = mask[c]

            # Combined predicted mask
            combined_pred = np.zeros((266, 266, 3))
            for c in range(3):
                # Apply threshold to get binary mask
                pred_binary = (prediction[c] > 0.5).astype(np.float32)

                axes[1, c+1].imshow(prediction[c], cmap='viridis')
                axes[1, c+1].set_title(f"Pred: {class_names[c]}")
                axes[1, c+1].axis('off')

                # Add to combined predicted mask
                combined_pred[:, :, c] = prediction[c]

            # Display original image overlaid with prediction
            img_with_mask = np.copy(image)
            img_with_mask = np.stack([img_with_mask, img_with_mask, img_with_mask], axis=-1)

            # Create overlay for prediction
            for c in range(3):
                # Add colored overlay for prediction
                overlay = np.zeros_like(img_with_mask)
                overlay[:, :, c] = prediction[c] * 0.7  # Adjust opacity

                # Combine with image
                img_with_mask += overlay

            # Normalize to [0, 1]
            img_with_mask = np.clip(img_with_mask, 0, 1)

            # Display overlaid image
            axes[1, 0].imshow(img_with_mask)
            axes[1, 0].set_title("Image with Prediction")
            axes[1, 0].axis('off')

            # Show combined masks
            axes[0, 3].imshow(combined_mask)
            axes[0, 3].set_title("GT: Combined")
            axes[0, 3].axis('off')

            axes[1, 3].imshow(combined_pred)
            axes[1, 3].set_title("Pred: Combined")
            axes[1, 3].axis('off')

            plt.tight_layout()
            plt.show()

            # Calculate Dice score for this sample
            dice_scores = []
            for c in range(3):
                pred_binary = (prediction[c] > 0.5).astype(np.float32)
                intersection = np.sum(pred_binary * mask[c])
                dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(mask[c]) + 1e-8)
                dice_scores.append(dice)

            print(f"Dice scores - Large Bowel: {dice_scores[0]:.4f}, Small Bowel: {dice_scores[1]:.4f}, Stomach: {dice_scores[2]:.4f}")
            print(f"Average Dice: {np.mean(dice_scores):.4f}")
            print("=" * 50)

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

def load_model_and_visualize(model_path='gi_tract_segmentation_model.pth', num_samples=5):
    """
    Load the trained model and visualize predictions

    Args:
        model_path: Path to the saved model weights
        num_samples: Number of samples to visualize
    """
    print("Loading model and visualizing predictions...")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model architecture
    model = UNetWithResNetBackbone(backbone='resnet34', num_classes=3, pretrained=False)

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Make sure val_loader is defined
    if 'val_loader' not in globals() or globals()['val_loader'] is None:
        print("WARNING: val_loader not found. Make sure it's defined.")
        return

    # Visualize predictions
    visualize_predictions(model, val_loader, num_samples=num_samples, device=device)

    print("Visualization completed!")

# Run the visualization
if __name__ == "__main__":
    # Make sure val_loader is defined before running this
    load_model_and_visualize(num_samples=10)
