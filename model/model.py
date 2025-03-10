import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class UNetWithResNetBackbone(nn.Module):
    """
    UNet architecture with ResNet backbone for image segmentation.
    Includes padding to handle dimensions not divisible by 32.
    """

    def __init__(self, backbone='resnet34', num_classes=3, pretrained=True):
        """
        Initialize the model

        Args:
            backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50', etc.)
            num_classes: Number of output segmentation classes
            pretrained: Whether to use pretrained weights for backbone
        """
        super(UNetWithResNetBackbone, self).__init__()

        # Create the model using segmentation-models-pytorch for efficiency
        self.model = smp.Unet(
            encoder_name=backbone,        # backbone encoder name
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,                # input channels (1 for grayscale)
            classes=num_classes,          # number of output classes
            activation=None,              # no activation, we'll apply it in the loss function
        )

        # Calculate padding needed
        # UNet with ResNet backbone requires dimensions divisible by 32
        self.pad_h = (32 - 266 % 32) % 32
        self.pad_w = (32 - 266 % 32) % 32

        print(f"Model will use padding: h_pad={self.pad_h}, w_pad={self.pad_w}")

    def forward(self, x):
        """Forward pass with padding and cropping to handle dimensions"""
        batch_size, channels, height, width = x.shape

        # Apply padding to make dimensions divisible by 32
        # Pad with zeros evenly on both sides
        x_padded = F.pad(x, (self.pad_w//2, self.pad_w-self.pad_w//2,
                             self.pad_h//2, self.pad_h-self.pad_h//2))

        # Pass through model
        output_padded = self.model(x_padded)

        # Crop back to original dimensions
        output = output_padded[:, :,
                               self.pad_h//2:self.pad_h//2 + height,
                               self.pad_w//2:self.pad_w//2 + width]

        return output

class DiceCoefficient(nn.Module):
    """
    Dice coefficient for segmentation evaluation.
    Formula: 2 * |X ∩ Y| / (|X| + |Y|)
    Where X and Y are the predicted and ground truth masks.
    """

    def __init__(self, smooth=1.0):
        super(DiceCoefficient, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Flatten predicted and true masks
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate Dice
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return dice

class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss for segmentation.
    Combines Dice loss and Binary Cross-Entropy loss for better training
    of segmentation models.
    """

    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(mode='binary', smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true):
        # Calculate losses for each class separately
        total_loss = 0
        dice_scores = []

        # Handle multi-class segmentation by processing each channel
        for i in range(y_pred.shape[1]):
            # Get the i-th channel for predicted and true
            pred_channel = y_pred[:, i, :, :]
            true_channel = y_true[:, i, :, :]

            # Calculate Dice loss for this channel
            dice_loss = self.dice(pred_channel, true_channel)

            # Calculate BCE loss for this channel
            bce_loss = self.bce(pred_channel, true_channel)

            # Combine losses
            channel_loss = dice_loss + self.bce_weight * bce_loss

            # Add to total loss
            total_loss += channel_loss

            # Store dice score (1 - dice_loss) for monitoring
            dice_scores.append(1.0 - dice_loss.item())

        # Average over all classes
        return total_loss / y_pred.shape[1], dice_scores

def create_model(backbone='resnet34', num_classes=3, pretrained=True):
    """
    Create a UNet model with ResNet backbone

    Args:
        backbone: ResNet variant to use
        num_classes: Number of output segmentation classes
        pretrained: Whether to use pretrained weights

    Returns:
        model: The initialized model
    """
    model = UNetWithResNetBackbone(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    return model

def get_optimizer_and_scheduler(model, lr=3e-4, weight_decay=1e-6, epochs=30):
    """
    Get optimizer and learning rate scheduler

    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs

    Returns:
        optimizer: The optimizer
        scheduler: The learning rate scheduler
    """
    # Initialize optimizer with different parameters for backbone and decoder
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    return optimizer, scheduler

def get_3d_hausdorff_distance(pred_masks, true_masks, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate 3D Hausdorff distance between predicted and true 3D masks

    Args:
        pred_masks: Predicted masks (B, C, D, H, W) where D is the depth dimension
        true_masks: Ground truth masks (B, C, D, H, W)
        spacing: Voxel spacing in (z, y, x) order, defaults to (1.0, 1.0, 1.0)

    Returns:
        hausdorff_distance: Average 3D Hausdorff distance
    """
    from scipy.spatial.distance import directed_hausdorff

    # Validate input
    assert pred_masks.shape == true_masks.shape, "Predicted and ground truth masks must have the same shape"

    batch_size, num_classes, depth, height, width = pred_masks.shape
    hausdorff_distances = []

    for b in range(batch_size):
        for c in range(num_classes):
            # Get prediction and ground truth for current batch and class
            pred = pred_masks[b, c].cpu().numpy()
            true = true_masks[b, c].cpu().numpy()

            # Skip if either mask is empty
            if np.sum(pred) == 0 and np.sum(true) == 0:
                # Both empty, distance is 0
                hausdorff_distances.append(0.0)
                continue
            elif np.sum(pred) == 0 or np.sum(true) == 0:
                # One is empty, one is not - maximum distance
                hausdorff_distances.append(1.0)  # Normalized distance
                continue

            # Get coordinates of non-zero pixels
            pred_points = np.array(np.where(pred > 0.5)).T * spacing
            true_points = np.array(np.where(true > 0.5)).T * spacing

            if len(pred_points) == 0 or len(true_points) == 0:
                hausdorff_distances.append(1.0)  # Normalized distance
                continue

            # Calculate directed Hausdorff distances
            forward, _, _ = directed_hausdorff(pred_points, true_points)
            backward, _, _ = directed_hausdorff(true_points, pred_points)

            # Take maximum of both directions
            hausdorff = max(forward, backward)

            # Normalize to [0, 1] based on the diagonal of the volume
            volume_diagonal = np.sqrt(
                (depth * spacing[0])**2 + (height * spacing[1])**2 + (width * spacing[2])**2
            )
            normalized_hausdorff = min(hausdorff / volume_diagonal, 1.0)

            hausdorff_distances.append(normalized_hausdorff)

    # Return average Hausdorff distance
    return sum(hausdorff_distances) / len(hausdorff_distances) if hausdorff_distances else 1.0

def calculate_combined_metric(dice_score, hausdorff_distance, dice_weight=0.4, hausdorff_weight=0.6):
    """
    Calculate combined metric based on Dice coefficient and Hausdorff distance

    Args:
        dice_score: Dice coefficient (higher is better)
        hausdorff_distance: Hausdorff distance (lower is better)
        dice_weight: Weight for Dice score in combined metric
        hausdorff_weight: Weight for Hausdorff distance in combined metric

    Returns:
        combined_score: Combined metric
    """
    # Convert Hausdorff distance to a score (1 - distance) so higher is better
    hausdorff_score = 1.0 - hausdorff_distance

    # Calculate weighted combination
    combined_score = (dice_weight * dice_score) + (hausdorff_weight * hausdorff_score)

    return combined_score
