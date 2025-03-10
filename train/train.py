import torch
import gc
import numpy as np
from tqdm.notebook import tqdm
from model import UNetWithResNetBackbone, DiceBCELoss, create_model, get_optimizer_and_scheduler

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch with progress bar"""
    model.train()
    train_loss = 0.0
    dice_scores = []

    # Create a progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        images = batch['image'].to(device)
        targets = batch['mask'].to(device)

        # Forward pass
        outputs = model(images)
        loss, batch_dice_scores = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss += loss.item() * images.size(0)
        avg_dice = sum(batch_dice_scores) / len(batch_dice_scores)
        dice_scores.extend(batch_dice_scores)

        # Update progress bar description
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Dice': f"{avg_dice:.4f}"
        })

        # Periodic memory cleanup
        if batch_idx % 50 == 49:
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate average metrics
    train_loss = train_loss / len(train_loader.dataset)
    train_dice = sum(dice_scores) / len(dice_scores)

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return train_loss, train_dice

def validate(model, val_loader, criterion, device):
    """
    Validate the model with progress bar

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda or cpu)

    Returns:
        val_loss: Average validation loss
        val_dice: Average Dice coefficient on validation set
    """
    model.eval()
    val_loss = 0.0
    dice_scores = []

    # Create a progress bar for validation
    progress_bar = tqdm(val_loader, desc="Validating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            # Get data
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
            loss, batch_dice_scores = criterion(outputs, targets)

            # Update metrics
            val_loss += loss.item() * images.size(0)
            avg_dice = sum(batch_dice_scores) / len(batch_dice_scores)
            dice_scores.extend(batch_dice_scores)

            # Update progress bar description
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{avg_dice:.4f}"
            })

    # Calculate average metrics
    val_loss = val_loss / len(val_loader.dataset)
    val_dice = sum(dice_scores) / len(dice_scores)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss, val_dice

def train_model(model, train_loader, val_loader, epochs=30, lr=3e-4, device='cuda'):
    """
    Train the model

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        model: Trained model
        history: Training history
    """
    # Move model to device
    model = model.to(device)

    # Define loss function
    criterion = DiceBCELoss()

    # Define optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, lr=lr, epochs=epochs)

    # Initialize history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    # Training loop
    best_val_dice = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train for one epoch
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with Dice: {best_val_dice:.4f}")

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return model, history

def main_train_model(train_loader, val_loader, backbone='resnet34', epochs=15, lr=3e-4, batch_size=16):
    """
    Main function to train the model

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        backbone: ResNet backbone to use
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size

    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 50)
    print(f"Training UNet with {backbone} backbone")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_model(backbone=backbone, num_classes=3, pretrained=True)
    print(f"Model created with {backbone} backbone")

    # Check if we have enough GPU memory
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_memory_gb = free_memory / 1024**3
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")

        # If less than 2GB available, reduce batch size
        if free_memory_gb < 2.0 and batch_size > 8:
            batch_size = 8
            print(f"Limited GPU memory. Reducing batch size to {batch_size}")

            # Recreate data loaders with smaller batch size
            if hasattr(train_loader, 'batch_size') and train_loader.batch_size > batch_size:
                from torch.utils.data import DataLoader
                train_dataset = train_loader.dataset
                val_dataset = val_loader.dataset

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device
    )

    print("Model training completed!")

    return model, history

if __name__ == "__main__":
    # Make sure the data loaders are available from the preprocessing step
    # Either import them or use the global variables

    try:
        # Get the data loaders from global variables
        # Note: train_loader and val_loader should be defined previously
        import sys

        # Try to import from the previous script
        try:
            from data_preprocessing_fixed import create_data_loaders, folded_df
            print("Successfully imported preprocessing modules")

            # If loaders don't exist, create them
            if 'train_loader' not in globals() or 'val_loader' not in globals() or globals()['train_loader'] is None:
                print("Creating data loaders")
                train_loader, val_loader = create_data_loaders(folded_df, fold=0, batch_size=8)
        except ImportError:
            print("Could not import from data_preprocessing_fixed")
            # Check if they're already in globals (run in the same notebook)
            if 'train_loader' not in globals() or 'val_loader' not in globals():
                raise NameError("train_loader and val_loader are not defined")

        # Set training parameters - reduced for Kaggle notebooks
        backbone = 'resnet34'  # Options: resnet18, resnet34, resnet50
        epochs = 20
        lr = 3e-4
        batch_size = 8

        # Train the model
        model, history = main_train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            backbone=backbone,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )

        # Save the trained model
        torch.save(model.state_dict(), 'gi_tract_segmentation_model.pth')
        print("Model saved to gi_tract_segmentation_model.pth")

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    except NameError as e:
        print(f"Error: {e}")
        print("Make sure to run the data preprocessing script first to create the data loaders.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
