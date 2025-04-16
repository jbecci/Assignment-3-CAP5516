import os
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import KFold
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

from dataset import NucleiSegmentationDataset
from metrics import get_fast_aji, get_pq
from mobile_sam import sam_model_registry
from lora_utils import apply_lora_to_vit

def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1) #apply softmax for class probabilities
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = torch.sum(pred * target_one_hot)
    union = torch.sum(pred + target_one_hot)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def save_mask_prediction(pred_mask, image_id, fold_index, output_dir="predictions"):
    """
    Saves prediction mask (grayscale)
    """
    fold_dir = os.path.join(output_dir, f"fold_{fold_index}")
    os.makedirs(fold_dir, exist_ok=True)
    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8))
    save_path = os.path.join(fold_dir, f"pred_{image_id}.png")
    mask_image.save(save_path)

def save_visual_comparison(image_tensor, pred_mask, gt_mask, image_id, fold_index, output_dir="visuals"):
    """
    Save comparison of image (preprocessed), ground truth, and predicted mask
    """
    import matplotlib.pyplot as plt
    folder = os.path.join(output_dir, f"fold_{fold_index}")
    os.makedirs(folder, exist_ok=True)
    image_np = to_pil_image(image_tensor.cpu()).convert("RGB") #tensors to images
    pred_mask_np = (pred_mask.cpu().numpy() * 255).astype(np.uint8)
    gt_mask_np = (gt_mask.cpu().numpy() * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4)) #plot 3 side by side
    axes[0].imshow(image_np) #preprocessd image from dataset.py
    axes[0].set_title("Input Image") 
    axes[1].imshow(gt_mask_np, cmap="gray") #corresponding OG segmentation mask
    axes[1].set_title("Ground Truth") 
    axes[2].imshow(pred_mask_np, cmap="gray") #predicted mask
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(folder, f"vis_{image_id}.png")
    plt.savefig(save_path)
    plt.close()

#training loop
def train_one_fold(model, seg_head, train_loader, val_loader, device, fold_index):
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-3)  # Only train seg_head
    for epoch in range(5):  # Reduced to 5 epochs for speed
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"[Fold {fold_index}] Epoch {epoch+1}/5"):
            images = images.to(device)
            masks = masks.to(device)
            features = model.image_encoder(images) #f.maps from LoRA encoder
            #resize to match mask sizes
            output = seg_head(features)
            output = F.interpolate(output, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = dice_loss(output, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"[Fold {fold_index}] Epoch {epoch+1} - Dice Loss: {avg_loss:.4f}")
    #save weights after fold
    torch.save(seg_head.state_dict(), f"seg_head_fold_{fold_index}.pt")

#evaluation mode
def evaluate_model(model, seg_head, val_loader, device, fold_index, save_limit=5):
    model.eval()
    saved = 0
    for i, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            features = model.image_encoder(images)
            output = seg_head(features)
            output = F.interpolate(output, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred_classes = torch.argmax(output, dim=1)
        for b in range(pred_classes.shape[0]):
            if saved >= save_limit:
                return
            
            pred_mask = pred_classes[b].cpu().numpy()

            image_id = f"val_batch{i}_sample{b}"
            save_mask_prediction(pred_mask, image_id, fold_index)
            save_visual_comparison(images[b], torch.tensor(pred_mask), masks[b], image_id, fold_index)
            #apply metrics
            aji = get_fast_aji(masks[b].cpu().numpy(), pred_mask)
            pq = get_pq(masks[b].cpu().numpy(), pred_mask)
            dice = dice_loss(output, masks).item()
            csv_path = os.path.join("metrics", f"fold_{fold_index}_metrics.csv") #save in csv file
            os.makedirs("metrics", exist_ok=True)
            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([fold_index, 5, b, round(dice, 4), round(aji, 4), round(pq, 4)])
            print(f"[Fold {fold_index}] Eval Sample {b} - Dice: {dice:.4f}, AJI: {aji:.4f}, PQ: {pq:.4f}")
            saved += 1

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu" #sned to gpu
    print(f"Using device: {device}")
    image_dir = "/path/to/data/images" #replace with path to images folder
    mask_dir = "/path/to/masks" #replace with path to masks folder
    dataset = NucleiSegmentationDataset(image_dir, mask_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    checkpoint_path = "/path/to/mobile_sam.pt" #replace with path to downloaded mobileSAM model
    model_type = "vit_t"

    for fold_index, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        #create dataloaders
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device) #load SAM and freeze
        model.eval() #set to evaluation mode
        for param in model.parameters():
            param.requires_grad = False
        apply_lora_to_vit(model.image_encoder, target_keywords=["qkv"]) #inject LoRA layers, find keywords

        with torch.no_grad():
            sample_input = next(iter(train_loader))[0].to(device)
            features = model.image_encoder(sample_input)

        # multi-layer convolutional segmentation head
        seg_head = nn.Sequential(
            nn.Conv2d(features.shape[1], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        ).to(device)

        train_one_fold(model, seg_head, train_loader, val_loader, device, fold_index)
        evaluate_model(model, seg_head, val_loader, device, fold_index)

if __name__ == "__main__":
    main()