import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class NucleiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir): 
        self.image_dir = image_dir #directory to input images as .png
        self.mask_dir = mask_dir #directory to segmentation masks as .tif
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        #resize to 1024x1024, convert to tensor and normalize 
        self.image_transform = T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.Resize((1024, 1024)) #resize mask to match image

    def __len__(self):
        return len(self.image_names) #gives number of image-mask pairs

    def __getitem__(self, idx):
        #should get corresponding file names
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.png', '.tif')

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        #load image and mask with PIL
        image = Image.open(image_path).convert("RGB") #RGB format
        mask = Image.open(mask_path) #grayscale

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
