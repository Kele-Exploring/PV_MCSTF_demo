"""
Alignment, Reading and Preprocessing Operations for Time Series and Image Data
"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

def load_and_preprocess_time_series(csv_file, group_size=72):
    df = pd.read_csv(csv_file)

    num_samples = df.shape[0] // group_size
    data = df.iloc[:num_samples * group_size]

    power_values = data['instantaneous_output_power'].values.reshape(-1, group_size)
    radiation_values = data['instantaneous_global_radiation'].values.reshape(-1, group_size)
    features = np.stack((power_values, radiation_values), axis=2)

    labels = data['label'].values[::group_size]
    return features, labels

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, csv_file, group_size=72, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.group_size = group_size

        self.time_series_data, self.time_series_labels = load_and_preprocess_time_series(csv_file, group_size)

        self.image_paths = []
        self.image_labels = []
        self.image_groups = []
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                group_part = img_name.split('group_')[-1].split('_')[0]
                group_num = int(group_part)
                label = int(img_name.split('_')[-1].split('.')[0])
                self.image_paths.append(os.path.join(image_dir, img_name))
                self.image_labels.append(label)
                self.image_groups.append(group_num)

        assert len(self.image_groups) == len(self.time_series_labels), \
            "The length of the image data does not match that of the time-series data."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        group_num = self.image_groups[idx]
        time_series = self.time_series_data[group_num-1]
        label = self.time_series_labels[group_num-1]

        return image, torch.tensor(time_series, dtype=torch.float32), label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    image_dir = 'dataset/MCSTF_RGB'
    csv_file = 'dataset/dataset.csv'

    dataset = MultimodalDataset(image_dir, csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for images, time_series, labels in dataloader:
        print(f"Image batch shape: {images.shape}")
        print(f"Time series batch shape: {time_series.shape}")
        print(f"Label batch shape: {labels.shape}")
        break
