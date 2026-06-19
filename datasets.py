import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
import scipy
import pandas as pd

def load_image(path):
    if path.endswith('.npy'):
        return np.load(path).astype('float32')
    elif path.endswith('.nii'):
        image = sitk.ReadImage(path)
        # print(image.GetSize())
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        # print("1", image_array.shape)
        # image_array = image_array.squeeze() #x, y, z
        image_array = image_array.transpose(2, 1, 0).squeeze()  # x, y, z
        # print("2", image_array.shape)
        return image_array.astype('float32')
    elif path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return np.array(img.convert('RGB'))


class ImageDataset(Dataset):
    def __init__(self, root, transforms_1=None, unaligned=True):
        self.transform1 = transforms.Compose(transforms_1)

        self.files_A = glob.glob("%s/A/*" % root)
        self.files_B = glob.glob("%s/B/*" % root)

        self.unaligned = unaligned

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            path_B = self.files_B[index % len(self.files_B)]

        item_A = self.transform1(load_image(path_A))
        item_B = self.transform1(load_image(path_B))

        # print(f"itemA: {item_A.shape}")
        return {'A': item_A, 'B': item_B, "A_paths": path_A, "B_paths": path_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDatasetFromFile(Dataset):
    def __init__(self, pathA, pathB, transforms_1=None, unaligned=True):
        self.transform1 = transforms.Compose(transforms_1)

        print(f"Reading dataset from {pathA} and {pathB} ...")
        input_A = pd.read_excel(pathA)
        input_B = pd.read_excel(pathB)

        self.files_A = input_A['files']
        self.files_B = input_B['files']

        self.unaligned = unaligned

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            path_B = self.files_B[index % len(self.files_B)]

        item_A = self.transform1(load_image(path_A))
        item_B = self.transform1(load_image(path_B))

        return {'A': item_A, 'B': item_B, "A_paths": path_A, "B_paths": path_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
