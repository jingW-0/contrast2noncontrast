import glob
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
import scipy


def load_image(path):
    if path.endswith('.npy'):
        return np.load(path).astype('float32')
    elif path.endswith('.nii'):
        image = sitk.ReadImage(path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = image_array.transpose(2, 1, 0).squeeze()
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
        img_A = load_image(path_A)

        best_path_B = ""
        if self.unaligned:
            img_A_ = scipy.ndimage.interpolation.zoom(img_A, [64 / img_A.shape[0],
                                                              64 / img_A.shape[1]], order=1)
            nsampling = 0
            prev_ssim = 0
            while 1:
                path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
                img_B = load_image(path_B)
                img_B_ = scipy.ndimage.interpolation.zoom(img_B, [64 / img_B.shape[0],
                                                                  64 / img_B.shape[1]], order=1)
                nsampling += 1
                s = ssim(img_A_, img_B_, channel_axis=-1, data_range=2)
                if s > prev_ssim:
                    prev_ssim = s
                    best_path_B = path_B

                if s > 0.5 or nsampling >= 20:
                    break
        else:
            best_path_B = self.files_B[index % len(self.files_B)]

        item_A = self.transform1(img_A)
        item_B = self.transform1(load_image(best_path_B))

        return {'A': item_A, 'B': item_B, "A_paths": path_A, "B_paths": best_path_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDatasetFromFile(Dataset):
    def __init__(self, pathA, pathB, transforms_1=None, unaligned=True, max_itr=1, with_metric=False, metric_tol=np.inf, ssim_th=np.inf):
        self.transform1 = transforms.Compose(transforms_1)
        self.with_metric = with_metric
        self.metric_tol = metric_tol
        self.max_itr = max_itr
        self.ssim_th = ssim_th

        input_A = pd.read_excel(pathA)
        input_B = pd.read_excel(pathB)
        self.files_A = input_A['files']
        self.files_B = input_B['files']

        if with_metric:
            self.metrics_A = input_A['metric']
            self.metrics_B = input_B['metric']

        self.unaligned = unaligned

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        if self.with_metric:
            metric_A = self.metrics_A[index % len(self.files_A)]

        img_A = load_image(path_A)

        best_path_B = ""
        if self.unaligned:
            img_A_ = scipy.ndimage.interpolation.zoom(img_A, [64 / img_A.shape[0],
                                                              64 / img_A.shape[1]], order=1)
            nsampling = 0
            prev_ssim = 0
            while nsampling < self.max_itr:
                index_B = random.randint(0, len(self.files_B) - 1)
                nsampling += 1
                path_B = self.files_B[index_B]

                if self.with_metric:
                    metric_B = self.metrics_B[index_B]
                    if np.abs(metric_A - metric_B) > self.metric_tol:
                        continue

                img_B = load_image(path_B)
                img_B_ = scipy.ndimage.interpolation.zoom(img_B, [64 / img_B.shape[0],
                                                                  64 / img_B.shape[1]], order=1)

                s = ssim(img_A_, img_B_, channel_axis=-1, data_range=2)
                if s > prev_ssim:
                    prev_ssim = s
                    best_path_B = path_B

                if s > self.ssim_th:
                    break
        else:
            path_B = self.files_B[index % len(self.files_B)]

        item_A = self.transform1(img_A)
        if best_path_B == "":
            item_B = self.transform1(load_image(path_B))
        else:
            item_B = self.transform1(load_image(best_path_B))

        return {'A': item_A, 'B': item_B, "A_paths": path_A, "B_paths": best_path_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
