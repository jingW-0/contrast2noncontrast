"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

from fontTools.misc.cython import returns

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import save_images
from util import html
import util.util as util

from torch.autograd import Variable
import datetime
import torch
import math
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

import pandas as pd
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import RandomAffine, ToPILImage, RandomCrop, ToTensor, Resize
from datasets0 import ImageDataset
from torch.utils.data import DataLoader
from skimage.transform import resize
import scipy

def save_result(path, save_path, model, opt):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, (2, 1, 0))  # .astype(float)
    print(image_array.shape)
    image_array = np.clip(image_array, a_min=-1000, a_max=2000)
    # print(f"image2 min, max: {np.min(image)}, {np.max(image)}")
    image_array = (image_array + 1000) / (2000 + 1000) * 2 - 1
    # print(image_array.dtype)
    image_array = image_array.astype('float32')
    # print(image_array.dtype)

    slices = []
    for i in range(image_array.shape[2]):
        slice = image_array[:, :, i]
        # print(slice.shape)

        slice = torch.from_numpy(slice[np.newaxis,np.newaxis, :, :])
        slice = Variable(slice.cuda())

        pred = model(slice)
        pred = pred.squeeze().data.cpu()
        slices.append(pred.numpy().squeeze())
    recon = np.stack(slices)
    recon = (recon + 1) * 0.5 * 3000 - 1000
    # print(recon.shape)
    recon = np.transpose(recon, (0, 2, 1))
    recon_image = sitk.GetImageFromArray(recon)
    sitk.WriteImage(recon_image, save_path)


def prepare_batch(image, ij_patch_indices):
    image_batches = []
    for batch in ij_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3]]
            image_batch.append(image_patch)
            # print(f"patch shape: {image_patch.shape}")

        image_batch = np.asarray(image_batch)
        # print(f"batch shape      : {image_batch.shape}")
        image_batches.append(image_batch)

    return image_batches


def inference(model, image, patch_size_x, patch_size_y, stride, batch_size=1):
    pad_x = patch_size_x - (patch_size_x - image.shape[0])
    pad_y = patch_size_x - (patch_size_y - image.shape[1])
    image_np = image
    label_np = np.zeros_like(image)
    weight_np = np.zeros_like(label_np)  # a weighting matrix will be used for averaging the overlapped region

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride))) + 1

    patch_total = 0
    ij_patch_indices = []
    ij_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            if patch_total % batch_size == 0:
                ij_patch_indicies_tmp = []

            istart = i * stride
            if istart + patch_size_x > image_np.shape[0]:  # for last patch
                istart = image_np.shape[0] - patch_size_x
            iend = istart + patch_size_x

            jstart = j * stride
            if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                jstart = image_np.shape[1] - patch_size_y
            jend = jstart + patch_size_y

            ij_patch_indicies_tmp.append([istart, iend, jstart, jend])

            if patch_total % batch_size == 0:
                ij_patch_indices.append(ij_patch_indicies_tmp)

            patch_total += 1

    batches = prepare_batch(image_np, ij_patch_indices)
    # print(f"len batches: {len(batches)}")
    for i in range(len(batches)):
        # for i in tqdm(range(len(batches))):
        batch = batches[i]

        print(f"batch shape: {batch.shape}")
        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        print(f"batch shape: {batch.shape}")
        batch = Variable(batch.cuda())


        pred = model(batch)
        pred = pred.squeeze().data.cpu().numpy()

        istart = ij_patch_indices[i][0][0]
        iend = ij_patch_indices[i][0][1]
        jstart = ij_patch_indices[i][0][2]
        jend = ij_patch_indices[i][0][3]

        label_np[istart:iend, jstart:jend] += pred[:, :]
        weight_np[istart:iend, jstart:jend] += 1.0

    # print("{}: Evaluation complete".format(datetime.datetime.now()))

    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y]
    return label_np


def save_result_patch(path, save_path, model, opt, direction='axial'):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, (2, 1, 0))
    print(image_array.shape)
    image_array = np.clip(image_array, a_min=-1024, a_max=2048)
    image_array = (image_array + 1024) / (2048 + 1024) * 2 - 1

    image_array = image_array.astype('float32')

    print("{}: Inference start".format(datetime.datetime.now()))

    slices = []
    if direction == 'axial':
        for i in tqdm(range(image_array.shape[2])):
            slice = image_array[:, :, i]  # x, y
            # print(f"slice shape {slice.shape}")
            pred = inference(model, slice, opt.patch_size, opt.patch_size, opt.stride)
            # print(f"pred size: {pred.shape}")
            slices.append(pred)
        recon = np.stack(slices)  # z, x, y
        # print(f"recon shape: {recon.shape}")
        recon = np.transpose(recon, (0, 2, 1))  # z, y, x
    elif direction == 'coronal':
        for i in tqdm(range(image_array.shape[1])):
            slice = image_array[:, i, :]  # x, z
            pred = inference(model, slice, opt.patch_size, opt.patch_size, opt.stride)
            # print(f"pred size: {pred.shape}")

            slices.append(pred)
        recon = np.stack(slices)  # y, x, z
        print(f"recon shape: {recon.shape}")
        recon = np.transpose(recon, (2, 0, 1))  # z, y, x
    elif direction == 'sagittal':
        for i in tqdm(range(image_array.shape[0])):
            slice = image_array[i, :, :]  # y,z
            pred = inference(model, slice, opt.patch_size, opt.patch_size, opt.stride)
            # print(f"pred size: {pred.shape}")

            slices.append(pred)
        recon = np.stack(slices)  # x, y, z
        print(f"recon shape: {recon.shape}")
        recon = np.transpose(recon, (2, 1, 0))  # z, y, x

    print("{}: Inference complete".format(datetime.datetime.now()))

    recon = (recon + 1) * 0.5 * 3072 - 1024
    #

    recon_image = sitk.GetImageFromArray(recon)
    sitk.WriteImage(recon_image, save_path)


def series_test(opt, result, netG):
    # test_num = 5
    df = pd.read_csv("D:/contrast/project/pe_contrast_transfer/paired_iso.csv")
    for i in range(len(df)):
        # for i in range(test_num):
        pair = df.iloc[i]
        # print(pair)
        pathA = pair["contrast"]
        pathB = pair["noncontrast"]

        file = pathA.split("\\")[-1]
        print(file)
        save_path = os.path.join(result, file)

        if opt.input_mode == 'patch':
            # print("patch inference")
            if opt.direction == "AtoB":
                save_result_patch(pathA, save_path, netG, opt, opt.inference_view)
            else:
                save_result_patch(pathB, save_path, netG, opt, opt.inference_view)
        else:
            print("full image prediction")
            if opt.direction == "AtoB":
                save_result(pathA, save_path, netG, opt)
            else:
                save_result(pathB, save_path, netG, opt)


def series_test_heart(opt, result, netG, header="file_name", path=None):
    # df = pd.read_csv("D:\\IMAGES\\Cardiac_data_for_image_translation\\contrast_test.csv")
    if path is None:
        df = pd.read_csv("D:\\IMAGES\\Cardiac_data_for_image_translation\\seg_train.csv")
    else:
        df = pd.read_csv(path)

    for file in df[header].values:
        # for i in range(test_num):
        # if file == "1225_43446_HB047807_2018-01-01_1.2.840.113619.2.55.3.2474655163.562.1544144623.682_4.nii":
        pathA = os.path.join("D:\\IMAGES\\Cardiac_data_for_image_translation\\result\\img-003-contrast", file)
        pathB = ""

        # file = pathA.split("\\")[-1]
        print(file)
        save_path = os.path.join(result, file)

        if opt.input_mode == 'patch':
            # print("patch inference")
            if opt.direction == "AtoB":
                save_result_patch(pathA, save_path, netG, opt, opt.inference_view)
            else:
                save_result_patch(pathB, save_path, netG, opt, opt.inference_view)
        else:
            print("full image prediction")
            if opt.direction == "AtoB":
                save_result(pathA, save_path, netG, opt)
            else:
                save_result(pathB, save_path, netG, opt)

def center_crop(img, size=256):
    # print(f"input shape: {img.shape}")
    center_x, center_y = img.shape[0]//2, img.shape[1]//2

    left = center_x - size//2
    top = center_y - size//2

    cropped = img[left:left+size, top:top+size]
    return cropped

def softmax(x):
    x_exp = np.exp(x)
    # print(f"sum {np.sum(x_exp)}")
    return x_exp / np.sum(x_exp)

# Define the similarity map computation function
def compute_similarity_map(features, target_position):
    B, C, W, H = features.size()
    x, y = target_position

    # Extract feature vector at the target position
    target_feature = features[:, :, x, y]  # Shape: (B, C)

    # Reshape features for pairwise similarity computation
    features_reshaped = features.view(B, C, -1)  # Shape: (B, C, H*W)
    print(f"f reshaped : {features_reshaped.shape}")
    target_feature = target_feature.unsqueeze(-1)  # Shape: (B, C, 1)
    print(f"tf reshaped : {target_feature.shape}")

    # Compute cosine similarity between target_feature and all other patches
    similarity_map = F.cosine_similarity(features_reshaped, target_feature, dim=1)  # Shape: (B, H*W)

    # Reshape similarity map back to (B, H, W)
    similarity_map = similarity_map.view(B, W, H)

    return similarity_map





if __name__ == '__main__':
    # opt = TestOptions().parse()  # get test options

    # # hard-code some parameters for test
    # opt.num_threads = 0  # test code only supports num_threads = 1
    # opt.batch_size = 1  # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    # opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt = TrainOptions().parse()  # get training options
    transforms_1 = [ToPILImage(),
                    RandomCrop(size=opt.patch_size, pad_if_needed=True, fill=-1),
                    ToTensor()]
    dataset = DataLoader(
        ImageDataset(opt.dataroot, transforms_1=transforms_1, unaligned=True), batch_size=opt.batch_size,
        shuffle=True, num_workers=opt.num_threads)

    model = create_model(opt)  # create a model given opt.model and other options

    # for i, data in enumerate(dataset):
    #     model.data_dependent_initialize(data)
    #     model.setup(opt)  # regular setup: load and print networks; create schedulers
    #     model.parallelize()
    #     break
    # # model.setup(opt)  # regular setup: load and print networks; create schedulers


    for i, data in enumerate(dataset):
        if i == 0:

            model.data_dependent_initialize(data)
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.parallelize()
            # if opt.eval:
            model.eval()

            break

    # # if opt.eval:
    # model.eval()
    #
    # if opt.model == 'cycle_gan':
    #     if opt.direction == "AtoB":
    #         netG = model.netG_A
    #     else:
    #         netG = model.netG_B
    # else:
    #     netG = model.netG
    netF = model.netF
    # print(netF)
    netG = model.netG
    # if opt.direction == "AtoB":
    #     result_dir = "D:\\contrast\\project2\\result_sim\\" + opt.name + "\\AtoB"
    # else:
    #     result_dir = "D:\\contrast\\project2\\result_sim\\" + opt.name + "\\BtoA"
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    #
    input_dir = "D:/contrast/project/pe_contrast_transfer/iso/test/axial/A"
    #
    input_folder = input_dir
    # file_list = os.listdir(input_folder)
    #
    # for file in file_list:
    file = "110968OYMANHALDUN_2019_03_30_2_1.2.392.200036.9116_img_axial_261.npy"
    name = file.split(".npy")[0]
    print(f"file name: {name}")
    img = np.load(os.path.join(input_folder, file)).astype('float32')
    print(f"img shape: {img.shape}")

    img_cropped = center_crop(img, 256)

    batch = torch.from_numpy(img_cropped[np.newaxis, np.newaxis, :, :])
    print(f"batch shape: {batch.shape}")
    batch = Variable(batch.cuda())

    # pred = netG(batch)
    # pred = pred.squeeze().data.cpu().numpy()
    # print(f"pred shape: {pred.shape}")
    # # pred = inference(netG, img, opt.patch_size, opt.patch_size, opt.stride)
    # pred = (pred + 1) * 0.5 * 255
    # pred = np.transpose(pred, (1, 0))
    # cv2.imwrite(os.path.join(result_dir, name + '.png'), pred)

    # print(netG)
    # layers = [1, 4, 8, 12, 16]

    layers = [1, 4, 8, 12, 16]
    feat_layers = []
    ref_size = 256
    query_position = (135, 38)
    for layer in layers:
        with torch.no_grad():
            feats = netG(batch, [layer], encode_only=True)

        feat_layers.append(feats[0])
        # print(len(feats))
        # # for i, f in enumerate(feats):
        # #     print(f"i {i}, fshape {f.shape}")
        print(f"fshape {feats[0].shape}")

        # target_position = (feats[0].shape[2]//2, feats[0].shape[3]//2)
        fw = feats[0].shape[2]
        fh = feats[0].shape[3]

        target_position = (int(query_position[0]/ref_size * fw), int(query_position[1]/ref_size * fh))
        feature_similarity = compute_similarity_map(feats[0], target_position)
        print(f"sim {feature_similarity.shape}, min, {feature_similarity.min()}, max, {feature_similarity.max()}")
        similarity_map_np = feature_similarity.cpu().detach().numpy().squeeze()
        # print()
        # similarity_map_exp = np.exp(similarity_map_np)
        # print(f" min, {similarity_map_exp.min()}, max, {similarity_map_exp.max()}")

        # similarity_map_softmax = softmax(similarity_map_np)
        # similarity_map_softmax = similarity_map_softmax / similarity_map_softmax.max()
        # print(f"sim softmax min {similarity_map_softmax.min()}, max {similarity_map_softmax.max()}")
        # similarity_map_np = (similarity_map_np - similarity_map_np.min()) / (
        #             similarity_map_np.max() - similarity_map_np.min())  # Normalize to [0, 1]

        # print(f"sim np: {similarity_map_np.shape}")
        similarity_map_np = scipy.ndimage.interpolation.zoom(similarity_map_np, [img_cropped.shape[0] / similarity_map_np.shape[0],
                                                          img_cropped.shape[1] / similarity_map_np.shape[1]], order=1)
        # print(f"sim np 2: {similarity_map_np.shape}")
        # similarity_map_softmax = scipy.ndimage.interpolation.zoom(similarity_map_softmax,
        #                                                      [img_cropped.shape[0] / similarity_map_softmax.shape[0],
        #                                                       img_cropped.shape[1] / similarity_map_softmax.shape[1]],
        #                                                      order=1)
        # similarity_map_exp = scipy.ndimage.interpolation.zoom(similarity_map_exp,
        #                                                           [img_cropped.shape[0] / similarity_map_exp.shape[
        #                                                               0],
        #                                                            img_cropped.shape[1] / similarity_map_exp.shape[
        #                                                                1]],
        #                                                           order=1)
        # Plot the mask
        plt.figure(figsize=(8, 8))
        # plt.imshow(np.transpose(img_cropped, (1, 0)))
        # plt.imshow(np.transpose(similarity_map_np, (1, 0)), cmap="viridis", alpha=0.5)
        # plt.imshow(np.transpose(similarity_map_softmax, (1, 0)), cmap="viridis", alpha=0.5)
        # plt.imshow(np.transpose(similarity_map_softmax, (1, 0)), cmap="viridis")
        # plt.imshow(np.transpose(similarity_map_exp, (1, 0)), cmap="viridis")
        plt.imshow(np.transpose(similarity_map_np, (1, 0)), cmap="viridis")
        # plt.colorbar(label="Exp(Cosine Similarity)")
        plt.colorbar(label="Cosine Similarity")

        # # Mark the target position
        # x, y = target_position
        x, y = query_position

        plt.scatter(x, y, color="red", marker="x", s=100, label="Target Position")
        plt.legend(loc="upper right")

        plt.title(f'Similarity map, layer {layer}', fontsize=16, fontweight='bold')

        # plt.title(f'Similarity map, layer {layer}, position {target_position}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    B0, H0, W0 = feat_layers[0].shape[0], feat_layers[0].shape[2], feat_layers[0].shape[3]
    aligned_feats_list = []

    for f in feat_layers:
        print(f.shape)
        B, H, W = f.shape[0], f.shape[2], f.shape[3]
        scale_factor = W0 // W
        resized_feature_map = f
        if scale_factor > 1:
            resized_feature_map = F.interpolate(f, scale_factor=scale_factor, mode='bilinear',
                                                align_corners=False)
        aligned_feats_list.append(resized_feature_map)
        # print(f"resized: {resized_feature_map.shape}")

    aligned_feats = torch.cat(aligned_feats_list, dim=1)
    print(f"aligned {aligned_feats.shape}")
    feature_similarity_gl = compute_similarity_map(aligned_feats, query_position)
    # print(f"sim {feature_similarity.shape}")
    similarity_map_gl = feature_similarity_gl.cpu().detach().numpy().squeeze()
    print(f"gl min {similarity_map_gl.min()}, max {similarity_map_gl.max()}")
    # similarity_map_softmax_gl = softmax(similarity_map_np)
    # similarity_map_softmax_gl_nm = similarity_map_softmax_gl / similarity_map_softmax_gl.max()

    # similarity_map_exp_gl = np.exp(similarity_map_gl)
    # similarity_map_exp_gl_nm = similarity_map_exp_gl / similarity_map_exp_gl.max()
    # similarity_map_np_nm = (similarity_map_np - similarity_map_np.min()) / (
    #         similarity_map_np.max() - similarity_map_np.min())  # Normalize to [0, 1]
    # print(f"similarity map min, max: {similarity_map_np.min()}, {similarity_map_np.max()}")
    plt.figure(figsize=(8, 8))
    # plt.imshow(np.transpose(img_cropped, (1, 0)))
    # plt.imshow(np.transpose(similarity_map_softmax_gl_nm, (1, 0)), cmap="viridis", alpha=0.5)
    # plt.imshow(np.transpose(similarity_map_softmax_gl_nm, (1, 0)), cmap="viridis")
    # plt.imshow(np.transpose(similarity_map_exp_gl, (1, 0)), cmap="viridis")
    plt.imshow(np.transpose(similarity_map_gl, (1, 0)), cmap="viridis")

    x, y = query_position
    plt.scatter(x, y, color="red", marker="x", s=100, label="Target Position")
    plt.legend(loc="upper right")
    plt.colorbar(label="Cosine Similarity")
    plt.title(f'Feature Pyramid Similarity', fontsize=16, fontweight='bold')

    # plt.title(f'Similarity, global, position {query_position}', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    for i, af in enumerate(aligned_feats_list):
        feature_similarity_aligned = compute_similarity_map(af, query_position)
        print(f"sim {feature_similarity_aligned.shape}")
        similarity_map_np_aligned = feature_similarity_aligned.cpu().detach().numpy().squeeze()
        # similarity_map_np_aligned_softmax = softmax(similarity_map_np_aligned)

        similarity_map_np_aligned_div = (similarity_map_np_aligned + 1 + 1e-7) / (similarity_map_gl + 1 + 1e-7)
        # similarity_map_np_aligned_sub = similarity_map_np_aligned - similarity_map_gl
        # print(f"sub min : {similarity_map_np_aligned_sub.min()}, max: {similarity_map_np_aligned_sub.max()}")
        # similarity_map_np_aligned_exp = np.exp(similarity_map_np_aligned)
        # similarity_map_np_aligned_softmax = simi
        # print(f"i: {i}")
        # print(similarity_map_np_aligned)
        # print(".......................")
        # print(similarity_map_np)
        # print("=========================")
        # s_reweight = similarity_map_np_aligned_softmax / (similarity_map_softmax_gl + 1e-7)
        # s_reweight = similarity_map_np_aligned_exp / (similarity_map_exp_gl + 1e-7)
        s_reweight = similarity_map_np_aligned_div
        # s_reweight = similarity_map_np_aligned_sub

        print(f"s min {s_reweight.min()}, max {s_reweight.max()}")

        # similarity_map_np_aligned_nm = (s_reweight - s_reweight.min()) / (
        #         s_reweight.max() - s_reweight.min())  # Norm
        # s_reweight_nm = s_reweight / s_reweight.max()
        # print(f"s min {s_reweight.min()}, max {s_reweight.max()}")
        s_reweight = softmax(s_reweight)
        print(f"s min {s_reweight.min()}, max {s_reweight.max()}")
        # s_reweight = s_reweight / s_reweight.max()
        #
        # print(f"s min {s_reweight.min()}, max {s_reweight.max()}")
        # print(f"s nm min {s_reweight_nm.min()}, max {s_reweight_nm.max()}")

        # Plot the mask
        plt.figure(figsize=(8, 8))
        # plt.imshow(np.transpose(img_cropped, (1, 0)))
        # plt.imshow(np.transpose(s_reweight, (1, 0)), cmap="viridis", alpha=0.5)
        plt.imshow(np.transpose(s_reweight, (1, 0)), cmap="viridis")

        plt.colorbar(label="Softmax")
        # # Mark the target position
        # x, y = target_position
        x, y = query_position

        plt.scatter(x, y, color="red", marker="x", s=100, label="Target Position")
        plt.legend(loc="upper right")
        plt.title(f'Normalized Similarity map, layer {layers[i]}', fontsize=16, fontweight='bold')

        # plt.title(f'Normalized Similarity map, layer {layers[i]}, position {query_position}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()