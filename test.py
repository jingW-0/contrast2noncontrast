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
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
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


def prepare_batch(image, ij_patch_indices):
    image_batches = []
    for batch in ij_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
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
        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
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
        print(f"recon shape: {recon.shape}")
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
    test_num = 5
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
            # if opt.direction == "AtoB":
            #     save_result(pathA, save_path, netG, opt)
            # else:
            #     save_result(pathB, save_path, netG, opt)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()

    if opt.model == 'cycle_gan':
        if opt.direction == "AtoB":
            netG = model.netG_A
        else:
            netG = model.netG_B
    else:
        netG = model.netG

    if opt.direction == "AtoB":
        result_dir = "D:\\contrast\\project2\\result2\\" + opt.name + "\\AtoB"
    else:
        result_dir = "D:\\contrast\\project2\\result2\\" + opt.name + "\\BtoA"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # if opt.direction == "AtoB":
    #     result_dir = "./slice_result/" + opt.name + "/AtoB"
    # else:
    #     result_dir = "./slice_result/" + opt.name + "/BtoA"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # input_dir = "D:/contrast/project/pe_contrast_transfer/iso/test/axial/A"
    # input_dir = "D:/contrast/project/pe_contrast_transfer/slices/axial/testA"
    # folder_test(opt, input_dir, result_dir, netG)
    series_test(opt, result_dir, netG)

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    # model = create_model(opt)      # create a model given opt.model and other options
    # # create a webpage for viewing the results
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    #
    # for i, data in enumerate(dataset):
    #     if i == 0:
    #         model.data_dependent_initialize(data)
    #         model.setup(opt)               # regular setup: load and print networks; create schedulers
    #         model.parallelize()
    #         if opt.eval:
    #             model.eval()
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, width=opt.display_winsize)
    # webpage.save()  # save the HTML
