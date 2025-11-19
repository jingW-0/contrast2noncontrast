from models.cut_model import CUTModel
from models.cut3_model import CUT3Model
from options.test_options import TestOptions
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch
from models.cut_model import CUTModel
from torch.autograd import Variable
import util.util as util
import os
import numpy as np
from datasets import load_image
import math
import cv2


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

    for i in range(inum): # width
        for j in range(jnum): # height
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
        # print(batch.shape)

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


def inference_onnx(session, image, patch_size_x, patch_size_y, stride, batch_size=1):
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
        # print(batch.shape)

        outputs = session.run(None, {'input': batch[np.newaxis, :, :, :]})
        # print(type(outputs), len(outputs))
        # print(type(outputs[0]), outputs[0].shape)

        pred = outputs[0].squeeze()

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


def single_test(model, opt, file, save_path):
    file_name = os.path.basename(file)
    print(f"file name: {file_name}")
    img = load_image(file)
    # print(f"img shape: {img.shape}")
    pred = inference(model, img, opt.patch_size, opt.patch_size, opt.stride)
    pred = (pred + 1) * 0.5 * 255
    pred = np.transpose(pred, (1, 0))
    #
    print(f"Save to {save_path}...")
    cv2.imwrite(save_path, pred)

def single_test_onnx(session, opt, file, save_path):
    file_name = os.path.basename(file)
    print(f"file name: {file_name}")
    img = load_image(file)

    # print(f"img shape: {img.shape}")
    pred = inference_onnx(session, img, opt.patch_size, opt.patch_size, opt.stride)
    pred = (pred + 1) * 0.5 * 255
    pred = np.transpose(pred, (1, 0))

    save_path = save_path
    print(f"Save to {save_path}...")
    cv2.imwrite(save_path, pred)


opt = TestOptions().parse()  # get test options
model = CUT3Model(opt=opt)
model.eval()

load_path = "D:\\contrast\\onnx\\5_net_G.pth"
# load_path = "D:\\contrast\\project2\\checkpoints_mix\\CUT\\10_net_G.pth"
netG = model.netG
state_dict = torch.load(load_path, map_location='cuda')
netG.load_state_dict(state_dict)

# A_path = "D:\\contrast\\project\\pe_contrast_transfer\\iso\\axial\\A\\324480SEVINCZULFIYE_2020_10_18_4_1.2.392.200036.91_img_axial_200.npy"
#
A_path = "D:\\contrast\\project\\pe_contrast_transfer\\iso\\axial_nii\\A\\324480SEVINCZULFIYE_2020_10_18_4_1.2.392.200036.91_img_axial_200.nii"

save_to = "onnxtest"
os.makedirs(save_to, exist_ok=True)
single_test(netG, opt, A_path, os.path.join(save_to, "pth_result.png"))

data_in = torch.rand([1, 1, 256, 256]).cuda()
torch.onnx.export(netG,
                  data_in,
                  "pth2onnx_translation.onnx",
                  verbose=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

import onnx

modelG = onnx.load("pth2onnx_translation.onnx")
onnx.checker.check_model(modelG)
print(onnx.helper.printable_graph(modelG.graph))

import onnxruntime as ort

# ort_session = ort.InferenceSession("pth2onnx_test.onnx")
ort_session = ort.InferenceSession("pth2onnx_translation.onnx")
single_test_onnx(ort_session, opt, A_path, os.path.join(save_to, "onnx_result.png"))
