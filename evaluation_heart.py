import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
def from_npy_to_png(input_folder, output_folder, model_list, epoch_list):
    for model, epoch in zip(model_list, epoch_list):
        file_dir = os.path.join(input_folder, model, "test_"+str(epoch))
        save_dir = os.path.join(output_folder, model, "test_"+str(epoch))
        os.makedirs(save_dir, exist_ok=True)

        for file in os.listdir(file_dir):
            name = file.split(".npy")[0]
            path = os.path.join(file_dir, file)
            slice = np.load(path).astype('float32')
            # print(image_array.shape)
            slice = (slice + 1) * 0.5 * 255
            slice = np.transpose(slice, (1, 0))

            # slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(os.path.join(save_dir, name + '.png'), slice)


def compare_images(imageA, imageB):
    """Compute SSIM and PSNR between two images."""
    s = ssim(imageA, imageB, channel_axis=-1, data_range=2)
    # p = psnr(imageA, imageB)
    return s

def ssim_heart(model_list, epoch, save_to):
    folder1 = "D:\\IMAGES\\Cardiac_data_for_image_translation\\axial\\test\\A"
    root_folder = "D:\\contrast\\project2\\heart_results_npy"

    # model_list = ['test_10', 'test_8', 'test_5']
    # for i in range(len(model_list)):
    #     save_path = os.path.join(root_folder, model_list[i] + "_slice_evaluation.csv")
    dict_all = {}
    dict_all['model'] = model_list
    dict_all['epoch'] = [epoch for i in range(len(model_list))]
    # dict_all['SSIM'] = []
    # dict_all['PSNR'] = []
    s_avgs = []
    s_stds = []

    for model in model_list:
        save_path = os.path.join(root_folder, model + "_epoch_"+str(epoch)+"_slice_evaluation.csv")
        print(save_path)
        ss = []
        for filename in sorted(os.listdir(folder1)):
            img1 = np.load(os.path.join(folder1, filename)).astype('float64')
            img2 = np.load(os.path.join(root_folder, model,"test_"+str(epoch), filename)).astype('float64')
            s = compare_images(img1, img2)
            # print(f"{fname}: SSIM = {s:.4f}, PSNR = {p:.2f} dB")
            ss.append(s)

        s_avg = np.mean(ss)

        s_avgs.append(s_avg)
        s_stds.append(np.std(ss))

        dict = {'SSIM': ss}
        df = pd.DataFrame(dict)
        df.to_csv(save_path, index=False)
        print(f"mean SSIM = {s_avg:.4f}")
        # print(f"1 {s_avg}, 2 {p_avg}")

    dict_all['SSIM'] = s_avgs
    dict_all['SSIM_std'] = s_stds

    df_all = pd.DataFrame(dict_all)
    df_all.to_csv(os.path.join(root_folder, save_to), index=False)



def loadimage(path):
    image = sitk.ReadImage(path)
    # print(f"image getsize: {image.GetSize()}")
    image_array = sitk.GetArrayFromImage(image)
    # print(f"image_array.shape {image_array.shape}")
    image_array = np.transpose(image_array, (2, 1, 0))
    # print(f"image_array.shape tr {image_array.shape}")
    return image_array

def save_to_npy(root_folder, result_dir, model_list, epoch_list, min_val=-1024, max_val=2048):

    for model, epoch in zip(model_list, epoch_list):
        folder = os.path.join(root_folder, model, 'test_'+str(epoch), 'AtoB')
        file_list = os.listdir(folder)
        print(f"# file: {len(file_list)}")

        save_dir = os.path.join(result_dir, model, 'test_'+str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for file in file_list:
            image = loadimage(os.path.join(folder, file))
            name = file.split("\\")[-1]
            print(name)
            loc = name.find(".nii")
            name = name[:loc]
            # print(f"image1 min, max: {np.min(image)}, {np.max(image)}")
            image = np.clip(image, a_min=min_val, a_max=max_val)
            # print(f"image2 min, max: {np.min(image)}, {np.max(image)}")
            image = (image - min_val) / (max_val - min_val) * 2 - 1
            # print(f"image3 min, max: {np.min(image)}, {np.max(image)}")
            print(image.shape)

            for z in range(image.shape[2]):
                if image.shape[2] < 150:
                    slice = image[:, :, z]
                    save_file = os.path.join(save_dir, name + "_axial_" + str(z) + ".npy")
                    with open(save_file, 'wb') as f:
                        np.save(f, slice)
                else:
                    if z%4 == 0:
                        slice = image[:, :, z]
                        save_file = os.path.join(save_dir, name + "_axial_" + str(z) + ".npy")
                        with open(save_file, 'wb') as f:
                            np.save(f, slice)

            # for z in range(image.shape[2]):
            #     slice = image[:, :, z]
            #     # count = np.sum(slice > -1)
            #     # if count < 10:
            #     #     continue
            #
            #     save_file = os.path.join(save_dir, name + "_axial_" + str(z) + ".npy")
            #     with open(save_file, 'wb') as f:
            #         np.save(f, slice)
        # break

def ssim_boxplot():

    path = "D:\\contrast\\project2\\heart_results_npy\\SSIM_CMP.xlsx"

    df = pd.read_excel(path)
    # Example data for three groups
    group1 = df["CycleGAN"].values
    group2 = df["CUT(NCE)"].values
    group3 = df["CUT(DCE)"].values

    # Combine the data into a list
    data = [group1, group2, group3]

    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=["CycleGAN", "CUT(NCE)", "Ours"])
    # Create boxplot with customized style
    # plt.figure(figsize=(8, 6))
    # boxprops = dict(color='blue', linewidth=2)
    # flierprops = dict(marker='o', color='red', markersize=5)
    # medianprops = dict(color='green', linewidth=2)
    # whiskerprops = dict(color='orange', linewidth=2, linestyle='--')
    #
    # plt.boxplot(data,
    #             labels=['Group 1', 'Group 2', 'Group 3'],
    #             boxprops=boxprops,
    #             flierprops=flierprops,
    #             medianprops=medianprops,
    #             whiskerprops=whiskerprops)

    plt.title('Boxplot of SSIM', fontsize=18, fontweight='bold')
    plt.ylabel('Values', fontsize=16, fontweight='bold')
    # plt.xlabel('Groups', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1)
    # Change xtick and ytick font size
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def fid_curve():
    # Example data
    time_points = [0, 2, 4, 6, 8, 10]  # Time points from 1 to 10
    set_cycle = [23.09188878
,22.50844866,
16.85625187,
16.0916162,
14.26715104, 14.22351344
                 ]
    set_dce = [23.09188878
,17.4333495,
14.27775698,
14.11066414,
13.73835733,13.29297812
]
    set_nce = [23.09188878
,22.9863383,
20.13224811,
17.04422085,
16.98038394,
16.21246216
]

    # Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, set_cycle, label='CycleGAN', marker='o', linestyle='-.', linewidth=2)
    plt.plot(time_points, set_nce, label='CUT(NCE)', marker='s', linestyle='--', linewidth=2)
    plt.plot(time_points, set_dce, label='Ours', marker='^', linestyle='-', linewidth=2)

    # Add title and labels
    plt.title('FID Over training', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('FID', fontsize=16, fontweight='bold')

    # Customize ticks
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Add legend
    plt.legend(fontsize=14)

    # Add grid
    plt.grid(linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

def image_to_npy():
    # heart
    # model_list = ['cardiac_cycle_512', 'cardiac_cycle_512', 'cardiac_cycle_512', 'cardiac_cycle_512',
    #               'cardiac_dce_1024','cardiac_dce_1024','cardiac_dce_1024', 'cardiac_dce_1024']
    # epoch_list = [2, 4, 6, 8, 2, 4, 6, 8]
    model_list = ['cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024']
    epoch_list = [2, 4, 6, 8, 10]

    save_to_npy("D:\\contrast\\project2\\result_heart", "D:\\contrast\\project2\\heart_results_npy", model_list, epoch_list, min_val=-1000, max_val=2000)

def npy_to_png():

    model_list = ['cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024', 'cardiac_nce_1024']
    epoch_list = [2, 4, 6, 8, 10]
    input_folder = "D:\\contrast\\project2\\heart_results_npy"
    result_dir = "D:\\contrast\\project2\\heart_results_png"
    from_npy_to_png(input_folder, result_dir, model_list, epoch_list)

def visualize_demo():
    pass
if __name__ == "__main__":

    # model_list = ['cardiac_nce_1024']
    # ssim_heart(model_list, 10, "ssim_heart2.csv")

    # ssim_boxplot()
    fid_curve()

    # visualize_demo()