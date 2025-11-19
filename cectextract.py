import SimpleITK as sitk
import os
import pandas as pd
import random

seed = 1234
def pick_cect_series():
    root_dir = "D:\\PE\\rsna-str-pulmonary-embolism-detection\\train"
    file_list = os.listdir(root_dir)

    id_list = list(range(len(file_list)))
    print(id_list)
    random.seed(seed)
    random.shuffle(id_list)
    print(id_list)

    # id_list_100 = id_list[:100]

    target_folder = "D:\\contrast\\image\\contrast17"
    cnt = 0
    for idx in id_list:
        folder = file_list[idx]
        for folder2 in os.listdir(os.path.join(root_dir, folder)):
            series_reader = sitk.ImageSeriesReader()
            path = os.path.join(root_dir, folder, folder2)
            print(path)
            dicom_names = series_reader.GetGDCMSeriesFileNames(path)
            print(len(dicom_names))
            if len(dicom_names) > 0:
                series_reader.SetFileNames(dicom_names)
                image = series_reader.Execute()

                sitk.WriteImage(image, os.path.join(target_folder, folder + "_"+folder2+".nii"))
                cnt += 1

        if cnt >= 17:
            break

        # break
import shutil
def pick_ncct_series():
    # old
    # table_path = "D:\\contrast\\project\\pe_contrast_transfer\\nonpaired_noncontrast.csv"
    #
    # paths = pd.read_csv(table_path)['path'].values
    #
    # target_folder = "D:\\contrast\\image\\noncontrast100"
    # for path in paths:
    #     print(path)
    #     name = path.split('\\')[-1]
    #     print(name)
    #     shutil.copy(path, os.path.join(target_folder, name))

    # new
    root_dir = "D:\\contrast\\image\\noncontrast15\\"
    src_dir = os.path.join(root_dir, "sccor")
    tgt_dir = root_dir

    folder_list = os.listdir(src_dir)

    for folder in folder_list:
        print(folder)
        datefolder = os.listdir(os.path.join(src_dir, folder))[0]
        print(datefolder)
        seriesfolder = os.listdir(os.path.join(src_dir, folder, datefolder))[0]
        print(seriesfolder)

        dcm_folder = os.path.join(os.path.join(src_dir, folder, datefolder, seriesfolder))

        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(dcm_folder)
        reader.SetFileNames(dicom_names)

        reader_info = sitk.ImageFileReader()
        reader_info.SetFileName(dicom_names[0])
        reader_info.LoadPrivateTagsOn()
        reader_info.ReadImageInformation()
        #
        # for k in reader_info.GetMetaDataKeys():
        #     v = reader_info.GetMetaData(k)
        #     print(f'({k}) = = "{v}"')
        # 0010|0020
        keys1 = ['0010|0020', '0010|0010']
        # keydate = '0008|0020'
        keys2 = ['0020|000d']

        name = ""
        for k in keys1:
            v = reader_info.GetMetaData(k)
            name += v.strip()

        name += "_"

        v_date = datefolder.split("_")[-1]
        v_date = v_date[0:4] + "-" + v_date[4:6] + "-" + v_date[6:]

        name += v_date

        name += "_"

        name += seriesfolder.strip()

        name += "_"
        for k in keys2:
            v = reader_info.GetMetaData(k)
            name += v.strip()
            name += "_"

        name += "img.nii"
        print(name)


        image = reader.Execute()

        size = image.GetSize()
        print(size)
        sitk.WriteImage(image, os.path.join(tgt_dir, name))
        # break





def check_spacing():
    folder = "D:\\contrast\\image\\contrast100"
    file_list = os.listdir(folder)

    for file in file_list:
        path = os.path.join(folder, file)

        img = sitk.ReadImage(path)
        print(img.GetSpacing())




if __name__ == '__main__':
    # check_spacing()
    # pick_ncct_series()
    pick_cect_series()




