import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torchvision.transforms import RandomAffine, ToPILImage, RandomCrop, ToTensor, Resize
# from datasets import ImageDataset, ImageDatasetFromFile
from datasets0 import ImageDataset
from torch.utils.data import DataLoader
import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # opt.continue_train = True

    if opt.input_mode == 'patch':
        print("using patches...")
        transforms_1 = [ToPILImage(),

                        RandomAffine(degrees=1, translate=[0.02, 0.02],
                                     scale=[1 - 0.1, 1 + 0.4], fill=-1),
                        RandomCrop(size=opt.patch_size, pad_if_needed=True, fill=-1),
                        ToTensor()]

    else:
        print("full image input ...")
        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=1, translate=[0.1, 0.1],
                                     scale=[1 - 0.4, 1 + 0.4], fill=-1),
                        Resize((opt.load_size, opt.load_size)),
                        ToTensor()]

    if opt.read_folder:
        dataset = DataLoader(
            ImageDataset(opt.dataroot, transforms_1=transforms_1, unaligned=True), batch_size=opt.batch_size,
            shuffle=True, num_workers=opt.num_threads)
    else:
        raise ValueError("USE FOLDER INPUT")
        # dataset = DataLoader(
        #     ImageDatasetFromFile(opt.data_train_A, opt.data_train_B, transforms_1=transforms_1, unaligned=True,
        #                          max_itr=opt.max_itr, with_metric=opt.with_metric, metric_tol=opt.metric_tol,
        #                          ssim_th=opt.ssim_th),
        #     batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)

    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # dataset.set_epoch(epoch)
        # model.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print(data["A"].shape)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                # print("intialllllllllllllllllllllllllllllllll")
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
