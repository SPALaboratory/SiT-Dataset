import os
import pickle
import argparse
import time
import subprocess


import torch
from torch.autograd import Variable

import numpy as np
from utils import DataLoader
from helper import get_mean_error, get_final_error

from helper import *
from grid import getSequenceGridMask


def main():
    
    parser = argparse.ArgumentParser()
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=17,
                        help='Epoch of model to be loaded')
    
    parser.add_argument('--seq_length', type=int, default=46,
                        help='RNN sequence length')

    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')

    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=7,
                        help='Total number of validation dataset will be visualized')
    # gru support
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    #for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'
    

    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # Save directory
    save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
    #plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

    plot_validation_file_directory = 'validation'



    # Define the path for the config file for saved args
    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        args = pickle.load(f)

    origin = (0,0)
    reference_point = (0,1)
    net = get_model(sample_args.method, args, True)
    if sample_args.use_cuda:        
        net = net.cuda()

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)

    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 5, sample_args.seq_length, num_of_validation = sample_args.num_validation, forcePreProcess = False, infer = True)
    create_directories(plot_directory, [plot_validation_file_directory])
    dataloader.reset_batch_pointer()

    print('****************Validation dataset batch processing******************')
    dataloader.reset_batch_pointer(valid=False)
    dataset_pointer_ins = dataloader.dataset_pointer

    loss_epoch = 0
    err_epoch_1s = 0
    f_err_epoch_1s = 0
    err_epoch_3s = 0
    f_err_epoch_3s = 0
    err_epoch_5s = 0
    f_err_epoch_5s = 0
    err_epoch = 0
    f_err_epoch = 0
    num_of_batch = 0
    smallest_err = 100000

    smallest_err_val_data = 100
    best_val_data_loss = 100

    #results of one epoch for all validation datasets
    epoch_result = []
    #results of one validation dataset
    results = []



    # For each batch
    for batch in range(dataloader.num_batches):
        start = time.time()
        # Get batch data
        x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch()

        if dataset_pointer_ins is not dataloader.dataset_pointer:
            if dataloader.dataset_pointer is not 0:
                print('Finished prosessed file : ', dataloader.get_file_name(-1),' Avarage error : ', err_epoch/num_of_batch)
                num_of_batch = 0
                epoch_result.append(results)

            dataset_pointer_ins = dataloader.dataset_pointer
            results = []



        # Loss for this batch
        loss_batch = 0
        err_batch_1s = 0
        f_err_batch_1s = 0
        err_batch_3s = 0
        f_err_batch_3s = 0
        err_batch_5s = 0
        f_err_batch_5s = 0
        err_batch = 0
        f_err_batch = 0

        # For each sequence
        for sequence in range(dataloader.batch_size):
            # Get data corresponding to the current sequence
            x_seq , y_seq, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
            target_id = target_ids[sequence]

            #get processing file name and then get dimensions of file
            folder_name = dataloader.get_directory_name_with_pointer(d_seq)
            dataset_data = dataloader.get_dataset_dimension(folder_name)


            
            #dense vector creation
            x_seq, y_seq, lookup_seq = dataloader.convert_proper_array(x_seq, y_seq, numPedsList_seq, PedsList_seq)
            
            #will be used for error calculation
            orig_x_seq = x_seq.clone() 
            orig_y_seq = y_seq.clone() 
            
            # target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:2]
            
            #grid mask calculation
            if sample_args.method == 2: #obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda, True)
            elif  sample_args.method == 1: #social lstm   
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
            
            if args.use_cuda:
                x_seq = x_seq.cuda()
                y_seq = y_seq.cuda()
                orig_x_seq = orig_x_seq.cuda()
                orig_y_seq = orig_y_seq.cuda()

            #vectorize datapoints
            x_seq, y_seq, first_values_dict = vectorize_seq(x_seq, y_seq, PedsList_seq, lookup_seq)

            #sample predicted points from model
            if sample_args.method == 3: #vanilla lstm
                ret_x_seq, loss = sample_validation_data_vanilla_5(x_seq, y_seq, PedsList_seq, sample_args, net, lookup_seq, numPedsList_seq, dataloader, dataset_data, target_id)

            else:
            	ret_x_seq, loss = sample_validation_data_5(x_seq, y_seq, PedsList_seq, grid_seq, args, net, lookup_seq, numPedsList_seq, dataloader, dataset_data, target_id)

            #revert the points back to original space
            ret_x_seq = revert_seq_5(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)

            #get mean and final error
            err, err_1s, err_2s, err_3s, err_5s, err_f_1s, err_f_2s, err_f_3s, err_f_5s = \
                get_mean_error_5(ret_x_seq.data, orig_y_seq.data, PedsList_seq[11:], PedsList_seq[11:], args.use_cuda, lookup_seq, target_id)
            f_err = get_final_error_5(ret_x_seq.data, orig_y_seq.data, PedsList_seq[11:], PedsList_seq[11:], lookup_seq, target_id)
            
            loss_batch += loss.item()
            err_batch_1s += err_1s
            f_err_batch_1s += err_f_1s
            err_batch_3s += err_3s
            f_err_batch_3s += err_f_3s
            err_batch_5s += err_5s
            f_err_batch_5s += err_f_5s
            err_batch += err
            f_err_batch += f_err
            print('Current file : ', dataloader.get_file_name(0),' Batch : ', batch+1, ' Sequence: ', sequence+1, ' Sequence mean error: ', err,' Sequence final error: ',f_err)
            results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq, dataloader.get_frame_sequence(args.seq_length), target_id))

        loss_batch = loss_batch / dataloader.batch_size
        err_batch_1s = err_batch_1s / dataloader.batch_size
        f_err_batch_1s = f_err_batch_1s / dataloader.batch_size
        err_batch_3s = err_batch_3s / dataloader.batch_size
        f_err_batch_3s = f_err_batch_3s / dataloader.batch_size
        err_batch_5s = err_batch_5s / dataloader.batch_size
        f_err_batch_5s = f_err_batch_5s / dataloader.batch_size
        err_batch = err_batch / dataloader.batch_size
        f_err_batch = f_err_batch / dataloader.batch_size
        num_of_batch += 1
        loss_epoch += loss_batch
        err_epoch_1s += err_batch_1s
        f_err_epoch_1s += f_err_batch_1s
        err_epoch_3s += err_batch_3s
        f_err_epoch_3s += f_err_batch_3s
        err_epoch_5s += err_batch_5s
        f_err_epoch_5s += f_err_batch_5s
        err_epoch += err_batch
        f_err_epoch += f_err_batch

    epoch_result.append(results)


    if dataloader.num_batches != 0:            
        loss_epoch = loss_epoch / dataloader.num_batches
        err_epoch_1s = err_epoch_1s / dataloader.num_batches
        f_err_epoch_1s = f_err_epoch_1s / dataloader.num_batches
        err_epoch_3s = err_epoch_3s / dataloader.num_batches
        f_err_epoch_3s = f_err_epoch_3s / dataloader.num_batches
        err_epoch_5s = err_epoch_5s / dataloader.num_batches
        f_err_epoch_5s = f_err_epoch_5s / dataloader.num_batches
        err_epoch = err_epoch / dataloader.num_batches
        f_err_epoch = f_err_epoch / dataloader.num_batches
        avarage_err = (err_epoch + f_err_epoch)/2

        # Update best validation loss until now
        if loss_epoch < best_val_data_loss:
            best_val_data_loss = loss_epoch

        if avarage_err<smallest_err_val_data:
            smallest_err_val_data = avarage_err

        f=open("./txt/validation_deepenai_social_lstm_sample5.txt", 'a') #JY
        f.write('\nade1s = {:.3f}, fde1s = {:.3f}, ade3s = {:.3f}, fde3s = {:.3f}, ade5s = {:.3f}, fde5s = {:.3f}, ade = {:.3f}, fde = {:.3f}\n'\
            .format(err_epoch_1s, f_err_epoch_1s, err_epoch_3s, f_err_epoch_3s, err_epoch_5s, f_err_epoch_5s, err_epoch, f_err_epoch))
        f.close()
        # print('(epoch {}), valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}'.format(epoch, loss_epoch, err_epoch, f_err_epoch))

    # dataloader.write_to_plot_file(epoch_result, os.path.join(plot_directory, plot_validation_file_directory))


if __name__ == '__main__':
    main()


