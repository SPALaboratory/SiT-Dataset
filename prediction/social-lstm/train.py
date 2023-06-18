import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess

from model import SocialModel
from utils import DataLoader
from grid import getSequenceGridMask
from helper import *


def main():
    
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=46, 
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=35, 
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=17,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400, 
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32, 
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4, 
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27, 
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # drive option
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=7, 
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    
    args = parser.parse_args()
    
    train(args)


def train(args):
    origin = (0,0) 
    reference_point = (0,1) 
    validation_dataset_executed = False 
  
    prefix = ''
    f_prefix = '.'
    if args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'

    
    if not os.path.isdir("log/"):
      print("Directory creation script is running...")
      subprocess.call([f_prefix+'/make_directories.sh'])

    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs+1, args.freq_validation)) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ...]
    validation_epoch_list[-1]-=1 #30 -> 29



    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, args.num_validation, forcePreProcess=False) #JY21

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"


    # Log directory
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    plot_train_file_directory = 'validation'



    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name,'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')

    # model directory
    save_directory = os.path.join(prefix, 'model/')
    
    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name+str(x)+'.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.cuda()

    #optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    #optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)

    learning_rate = args.learning_rate

    best_val_loss = 100
    best_val_data_loss = 100

    smallest_err_val = 100000
    smallest_err_val_data = 100000


    best_epoch_val = 0
    best_epoch_val_data = 0

    best_err_epoch_val = 0
    best_err_epoch_val_data = 0

    all_epoch_results = []
    grids = []
    num_batch = 0
    dataset_pointer_ins_grid = -1 #Q_JY

    [grids.append([]) for dataset in range(dataloader.get_len_of_dataset())] #[[], []]

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        if dataloader.additional_validation and (epoch-1) in validation_epoch_list:
            dataloader.switch_to_dataset_type(True)
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
            loss_batch = 0
            
            #if we are in a new dataset, zero the counter of batch
            if dataset_pointer_ins_grid is not dataloader.dataset_pointer and epoch is not 0:
                num_batch = 0
                dataset_pointer_ins_grid = dataloader.dataset_pointer


            # For each sequence
            for sequence in range(dataloader.batch_size):
                # Get the data corresponding to the current sequence
                x_seq , y_seq , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                target_id = target_ids[sequence]

                #get processing file name and then get dimensions of file
                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader.get_dataset_dimension(folder_name)

                #dense vector creation
                x_seq, y_seq, lookup_seq = dataloader.convert_proper_array(x_seq, y_seq, numPedsList_seq, PedsList_seq)
                # target_id_values = x_seq[0][lookup_seq[target_id], 0:2]

                #grid mask calculation and storage depending on grid parameter
                if(args.grid):
                    # if(epoch is 0):
                    if True:
                        grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)
                        grids[dataloader.dataset_pointer].append(grid_seq)
                    # else:
                    #     grid_seq = grids[dataloader.dataset_pointer][(num_batch*dataloader.batch_size)+sequence]
                else:
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)

                # vectorize trajectories in sequence
                x_seq, y_seq, _ = vectorize_seq(x_seq, y_seq, PedsList_seq, lookup_seq) 
                if args.use_cuda:                    
                    x_seq = x_seq.cuda()
                    y_seq = y_seq.cuda()


                #number of peds in this sequence per frame
                numNodes = len(lookup_seq)


                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()
                

                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq,numPedsList_seq ,dataloader, lookup_seq, y_seq, dataset_data, False, target_id) #21JY

                
                # Compute loss
                loss = Gaussian2DLikelihood(outputs, y_seq, PedsList_seq[11:], lookup_seq, target_id) 
                loss_batch += loss.item()
                # loss_batch += loss

                # Compute gradients
                loss.backward()
                

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            num_batch+=1

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        loss_epoch /= dataloader.num_batches
        # Log loss values
        log_file_curve.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+'\n')

        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))


    if dataloader.valid_num_batches != 0:        
        print('Best epoch', best_epoch_val, 'Best validation Loss', best_val_loss, 'Best error epoch',best_err_epoch_val, 'Best error', smallest_err_val)
        # Log the best epoch and best validation loss
        log_file.write('Validation Best epoch:'+str(best_epoch_val)+','+' Best validation Loss: '+str(best_val_loss))

    if dataloader.additional_validation:
        print('Best epoch acording to validation dataset', best_epoch_val_data, 'Best validation Loss', best_val_data_loss, 'Best error epoch',best_err_epoch_val_data, 'Best error', smallest_err_val_data)
        log_file.write("Validation dataset Best epoch: "+str(best_epoch_val_data)+','+' Best validation Loss: '+str(best_val_data_loss)+'\n')
        #dataloader.write_to_plot_file(all_epoch_results[best_epoch_val_data], plot_directory)

    if validation_dataset_executed:
        dataloader.switch_to_dataset_type(load_data=False)
        create_directories(plot_directory, [plot_train_file_directory])
        dataloader.write_to_plot_file(all_epoch_results[len(all_epoch_results)-1], os.path.join(plot_directory, plot_train_file_directory))

    # Close logging files
    log_file.close()
    log_file_curve.close()



if __name__ == '__main__':
    main()