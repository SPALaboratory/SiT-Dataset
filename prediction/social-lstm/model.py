import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from grid import getGridMask

class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0] #torch.Size([20, 3, 2])
        target_data = args[8] #21JY
        grids = args[1]
        hidden_states = args[2] #torch.Size([3, 128])
        cell_states = args[3] #torch.Size([3, 128])

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]

        dimensions = args[9]
        validation_step = args[10]
        target_id = args[11]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(35 * numNodes, self.output_size)) #21JY
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]] #[5, 6, 8]

            if len(nodeIDs) == 0:
                continue

            list_of_nodes = [look_up[x] for x in nodeIDs] #[0, 1, 2]

            corr_index = Variable((torch.LongTensor(list_of_nodes))) #tensor([0, 1, 2])
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            nodes_current = frame[list_of_nodes,:] #torch.Size([3, 2])
            grid_current = grids[framenum] #torch.Size([3, 3, 16])

            hidden_states_current = torch.index_select(hidden_states, 0, corr_index) #torch.Size([3, 128])


            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            social_tensor = self.getSocialTensor(grid_current, hidden_states_current) #torch.Size([3, 2048])

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current))) #torch.Size([3, 64])
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor))) #torch.Size([3, 64])

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))
            
            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        #21JY
        if validation_step == False:
            nodes_current = self.output_layer(h_nodes)
            outputs[0*numNodes + corr_index.data] = nodes_current
            for framenum,frame in enumerate(target_data):

                if framenum == 34:
                    continue

                nodeIDs = [int(nodeID) for nodeID in PedsList[11+framenum]] #[5, 6, 8]

                if len(nodeIDs) == 0:
                    continue

                list_of_nodes = [look_up[x] for x in nodeIDs] #[0, 1, 2]

                corr_index = Variable((torch.LongTensor(list_of_nodes))) #tensor([0, 1, 2])
                if self.use_cuda:            
                    corr_index = corr_index.cuda()

                # hidden_states_current = torch.index_select(hidden_states, 0, corr_index) #torch.Size([3, 128])

                # nodes_current = self.output_layer(hidden_states_current)
                # outputs[framenum*numNodes + corr_index.data] = nodes_current

                nodes_current = frame[list_of_nodes,:] #torch.Size([3, 2])
                grid_current = Variable(torch.from_numpy(getGridMask(nodes_current.cpu(), dimensions, len(corr_index), 32, 4, False)).float()) #torch.Size([3, 3, 16])
                if self.use_cuda:
                    grid_current = grid_current.cuda()

                hidden_states_current = torch.index_select(hidden_states, 0, corr_index) #torch.Size([3, 128])

                if not self.gru:
                    cell_states_current = torch.index_select(cell_states, 0, corr_index)

                social_tensor = self.getSocialTensor(grid_current, hidden_states_current) #torch.Size([3, 2048])

                # Embed inputs
                input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current))) #torch.Size([3, 64])
                # Embed the social tensor
                tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor))) #torch.Size([3, 64])

                # Concat input
                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                if not self.gru:
                    # One-step of the LSTM
                    h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                else:
                    h_nodes = self.cell(concat_embedded, (hidden_states_current))


                # Compute the output
                outputs[(framenum+1)*numNodes + corr_index.data] = self.output_layer(h_nodes)

                # Update hidden and cell states
                hidden_states[corr_index.data] = h_nodes
                if not self.gru:
                    cell_states[corr_index.data] = c_nodes
        else:
            nodes_current = self.output_layer(h_nodes)
            outputs[0*numNodes + corr_index.data] = nodes_current
            for framenum in range(35-1):

                nodeIDs = [int(nodeID) for nodeID in PedsList[11+framenum]] #[5, 6, 8]

                if len(nodeIDs) == 0:
                    continue

                list_of_nodes = [look_up[x] for x in nodeIDs] #[0, 1, 2]

                corr_index = Variable((torch.LongTensor(list_of_nodes))) #tensor([0, 1, 2])
                if self.use_cuda:            
                    corr_index = corr_index.cuda()

                # nodes_current = self.output_layer(hidden_states_current)
                # outputs[framenum*numNodes + corr_index.data] = nodes_current

                # nodes_current = torch.index_select(nodes_current, 0, corr_index) #torch.Size([3, 128])
                nodes_current = outputs[framenum*numNodes + corr_index.data]

                # nodes_current = frame[list_of_nodes,:] #torch.Size([3, 2])
                grid_current = Variable(torch.from_numpy(getGridMask(nodes_current[:, :2].cpu(), dimensions, len(corr_index), 32, 4, False)).float()) #torch.Size([3, 3, 16])
                if self.use_cuda:
                    grid_current = grid_current.cuda()

                hidden_states_current = torch.index_select(hidden_states, 0, corr_index) #torch.Size([3, 128])

                if not self.gru:
                    cell_states_current = torch.index_select(cell_states, 0, corr_index)

                social_tensor = self.getSocialTensor(grid_current, hidden_states_current) #torch.Size([3, 2048])

                # Embed inputs
                input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current[:, :2]))) #torch.Size([3, 64])
                # Embed the social tensor
                tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor))) #torch.Size([3, 64])

                # Concat input
                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                if not self.gru:
                    # One-step of the LSTM
                    h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                else:
                    h_nodes = self.cell(concat_embedded, (hidden_states_current))

                # Compute the output
                outputs[(framenum+1)*numNodes + corr_index.data] = self.output_layer(h_nodes)

                # Update hidden and cell states
                hidden_states[corr_index.data] = h_nodes
                if not self.gru:
                    cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(35, numNodes, self.output_size)) #21JY
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(35): #21JY
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
