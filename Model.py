"""
SUpporting file for building the neural networks for predicting multimnist
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VanillaCnn(nn.Module):

    def __init__(self, n_classes, n_channels, h_params, h_in=64,w_in=84):
        """
        Create a  vanilla convolutional neural network

        :param n_classes: number of classes to classify
        :param h_params: dictionary of hyperparameters set to define the model
        :param h_in: height of the input images
        :param w_in: width of the input images
        """
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.h_params = h_params
        self.conv_layer_dict = dict()
        channel_in = n_channels
        self.pool = nn.MaxPool2d(self.h_params['pool_kernel'], self.h_params['pool_stride']).to(device)
        for i in range(self.h_params['n_cnn_layers']):
            self.conv_layer_dict['conv_'+str(i)] = nn.Conv2d(in_channels=channel_in,
                                                             out_channels=channel_in+self.h_params['channel_growth'],
                                                             kernel_size=self.h_params['kernel_size'],
                                                             stride=self.h_params['stride'],device=device)

            h_in, w_in = self.get_conv_output_dim(h_in, w_in, self.conv_layer_dict['conv_'+str(i)])
            h_in, w_in = self.get_pool_output_dim(h_in, w_in, self.pool)

            channel_in = channel_in+self.h_params['channel_growth'] # increase by growth rate


        self.fc1 = nn.Linear(channel_in * h_in * w_in, self.h_params['connect_layer_1'],device=device)
        self.fc2 = nn.Linear(self.h_params['connect_layer_1'], self.h_params['connect_layer_2'],device=device)
        self.fc3 = nn.Linear(self.h_params['connect_layer_2'], n_classes,device=device)

        self.dropout = nn.Dropout(self.h_params['dropout_rate']).to(device)

    def get_conv_output_dim(self,h_in,w_in,layer):
        """
        Helper function for getting the output dimensions of a convolutional layer

        :param h_in: height of input
        :param w_in: width of input
        :param layer: instantiated convolutional layer
        :return: output (height, width)
        """
        h_out = (h_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1
        h_out = int(np.floor(h_out))
        w_out = (w_in + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1
        w_out = int(np.floor(w_out))
        return h_out, w_out

    def get_pool_output_dim(self,h_in,w_in,layer):
        """
        Helper function for getting the output dimensions of a pooling layer

        :param h_in: height of input
        :param w_in: width of input
        :param layer: instantiated pooling layer
        :return: output (height, width)
        """
        h_out = (h_in + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride + 1
        h_out = int(np.floor(h_out))
        w_out = (w_in + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride + 1
        w_out = int(np.floor(w_out))
        return h_out, w_out

    def forward(self, x):
        """
        given an input tensor of data, calculate the output of the network

        :param x: input tensor representing an image
        :return: raw class probabilities predictions
        """

        # iterate over convolutional layers
        for i in range(self.h_params['n_cnn_layers']):
            x = self.conv_layer_dict['conv_'+str(i)](x)
            x = self.pool(F.relu(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def get_acc_dataset(self,criterion, dataset, device):
        """
        Get the classification accuracy given the data set. Also gets the loss and class predictions

        :param criterion: loss criterion
        :param dataset: dataset to get the accuracy over
        :param device: device the network is on
        :return:
            accuracy on the data set
            loss over the data set
            predicted class labels for the data set
        """
        inputs, labels = dataset.tensors
        self.eval()
        outputs = self.forward(inputs)
        loss = criterion(outputs, labels)

        class_pred = np.rint(outputs.to('cpu')).to(device)
        shape = labels.shape
        acc = torch.sum(class_pred == labels) / (shape[0] * shape[1])
        return acc, loss, class_pred

    def train_cnn(self, n_epochs, train_loader, validation_loader, test_loader):
        """
        Executes the training the neural network. Validation and testing accuracies are checked when the validation
        accuracy is better than all previous steps. The best model and some metrics are saved for later use.

        :param n_epochs: Number of epochs to train the network over
        :param train_loader: Training data set
        :param validation_loader: validation data set
        :param test_loader: testing data set
        :return: best validation accuracy over the training process
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())
        valid_acc_best = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            # run validation accuracy
            with torch.no_grad():
                self.eval()

                # validation metrics
                train_acc, train_loss, _ = self.get_acc_dataset(criterion, train_loader.dataset, device)


                # validation metrics
                valid_acc, val_loss, class_pred =self.get_acc_dataset(criterion, validation_loader.dataset,device)

                print('Epoch: {:d}\tTrain Loss: {:.5f}\tTrain Acc: {:.5f}\tValid Loss: {:.5f}\tValid Acc: {:.5f}'.format(epoch, running_loss,train_acc, val_loss,valid_acc))

                # update best model
                if valid_acc > valid_acc_best:
                    valid_acc_best = valid_acc

                    # save the current best model
                    torch.save(self.state_dict(),os.path.join('models',
                                self.h_params['study_name']+'_'+str(self.h_params['trial_num'])+'.torch'))

                    # Testing metrics
                    test_acc, test_loss, _ = self.get_acc_dataset(criterion,test_loader.dataset, device)

                    inputs, labels = validation_loader.dataset.tensors
                    cm=multilabel_confusion_matrix(labels.to('cpu'),class_pred.to('cpu'))
                    fig = plt.figure(0, figsize=(10, 8))
                    for k in range(10):
                        tmp_cm = cm[k,:,:]
                        disp = ConfusionMatrixDisplay(confusion_matrix=tmp_cm)
                        ax = fig.add_subplot(3,4,k+1)
                        ax.set_title('Class='+str(k))
                        disp.plot(ax=ax)
                        plt.tight_layout()
                    plt.savefig(os.path.join('models',
                                self.h_params['study_name']+'_'+str(self.h_params['trial_num'])+'.png'))
                    plt.close()

                    # write to dataframe for saving
                    metrics_file_name = os.path.join('models',self.h_params['study_name']+'_metrics.csv')
                    is_file = os.path.isfile(metrics_file_name)
                    if not os.path.isfile(metrics_file_name):
                        # create dataframe
                        col_names = ['trial_num','train_acc','train_loss','valid_acc','valid_loss','test_acc','test_loss']
                        df = pd.DataFrame(data=np.zeros((self.h_params['n_trials'],len(col_names))),columns=col_names)
                    else:
                        # open data frame
                        df = pd.read_csv(metrics_file_name,index_col=False)
                    df.loc[self.h_params['trial_num'],'trial_num'] = self.h_params['trial_num']
                    df.loc[self.h_params['trial_num'],'train_acc'] = float(train_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'train_loss'] = float(train_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'],'valid_acc'] = float(valid_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'valid_loss'] = float(val_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'],'test_acc'] = float(test_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'test_loss'] = float(test_loss.to('cpu'))
                    df.to_csv(metrics_file_name,index=False)

        return float(valid_acc_best.to('cpu'))


class CAE(nn.Module):

    def __init__(self, n_classes, n_channels, h_params, h_in=64, w_in=84):
        """
        Create a convolutional auto encoder neural network. This network has two tails, one tries to reproduce the image,
        and the second predicts the classes. The class prediction branches from the bottleneck layer of the CAE.

        :param n_classes: number of classes to classify
        :param h_params: dictionary of hyperparameters set to define the model
        :param h_in: height of the input images
        :param w_in: width of the input images
        """
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.h_params = h_params
        self.conv_layer_dict = dict()
        channel_in = n_channels
        #self.pool = nn.MaxPool2d(self.h_params['pool_kernel'], self.h_params['pool_stride']).to(device)

        # build encoder
        for i in range(self.h_params['n_cnn_layers']):
            self.conv_layer_dict['encode_conv_' + str(i)] = nn.Conv2d(in_channels=channel_in,
                                                               out_channels=channel_in + self.h_params[
                                                                   'channel_growth'],
                                                               kernel_size=self.h_params['kernel_size'],
                                                               stride=self.h_params['stride'], device=device)

            h_in, w_in = self.get_conv_output_dim(h_in, w_in, self.conv_layer_dict['encode_conv_' + str(i)])
            #h_in, w_in = self.get_pool_output_dim(h_in, w_in, self.pool)

            channel_in = channel_in + self.h_params['channel_growth']  # increase by growth rate

        # latent/bottleneck layer
        flatten_size = channel_in * h_in * w_in
        self.fc1 = nn.Linear(flatten_size, self.h_params['bottle_neck'], device=device)
        self.fc2 = nn.Linear(self.h_params['bottle_neck'], flatten_size, device=device)

        # decoder
        for i in range(self.h_params['n_cnn_layers']):
            dec_idx = self.h_params['n_cnn_layers']-i-1

            mirror_conv = self.conv_layer_dict['encode_conv_' + str(dec_idx)]
            self.conv_layer_dict['decode_conv_' + str(i)] = nn.ConvTranspose2d(in_channels=mirror_conv.out_channels,out_channels=mirror_conv.in_channels,
                               kernel_size=mirror_conv.kernel_size,stride=mirror_conv.stride,device=device)

        # classifier parts
        self.class_fc1 = nn.Linear(self.h_params['bottle_neck'],self.h_params['class_layer_1'],device=device)
        self.class_fc2 = nn.Linear(self.h_params['class_layer_1'], self.h_params['class_layer_2'], device=device)
        self.class_fc3 = nn.Linear(self.h_params['class_layer_2'], n_classes, device=device)

        self.dropout = nn.Dropout(self.h_params['dropout_rate']).to(device)

    def get_conv_output_dim(self, h_in, w_in, layer):
        """
        Helper function for getting the output dimensions of a convolutional layer

        :param h_in: height of input
        :param w_in: width of input
        :param layer: instantiated convolutional layer
        :return: output (height, width)
        """
        h_out = (h_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1
        h_out = int(np.floor(h_out))
        w_out = (w_in + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1
        w_out = int(np.floor(w_out))
        return h_out, w_out

    def forward(self, x):
        """
        given an input tensor of data, calculate the output of the network

        :param x: input tensor representing an image
        :return:
            raw class probabilities predictions
            class predictions
            values of the latent variables
        """

        # encoder
        for i in range(self.h_params['n_cnn_layers']):
            x = self.conv_layer_dict['encode_conv_' + str(i)](x)
            #x = self.pool(F.relu(x))

        # post encoder size
        enc_shape = x.shape

        # bottleneck
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        latent_space = F.relu(self.fc1(x))
        x = F.relu(self.fc2(latent_space))
        x = torch.unflatten(x,1,tuple(enc_shape[1:]))

        # decoder
        for i in range(self.h_params['n_cnn_layers']):
            x = self.conv_layer_dict['decode_conv_' + str(i)](x)
        x = F.sigmoid(x)

        # class predictor
        class_pred = self.dropout(latent_space)
        class_pred = F.relu(self.class_fc1(class_pred))
        class_pred = F.relu(self.class_fc2(class_pred))
        class_pred = F.sigmoid(self.class_fc3(class_pred))


        return x, class_pred, latent_space

    def get_acc_dataset(self, criterion_ae, criterion_class, dataset, device):
        """
        Get the classification accuracy given the data set. Also gets the loss and class predictions

        :param criterion_ae: loss criterion for the autoencoder tail
        :param criterion_class: loss criterion for the classification tail
        :param dataset: dataset to get the accuracy over
        :param device: device the network is on
        :return:
            accuracy on the data set
            loss over the data set
            predicted class labels for the data set
        """
        inputs, labels = dataset.tensors
        self.eval()
        outputs, class_pred, bottle_neck = self.forward(inputs)
        loss_ae = criterion_ae(outputs, inputs)
        loss_class = criterion_class(class_pred, labels)
        loss = loss_ae + loss_class

        class_pred = np.rint(class_pred.to('cpu')).to(device)
        shape = labels.shape
        acc = torch.sum(class_pred == labels) / (shape[0] * shape[1])
        return acc, loss, class_pred

    def train_cnn(self, n_epochs, train_loader, validation_loader, test_loader):
        """
        Executes the training the neural network. Validation and testing accuracies are checked when the validation
        accuracy is better than all previous steps. The best model and some metrics are saved for later use.

        :param n_epochs: Number of epochs to train the network over
        :param train_loader: Training data set
        :param validation_loader: validation data set
        :param test_loader: testing data set
        :return: best validation accuracy over the training process
        """
        criterion_ae = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())
        valid_acc_best = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, class_pred, bottle_neck = self.forward(inputs)
                loss_ae = criterion_ae(outputs, inputs)
                loss_class = criterion_class(class_pred,labels)
                loss = loss_ae+loss_class
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            # run validation accuracy
            with torch.no_grad():
                self.eval()

                # training metrics
                train_acc, train_loss, _ = self.get_acc_dataset(criterion_ae, criterion_class,
                                                                       train_loader.dataset, device)

                # validation metrics
                valid_acc, val_loss, class_pred = self.get_acc_dataset(criterion_ae, criterion_class,
                                                                       validation_loader.dataset, device)

                print(
                    'Epoch: {:d}\tTrain Loss: {:.5f}\tTrain Acc: {:.5f}\tValid Loss: {:.5f}\tValid Acc: {:.5f}'.format(
                        epoch, running_loss, train_acc, val_loss, valid_acc))

                # update best model
                if valid_acc > valid_acc_best:
                    valid_acc_best = valid_acc

                    # save the current best model
                    torch.save(self.state_dict(), os.path.join('models',
                                                               self.h_params['study_name'] + '_' + str(
                                                                   self.h_params['trial_num']) + '.torch'))

                    # Testing metrics
                    test_acc, test_loss, _ = self.get_acc_dataset(criterion_ae, criterion_class, test_loader.dataset, device)

                    inputs, labels = validation_loader.dataset.tensors
                    cm = multilabel_confusion_matrix(labels.to('cpu'), class_pred.to('cpu'))
                    fig = plt.figure(0, figsize=(10, 8))
                    for k in range(10):
                        tmp_cm = cm[k, :, :]
                        disp = ConfusionMatrixDisplay(confusion_matrix=tmp_cm)
                        ax = fig.add_subplot(3, 4, k + 1)
                        ax.set_title('Class=' + str(k))
                        disp.plot(ax=ax)
                        plt.tight_layout()
                    plt.savefig(os.path.join('models',
                                             self.h_params['study_name'] + '_' + str(
                                                 self.h_params['trial_num']) + '.png'))
                    plt.close()

                    # write to dataframe for saving
                    metrics_file_name = os.path.join('models', self.h_params['study_name'] + '_metrics.csv')
                    is_file = os.path.isfile(metrics_file_name)
                    if not os.path.isfile(metrics_file_name):
                        # create dataframe
                        col_names = ['trial_num', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss', 'test_acc',
                                     'test_loss']
                        df = pd.DataFrame(data=np.zeros((self.h_params['n_trials'], len(col_names))), columns=col_names)
                    else:
                        # open data frame
                        df = pd.read_csv(metrics_file_name, index_col=False)
                    df.loc[self.h_params['trial_num'], 'trial_num'] = self.h_params['trial_num']
                    df.loc[self.h_params['trial_num'], 'train_acc'] = float(train_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'], 'train_loss'] = float(train_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'], 'valid_acc'] = float(valid_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'], 'valid_loss'] = float(val_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'], 'test_acc'] = float(test_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'], 'test_loss'] = float(test_loss.to('cpu'))
                    df.to_csv(metrics_file_name, index=False)

        return float(valid_acc_best.to('cpu'))
    
lass DenseNetBC(nn.Module):

    def __init__(self, n_classes, h_params, h_in=64, w_in=84):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.h_params = h_params
        self.blocks = nn.ModuleList()
        channel_in = 1  # grayscale images

        self.pool = nn.MaxPool2d(self.h_params['pool_kernel'], self.h_params['pool_stride']).to(device)

        for i in range(self.h_params['n_dense_layers']):
            bottleneck = nn.Conv2d(channel_in, self.h_params['bottleneck_width'], kernel_size=1, stride=1, padding=0, bias=False).to(device)
            conv = nn.Conv2d(self.h_params['bottleneck_width'], self.h_params['growth_rate'],
                             kernel_size=self.h_params['kernel_size'],
                             stride=self.h_params['stride'],
                             padding=self.h_params['padding'],
                             bias=False).to(device)
            self.blocks.append(nn.ModuleDict({
                'bottleneck': bottleneck,
                'conv': conv
            }))

            h_in, w_in = self.get_conv_output_dim(h_in, w_in, conv)
            h_in, w_in = self.get_pool_output_dim(h_in, w_in, self.pool)
            channel_in += self.h_params['growth_rate']

            # Compression layer
            out_channels = int(channel_in * self.h_params['compression'])
            self.blocks.append(nn.Conv2d(channel_in, out_channels, kernel_size=1, stride=1, padding=0, bias=False).to(device))
            channel_in = out_channels

        self.fc1 = nn.Linear(channel_in * h_in * w_in, self.h_params['connect_layer_1'], device=device)
        self.fc2 = nn.Linear(self.h_params['connect_layer_1'], self.h_params['connect_layer_2'], device=device)
        self.fc3 = nn.Linear(self.h_params['connect_layer_2'], n_classes, device=device)

        self.dropout = nn.Dropout(self.h_params['dropout_rate']).to(device)

    def get_conv_output_dim(self, h_in, w_in, layer):
        h_out = (h_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1
        h_out = int(np.floor(h_out))
        w_out = (w_in + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1
        w_out = int(np.floor(w_out))
        return h_out, w_out

    def get_pool_output_dim(self, h_in, w_in, layer):
        h_out = (h_in + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride + 1
        h_out = int(np.floor(h_out))
        w_out = (w_in + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) / layer.stride + 1
        w_out = int(np.floor(w_out))
        return h_out, w_out
    def forward(self, x):
        for block in self.blocks:
            if isinstance(block, nn.ModuleDict):
                out = F.relu(block['bottleneck'](x))
                out = F.relu(block['conv'](out))
                x = torch.cat([x, out], dim=1)
                x = self.pool(x)
            else:
                x = block(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)   #
        return x

    def get_acc_dataset(self, criterion, dataset, device):
        inputs, labels = dataset.tensors
        self.eval()
        outputs = self.forward(inputs)
        loss = criterion(outputs, labels)

        preds = torch.sigmoid(outputs)   # Apply sigmoid for prediction
        class_pred = torch.round(preds).to(device)  # Now safely threshold at 0.5
        shape = labels.shape
        acc = torch.sum(class_pred == labels) / (shape[0] * shape[1])
        return acc, loss, class_pred

    def train_cnn(self, n_epochs, train_loader, validation_loader, test_loader):
        criterion = nn.BCEWithLogitsLoss()   #
        optimizer = optim.Adam(self.parameters())
        valid_acc_best = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(n_epochs):
            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                self.eval()
                train_acc, train_loss, _ = self.get_acc_dataset(criterion, train_loader.dataset, device)
                valid_acc, val_loss, class_pred = self.get_acc_dataset(criterion, validation_loader.dataset, device)

                print('Epoch: {:d}\tTrain Loss: {:.5f}\tTrain Acc: {:.5f}\tValid Loss: {:.5f}\tValid Acc: {:.5f}'.format(
                    epoch, running_loss, train_acc, val_loss, valid_acc))

                if valid_acc > valid_acc_best:
                    valid_acc_best = valid_acc

                    torch.save(self.state_dict(), os.path.join('models',
                                self.h_params['study_name']+'_'+str(self.h_params['trial_num'])+'.torch'))

                    test_acc, test_loss, _ = self.get_acc_dataset(criterion, test_loader.dataset, device)

                    inputs, labels = validation_loader.dataset.tensors
                    cm = multilabel_confusion_matrix(labels.to('cpu'), class_pred.to('cpu'))
                    fig = plt.figure(0, figsize=(10, 8))
                    for k in range(10):
                        tmp_cm = cm[k,:,:]
                        disp = ConfusionMatrixDisplay(confusion_matrix=tmp_cm)
                        ax = fig.add_subplot(3,4,k+1)
                        ax.set_title('Class='+str(k))
                        disp.plot(ax=ax)
                        plt.tight_layout()
                    plt.savefig(os.path.join('models', self.h_params['study_name']+'_'+str(self.h_params['trial_num'])+'.png'))
                    plt.close()

                    metrics_file_name = os.path.join('models', self.h_params['study_name']+'_metrics.csv')
                    if not os.path.isfile(metrics_file_name):
                        col_names = ['trial_num','train_acc','train_loss','valid_acc','valid_loss','test_acc','test_loss']
                        df = pd.DataFrame(data=np.zeros((self.h_params['n_trials'],len(col_names))),columns=col_names)
                    else:
                        df = pd.read_csv(metrics_file_name,index_col=False)
                    df.loc[self.h_params['trial_num'],'trial_num'] = self.h_params['trial_num']
                    df.loc[self.h_params['trial_num'],'train_acc'] = float(train_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'train_loss'] = float(train_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'],'valid_acc'] = float(valid_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'valid_loss'] = float(val_loss.to('cpu'))
                    df.loc[self.h_params['trial_num'],'test_acc'] = float(test_acc.to('cpu'))
                    df.loc[self.h_params['trial_num'],'test_loss'] = float(test_loss.to('cpu'))
                    df.to_csv(metrics_file_name,index=False)

        return float(valid_acc_best.to('cpu'))