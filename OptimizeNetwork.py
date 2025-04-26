"""
Optimizes the networks for classification of the multi-mnist data set via hyperparameter optimzation
"""

# native modules

# 3rd party modules
import kagglehub
import numpy as np
import optuna
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset

# own modules
import Model

def download_data():
    """
    # only need to invoke this to download the data
    :return:
    """

    # Download latest version
    path = kagglehub.dataset_download("farhanhubble/multimnistm2nist")

    print("Path to dataset files:", path)

def gen_labels(data):
    """
    Given the data set is originally for image segmentation, this extracts data so that classification may be completed
    on the data set. The inpput are transformed in two ways. The first is to extract the plane in the input tensor
    that has the hand written digit. This data set has up to three hand written digits composited. The extracted plane
    is placed in a random channel for the input. The height, and width of the pixels are preserved. THe channels denote
    different numbers. The randomization of which channel the number goes into prevents patterns there the network
    can exploit. The second method is to put all three numbers that are composited into a single changell. If multiple
    numbers overlap, a maximum input value of 1 is used. E.g. the max at a given (x,y) pixel across the input depth is
    used. Lastly the labels are extracted for which number exists for each image.

    :param data:
    :return:
    """
    n_data, height, width, n_classes = data.shape
    labels = np.zeros((n_data, n_classes-1)) # drop last column as that denotes background with now numbers
    np.random.seed(0)

    x_extract = np.zeros((n_data, height, width,3))
    x_extract_comp = np.zeros((n_data, height, width, 1))
    for i in range(n_data):
        # draw random order for composite number
        idx = np.arange(3)
        np.random.shuffle(idx)
        k = 0

        for j in range(n_classes - 1):
            tmp = data[i,:,:,j]
            if tmp.max() == 1:
                # number denoted here. place in random index in x data
                x_extract[i,:,:,idx[k]] = tmp
                k += 1
        x_extract_comp[i,:,:,0] = np.max(data[i, :, :, :n_classes-1], axis=2)

    # generate one hot labels
    for i in range(n_data):
        for j in range(n_classes-1):
            labels[i, j] = data[i, :, :, j].max()

    # change shape to have, (n_data, n_channels, x, y)
    x_extract = np.transpose(x_extract,(0,3,1,2))
    x_extract_comp = np.transpose(x_extract_comp, (0,3,1,2))

    return x_extract, x_extract_comp, labels

def load_data(compress_channel):
    """
    Loads the dataset from a locally saved numpy file
    :return:
        input training data
        targets for training data
        input validation data
        targets for validation data
        input for testing data
        targets for testing data
        number of classes in the data set
    """
    segmented = np.load("data/segmented.npy")
    n_data, height, width, n_classes = segmented.shape
    n_classes -=1

    training_data, validation_data, test_data = torch.utils.data.random_split(segmented,[0.7,0.2,0.1],generator=torch.Generator().manual_seed(0))

    if compress_channel:
        _, x_train, y_train = gen_labels(training_data.dataset[training_data.indices])

        _, x_valid, y_valid = gen_labels(validation_data.dataset[validation_data.indices])
        _, x_test, y_test = gen_labels(test_data.dataset[test_data.indices])
    else:
        x_train, _, y_train = gen_labels(training_data.dataset[training_data.indices])

        x_valid, _, y_valid = gen_labels(validation_data.dataset[validation_data.indices])
        x_test, _, y_test = gen_labels(test_data.dataset[test_data.indices])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, n_classes

def objective_Vanilla_Cnn(trial: optuna.Trial) -> float:
    """
    Optuna objective function for optimizing the hyperparameters for the vanilla convolutional net. This is given
    to the optuna study to sample and evaluate the model.

    :param trial: The optuna trial that defines how the next sample is derived
    :return: accuracy of the model over the validation data
    """

    # hyperparameters to optimize
    dropout_rate = trial.suggest_float("dropout",0.0,0.5)
    n_cnn_layers = trial.suggest_int("num_cnn_layers",2,4)
    kernel_size = trial.suggest_int('kernel_size',3,10)
    stride = 1 #trial.suggest_int('stride',1,1)
    channel_growth = trial.suggest_int('channel_growth',0,4)
    pool_kernel = 2 #trial.suggest_int('pool_kernel',2,2)
    pool_stride = 1 #trial.suggest_int('pool_stride',1,1)
    connect_layer_1 = trial.suggest_int('connect_layer_1',16,1028)
    connect_layer_2 = trial.suggest_int('connect_layer_2', 16, 1028)
    batch_size = trial.suggest_int('batch_size',16,1028)

    h_params = {'dropout_rate':dropout_rate,
                'n_cnn_layers':n_cnn_layers,
                'kernel_size':kernel_size,
                'stride':stride,
                'channel_growth':channel_growth,
                'pool_kernel': pool_kernel,
                'pool_stride':pool_stride,
                'connect_layer_1':connect_layer_1,
                'connect_layer_2':connect_layer_2,
                'trial_num':trial.number,
                'study_name':trial.study.study_name}

    compress_channel = True
    x_train, y_train, x_valid, y_valid, x_test, y_test, n_classes = load_data(compress_channel)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = TensorDataset(torch.Tensor(x_valid).to(device), torch.Tensor(y_valid).to(device))
    valid_dataloader = DataLoader(valid_dataset)

    test_dataset = TensorDataset(torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device))
    test_dataloader = DataLoader(test_dataset)

    n_channels = x_train.shape[1]
    cnn = Model.VanillaCnn(n_classes,n_channels,h_params)
    acc = cnn.train_cnn(100, train_dataloader, valid_dataloader, test_dataloader)
    return acc

def objective_CAE(trial: optuna.Trial) -> float:
    """
    Optuna objective function for optimizing the hyperparameters for the convolutional autoencoder. This is given
    to the optuna study to sample and evaluate the model.

    :param trial: The optuna trial that defines how the next sample is derived
    :return: accuracy of the model over the validation data
    """

    # hyperparameters to optimize
    dropout_rate = trial.suggest_float("dropout",0.0,0.5)
    n_cnn_layers = trial.suggest_int("num_cnn_layers",2,4)
    kernel_size = trial.suggest_int('kernel_size',3,10)
    stride = 1 #trial.suggest_int('stride',1,1)
    channel_growth = trial.suggest_int('channel_growth',0,4)
    bottle_neck = trial.suggest_int('bottle_neck', 16, 128)
    class_layer_1 = trial.suggest_int('class_layer_1',128,1028)
    class_layer_2 = trial.suggest_int('class_layer_2', 128, 1028)
    batch_size = trial.suggest_int('batch_size',16,1028)

    h_params = {'dropout_rate':dropout_rate,
                'n_cnn_layers':n_cnn_layers,
                'kernel_size':kernel_size,
                'stride':stride,
                'channel_growth':channel_growth,
                'bottle_neck': bottle_neck,
                'class_layer_1':class_layer_1,
                'class_layer_2':class_layer_2,
                'trial_num':trial.number,
                'study_name':trial.study.study_name}

    compress_channel = True
    x_train, y_train, x_valid, y_valid, x_test, y_test, n_classes = load_data(compress_channel)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = TensorDataset(torch.Tensor(x_valid).to(device), torch.Tensor(y_valid).to(device))
    valid_dataloader = DataLoader(valid_dataset)

    test_dataset = TensorDataset(torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device))
    test_dataloader = DataLoader(test_dataset)

    n_channels = x_train.shape[1]
    cnn = Model.CAE(n_classes,n_channels,h_params)
    acc = cnn.train_cnn(100, train_dataloader, valid_dataloader, test_dataloader)
    return acc

def optimize_network():

    '''
    0 = Vanilla CNN
    1 = Convolutional Autoencoder
    '''
    case_to_run = 1


    # do the hyperparameter optimzation
    if case_to_run == 0:
        study = optuna.create_study(storage='sqlite:///db.sqlite3Test',study_name='VanillaCnnCompressed',direction='maximize',load_if_exists=True)
        study.optimize(objective_Vanilla_Cnn,n_trials=50)
    elif case_to_run == 1:
        study = optuna.create_study(storage='sqlite:///db.sqlite3Test', study_name='CAECompressed', direction='maximize',
                                    load_if_exists=True)
        study.optimize(objective_CAE, n_trials=50)


if __name__ == '__main__':

    optimize_network()