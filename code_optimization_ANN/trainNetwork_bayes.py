'''
MODULE: trainNetwork_bayes.py

@Author:
    G. D'Alessio [1]
    [1]: Department of Mechanical and Aerospace Engineering, Princeton University, Princeton, USA
@Contacts:
    gd6644@princeton.edu

@Description:
    In this module, Bayesian optimization is used to find the optimal settings for an Artificial Neural Network.
    The design space is given by: i) Number of layers, ii) number of neurons, iii) activation function, iv) LR.

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: gd6644@princeton.edu

'''


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Activation
import os
import os.path
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *
import tensorflow as tf

#################################################################################
# Dictionary with the instruction for the algorithm:                            #
settings ={                                                                     #
    ##### DATA PREPROCESSING SETTINGS ######                                    #
    # Centering and scaling options (string/string)                             #
    "centering_method"          : "mean",                                       #
    "scaling_method"            : "auto",                                       #
    # Percentage of input data to be used for training (int)                    #
    "training_ratio"            : 70, #%                                        #
                                                                                #
    ##### ANN AND OPTIMIZER SETTINGS #####                                      #
    # Number of epochs for the ANN (int)                                        #
    "network_epochs"            : 500,                                          #
    # Number of iterations for the optimizer (int)                              #
    "iterations_optimizer"      : 30,                                           #
    # Acquisition function to be utilized (string)                              #
    "acquisitionFunction"       : "EI"                                          #
    # Settings for the first iteration of the optimizer (int/int/string/float)  #
    "initial_neurons"           : 16,                                           #
    "initial_layers"            : 1,                                            #
    "initial_activation"        : "relu",                                       #
    "initial_learningRate"      : 1e-4,                                         #
                                                                                #
    ##### DESIGN SPACE SETTINGS #####                                           #
    # Lower and upper bound for number of layers (int/int)                      #
    "layers_lowerBound"         : 1,                                            #
    "layers_upperBound"         : 25,                                           #
    # Lower and upper bound for number of neurons (int/int)                     #
    "neurons_lowerBound"        : 5,                                            #
    "neurons_upperBound"        : 512,                                          #
    # Lower and upper bound for learning rate (float/float)                     #
    "learning_lowerBound"       : 1e-6,                                         #
    "learning_upperBound"       : 1e-2,                                         #
                                                                                #
    ##### OTHER UTILITIES ##### (bool/bool/int)                                 #
    "plot_results"              : True,                                         #
    "save_model"                : True,                                         #
    # Early stop to avoid overfit:don't touch unless you know what you're doing!#
    "earlyStop_patience"        : 5,                                            #
}                                                                               #
################################################################################

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("I am working on GPU")
else:
    print("I am working on CPU")
for gpu in gpus:
    print("Name: {}".format(gpu.name))

# Load the data: X is the input matrix, Y is the output matrix (both should be in .txt format)
X = np.genfromtxt(os.path.abspath(os.path.join(__file__ ,"../../test/test_ANN")) + "/" + "sampleX.txt")
Y = np.genfromtxt(os.path.abspath(os.path.join(__file__ ,"../../test/test_ANN")) + "/" + "sampleY.txt")
Y = np.reshape(Y, (Y.shape[0],1))

# Preprocess the data with centering and scaling
mu = center(X, settings["centering_method"])
sigma = scale(X, settings["scaling_method"])
scaled_X = center_scale(X, mu, sigma)


# Initialize other quantitites and create a folder to store the trained model
path_best_model = '19_best_model.hdf5'
best_errorPrediction = 10000000000000000
newDirName =   "optimizeNetwork" + "_epochs=" + str(settings["iterations_optimizer"]) + "_evals=" + str(settings["network_epochs"]) + "scaling=" + str(settings["scaling_method"])
os.mkdir(newDirName)
os.chdir(newDirName)

# Write down the centering/scaling factors in a separate file to be used later on
import csv
f = open("mean.csv", 'w')
writer = csv.writer(f)
writer.writerow(mu)
f.close()
g = open("std.csv", 'w')
writer = csv.writer(g)
writer.writerow(sigma)
g.close()

# Define the design space with the values that were set in the dictionary above
numberLayers = Integer(low=int(settings["layers_lowerBound"]), high=int(settings["layers_upperBound"]), name='layers')
numberNeurons = Integer(low=int(settings["neurons_lowerBound"]), high=int(settings["neurons_upperBound"]), name='neurons')
dim_activation = Categorical(categories=['relu', 'elu', 'selu'], name='activation')
dim_learning_rate = Real(low=float(settings["learning_lowerBound"]), high=float(settings["learning_upperBound"]), prior='log-uniform', name='alpha')

# Initialize the optimizer with the values chosen above
dimensions = [numberLayers, numberNeurons, dim_activation, dim_learning_rate]
default_parameters = [int(settings["initial_layers"]), int(settings["initial_neurons"]), str(settings["initial_activation"]), float(settings["initial_learningRate"])]


def log_dir_name(layers, neurons, activation, alpha):
    '''
    This function creates a folder for each training.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    '''

    s = "./Logs/layers_{0}_nodes_{1}_activation_{2}_alpha_{3}/"
    log_dir = s.format(layers, neurons, activation, alpha)

    return log_dir

def create_model(layers, neurons, activation, alpha):
    '''
    This function creates the ANN model. It starts with splitting the dataset in
    train/test, and then it creates the architecture given the input variables.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    Output vars:
        model: model for the ANN.
        X_train, X_test: training and test samples from input matrix (80/20, respectively).
        y_train, y_test: training and test samples from output matrix (80/20, respectively).
    '''
    # Compute how much is to be used for test:
    # Tranining ratio (in settings) is in percentage
    test_ratio = 1-float(settings["training_ratio"])*0.01
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=test_ratio, random_state=42)
    model = Sequential()
    # Add all the hidden layers and activate each of them
    for i in range(layers+1):
        if i == 0:
            model.add(Dense(neurons, kernel_initializer='normal', input_dim = X_train.shape[-1]))
            model.add(Activation(str(activation)))
        model.add(Dense(neurons, kernel_initializer='normal'))
        model.add(Activation(str(activation)))

    # Add the output layer (no activation needed, since it is for prediction)
    model.add(Dense(Y.shape[1]))
    # Define the optimizer with the given initial LR
    optimizer = Adam(lr=float(alpha))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model, X_train, y_train, X_test, y_test


@use_named_args(dimensions=dimensions)
def fitness(layers, neurons, activation, alpha):
    '''
    This function trains the ANN model, and evaluates the error that corresponds
    to that particular architecture.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    Output vars:
        errorPrediction: error which is observed for the i-th training.
    '''
    try:
        model, X_train, y_train, X_test, y_test = create_model(layers=layers, neurons=neurons, activation=activation, alpha=alpha)
        batchS = int(X_train.shape[0]/100) +1
        # Create a folder to store the training
        log_dir = log_dir_name(layers, neurons, activation, alpha)
        callback_log = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=int(settings["earlyStop_patience"]), verbose=0, mode='min')
        history = model.fit(x= X_train, y= y_train, validation_data=(X_test,y_test), epochs=int(settings["network_epochs"]), batch_size=batchS, callbacks=[callback_log, earlyStopping])
        errorPrediction = history.history['mse'][-1]
        print("Regression error (MSE): {}".format(errorPrediction))
        # Save the model if it improves on the best performance.
        global best_errorPrediction

        # If the errorPrediction is improved, then this is the best model so far
        if errorPrediction < best_errorPrediction:
            model.save(path_best_model)
            best_errorPrediction = errorPrediction
            print("********* NEW BEST MODEL FOUND *********")
            print('Number of layers:', layers)
            print('Number of neurons:', neurons)
            print('Activation function:', activation)
            print('Learning rate:', alpha)
            print("******************")

        del model
        K.clear_session()

    except:
        print("I am in Except")
        exit()
        pass

    return errorPrediction

# This is the function of skopt to train the optimizer
fitness(x= default_parameters)
search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func=str(settings["acquisitionFunction"]), n_calls=int(settings["iterations_optimizer"]), x0=default_parameters)


##### PART 2: plotting the results and saving the model #####
if settings["plot_results"]:
    plot_convergence(search_result, y_scale="log")
    plt.savefig("Converge.png", dpi=400)
    #plt.show()
    plt.close()


print(search_result.x)
print(search_result.fun)


best_model= load_model(path_best_model)
opt_par = search_result.x


# use hyper-parameters from optimization
num_layers = opt_par[0]
num_nodes = opt_par[1]
opti_acti = opt_par[2]
opti_LR = opt_par[3]


text_file = open("best_training.txt", "wt")
neurons_number = text_file.write("The optimal number of neurons is equal to: {} \n".format(num_nodes))
layers_number = text_file.write("The optimal number of layers is equal to: {} \n".format(num_layers))
acti_opt = text_file.write("The optimal acquisition function is: {} \n".format(str(opti_acti)))
layers_number = text_file.write("The optimal LR is equal to: {} \n".format(opti_LR))
text_file.close()


pred_all = best_model.predict(scaled_X)
if settings["save_model"]:
    np.save("prediction_trainData.npy", pred_all)
    best_model.save("bestModel.hdf5")

    # save model structure
    model_json = best_model.to_json()
    with open('bayesModel.json', 'w', encoding = 'utf8') as json_file:
        json_file.write(model_json)


if settings["plot_results"]:
    for ii in range(Y.shape[1]):
        varToPlot = ii

        matplotlib.rcParams.update({'font.size' : 16, 'text.usetex' : True})
        a = plt.axes(aspect='equal')
        im = plt.scatter(Y[:,varToPlot], pred_all[:,varToPlot], 1, c='dodgerblue', label= "$Prediction$")
        plt.xlabel('$Original\ output$')
        plt.ylabel('$Predicted\ output$')
        lims = [np.min(Y[:,varToPlot]), np.max(Y[:,varToPlot])]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims, 'k', label= "$True\ values$")
        plt.ticklabel_format(style='sci')
        lgnd = plt.legend(loc="best")
        lgnd.legendHandles[0]._sizes = [45]
        lgnd.legendHandles[1]._sizes = [45]
        plt.savefig('prediction_var' + str(ii) + '.png', dpi=300)
        #plt.show()
        plt.close()
