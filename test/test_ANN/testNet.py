import numpy as np
#import OpenMORe.model_order_reduction as model_order_reduction
#from OpenMORe.utilities import *
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


import tensorflow as tf

from tensorflow.keras.models import load_model
#X = np.load(os.path.abspath(os.path.join(__file__ ,"../wendysData/data/")) + "/" + "X.npy")
#Y = np.load(os.path.abspath(os.path.join(__file__ ,"../wendysData/data/")) + "/" + "Y.npy")
#Y = np.reshape(Y, (Y.shape[0],1))

Xsamp = np.genfromtxt("sampleX.txt")
Ysamp = np.genfromtxt("sampleY.txt")

mu = np.genfromtxt("DF_mean.csv", delimiter=',')
sigma = np.genfromtxt("DF_std.csv", delimiter=',')

'''
number_of_rows = X.shape[0]
random_indices = np.random.choice(number_of_rows, 2000)

Xsamp = X[random_indices,:]
Ysamp = Y[random_indices,:]

np.savetxt("sampleX.txt", Xsamp)
np.savetxt("sampleY.txt", Ysamp)
'''

model = load_model('DFmodel.hdf5')

#mu = center(X, "mean")
#sigma = scale(X, "auto")
X_tilde = (Xsamp-mu)/(sigma +1E-16)

prediction_sample = model.predict(X_tilde)

matplotlib.rcParams.update({'font.size' : 16, 'text.usetex' : True})
a = plt.axes(aspect='equal')
im = plt.scatter(Ysamp[:,], prediction_sample[:,], 1, c='dodgerblue', label= "$Predictions$")
plt.xlabel('$Original\ output$')
plt.ylabel('$Predicted\ output$')
lims = [np.min(Ysamp[:,]), np.max(Ysamp[:,])]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, 'k', label= "$True\ values$")
plt.ticklabel_format(style='sci')
lgnd = plt.legend(loc="best")
lgnd.legendHandles[0]._sizes = [45]
lgnd.legendHandles[1]._sizes = [45]
plt.savefig('prediction_samples.png', dpi=300)
plt.show()
plt.close()
