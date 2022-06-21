Data-based drag model for multiphase flows

The aim of this code is to include Artificial Neural Network-based models in CFD simulations of gas-solid systems.
In particular, it is possible to employ this code and the associated trained Artificial Neural Network (ANN) for:

1) Euler-Euler simulations
2) Euler-Langrange simulations

The Euler-Euler approach treates two phases as interpenetrating continua, twoPhaseEulerFOAM solver in OpenFOAM-3.x is employed for this simulation. 
The Euler-Language approach, instead, solves the continuous phase on eulerian grids and tracks each particle and solve their behavior using Newtons equations of motion.
This is accomplished by means of CFDEM software.

In the folder "code_openfoam", it is possible to find the files to be included in OpenFOAM.
In the folder "trained_network", it is possible to find the trained ANN, as well as additional files required to normalize the simulation data (i.e., the mean
and the standard deviation of the training data). 
The file "revisedNNmodel_TFM_DragCorrection.pdf" contains detailed information on how the approach works and the required files.

The network was trained in Keras, and it was adapted to OpenFOAM format by means of keras2cpp (https://github.com/pplonski/keras2cpp).
Additional information on the approach works can be found in the following papers:

- Jiang, Y., Chen, X., Kolehmainen, J., Kevrekidis, I. G., Ozel, A., & Sundaresan, S. (2021). Development of data-driven filtered drag model for industrial-scale fluidized beds. Chemical Engineering Science, 230, 116235.
- Jiang, Y., Kolehmainen, J., Gu, Y., Kevrekidis, Y. G., Ozel, A., & Sundaresan, S. (2019). Neural-network-based filtered drag model for gas-particle flows. Powder Technology, 346, 403-413.
