# Data-based drag model for multiphase flows
```diff
+**If you use the code contained in this repo for your publications, we kindly ask you to cite the following papers:**
```
- **D'Alessio G., Sundaresan S., Mueller M.E. (2022). Automated and efficient local adaptive regression for principal component-based reduced-order modeling of turbulent reacting flows. Proceedings of the Combustion Institute 39.**
- **Jiang, Y., Chen, X., Kolehmainen, J., Kevrekidis, I. G., Ozel, A., & Sundaresan, S. (2021). Development of data-driven filtered drag model for industrial-scale fluidized beds. Chemical Engineering Science, 230, 116235.**

The aim of this code is to include Artificial Neural Network-based models in CFD simulations of gas-solid systems.
In particular, it is possible to employ this code and the associated trained Artificial Neural Network (ANN) for:

1) Euler-Euler simulations
2) Euler-Langrange simulations

With an Euler-Euler approach, the two phases are seen as interpenetrating continua. The solver: twoPhaseEulerFOAM in OpenFOAM-3.x is employed for this simulation. 
The Euler-Language approach, instead, solves the continuous phase on eulerian grids and tracks each particle to solve their behavior using Newtons equations of motion.
This is accomplished by means of the CFDEM software.

In the folder "code_openfoam", it is possible to find the files to be included in OpenFOAM to read the network and get the prediction.
In the folder "trained_network", it is possible to find a trained ANN, as well as additional files required to normalize the simulation data (i.e., the mean
and the standard deviation of the training data). 
The file "revisedNNmodel_TFM_DragCorrection.pdf" contains detailed information on how the approach works and the required files.

The network was trained in Keras, and it was adapted to OpenFOAM format by means of keras2cpp (https://github.com/pplonski/keras2cpp).
The number of layers and neurons for the ANN was found by means of the Bayesian optimization, a tool to construct a probabilistic model to converge to 
the best combination of hyperparameters, given a design space in input. Early stopping was used to prevent the network from overfitting.

Additional information on the approach can be found in the following papers:

- Jiang, Y., Chen, X., Kolehmainen, J., Kevrekidis, I. G., Ozel, A., & Sundaresan, S. (2021). Development of data-driven filtered drag model for industrial-scale fluidized beds. Chemical Engineering Science, 230, 116235.
- Jiang, Y., Kolehmainen, J., Gu, Y., Kevrekidis, Y. G., Ozel, A., & Sundaresan, S. (2019). Neural-network-based filtered drag model for gas-particle flows. Powder Technology, 346, 403-413.
```diff
!# ******* UPDATE 2 - September 2022 *******
```
In the folder 'code_optimization_ANN' I have included a script to train the ANN by means of Bayesian optimization (based on skopt). The script can work with the sample data I have included from our training matrix (see update 1 for additional information).
IMPORTANT:
For the script to work, the following dependencies must be satisfied:

i)   scikit-optimize: it is needed for the optimization part (please refer to: https://scikit-optimize.github.io/stable/); 

ii)  OpenMORe: it is needed for data processing (please refer to: (https://github.com/gdalessi/OpenMORe);

iii) LaTeX: it is needed for plots' labels (please refer to: https://www.latex-project.org);

iv)  Matplotlib: it is needed for plotting (please refer to: https://matplotlib.org).
```diff
!# ******* UPDATE 1 - July 2022 *******
```
In the folder “/test/test_ANN”, I put two additional  files: “sampleX.txt" and “sampleY.txt". These two files correspond to a small batch (2,000 samples, chosen randomly) of the training input and the output matrix, respectively, which can be used to test the network implementation and accuracy in the CFD solver. 
In addition, I have coded a small Python script to test "offline" the trained architecture using sampleX and sampleY, and I have put it in the same folder (it is the file “testNet.py”). Finally, “prediction_samples.png” is a parity plot to assess the accuracy of the network when sampleX is used as an input. I put this figure as a reference, as theoretically the very same result should be obtained using your own testing framework (as the ANN, the preprocessing factors, and the input/output matrix you will be using are the same loaded/used by the Python script). 

On a separate note, in: “/test/noGradP” you will find another DFnet file (as well as two other files with different centering and scaling factors) corresponding to a network trained by means of the same input data, but removing the pressure gradient from the input variables (i.e., the ANN model is working with four input variables, instead of the standard five described in the pdf I put on GitHub). From a priori results, we could see that gradP has a strong impact on the network accuracy, as the prediction error for this new network was much higher than the one we got with the five-dimensional input case. Despite the architectures of the two networks are different, they were obtained by means the same Bayesian-based automated procedure and using the very same initial conditions and parameters (i.e., same design space and number of iterations). Thus, it can be considered a coherent comparison from a modeling perspective.
