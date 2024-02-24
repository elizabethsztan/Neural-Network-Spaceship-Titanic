# Neural-Network-Spaceship-Titanic
Multi-layered neural network to predict outcomes for the Kaggle dataset, Spaceship Titanic (kaggle.com/competitions/spaceship-titanic/).
The neural network has been produced from scratch without the use of any external libraries (except Pytorch used in finding gradients of the loss). I have done for a deeper understanding of the mathematics of neural networks.

Aims:
- Predict whether passengers are to be transported or not from this dataset considering different parameters.

Method:
- Two layer neural network.
- Clean the data (using clean_data_functions.py).
- With a single layer neural network, find most important parameters to use in training (single_layer_NN.py).
- Train two layer neural network with the most important parameters (multi_layer_NN.py).
- Trains in batches to reduce overfitting.
- Check results on validation set before running model on test set.

About the neural network:
- Two layers and n hidden layers.
- First layer is a matrix of size number of parameters/columns X number of hidden layers.
- Second layer hence needs to be a matrix of size number of hidden layers X 1 to get output of size equal to number of parameters/columns.
- Constant term also added.
- Gradient descent to fit layers and decrease the loss.
 
Result:
- Obtained 0.75 accuracy on the Kaggle dataset.
- Trains very quickly (a few seconds).
   
Possible extensions
- Could change the columns used in the data and see how it performs
- Many columns were collapsed into one (VRDeck, Spa etc... was collapsed into Expenditure). Could see how model trains if these were not collapsed
- Could add more layers to capture higher order relationships in the data 
  
