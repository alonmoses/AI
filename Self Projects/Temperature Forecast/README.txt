Hi,

This is my first AI project of predicting the Temperature
using a data stored in the 'Weather Stats.csv' table located in this folder.

The program performs 1000 epochs on the training data.

The NN used for this project is structured in the following two ways-
** Both NNs last layers are constructed the same way- 3 Linear layers, activated by a ReLu function.

This project introduce two different aproaches of predicting the temperature-

1. 'Predict Temperature.ipynb'-

    a. Structure of the NN-
    TabularWeatherModel(
    (embeds): ModuleList((0): Embedding(8, 4) (1): Embedding(2, 1))
    (embeds_drop): Dropout(p=0.4)
    (bn_cont): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (layers): Sequential(
    (0): Linear(in_features=10, out_features=200, bias=True)
    (1): ReLU(inplace)
    (2): Linear(in_features=200, out_features=100, bias=True)
    (3): ReLU(inplace)
    (4): Linear(in_features=100, out_features=50, bias=True)
    (5): ReLU(inplace)
    (6): Linear(in_features=50, out_features=1, bias=True)
    )
    )
    
    b. Changing all rows contains missing data to zero.
       the temperature cells which are missing are changed to the mean temperature of all the
       other temperatures in the same hour.


2. 'Predict Temperature- Better.ipynb'-

    a. Structure of the NN-
    TabularWeatherModel(
    (layers): Sequential(
    (0): Linear(in_features=5, out_features=200, bias=True)
    (1): ReLU(inplace)
    (2): Linear(in_features=200, out_features=100, bias=True)
    (3): ReLU(inplace)
    (4): Linear(in_features=100, out_features=50, bias=True)
    (5): ReLU(inplace)
    (6): Linear(in_features=50, out_features=1, bias=True)
    )
    )
    
    b. Deleting each row which contains a cell with missing data.


RESULTS-

1. NN Performance on the Data-
loss on training set(MSE)- 0.08800764
Average difference between predicted and actual tempertures in test set(in Celsius)- 0.532826832930247

2. NN Performance on the Data-
loss on training set(MSE)- 0.09727259
Average difference between predicted and actual tempertures in test set(in Celsius)- 0.2540586479504903


Imporant NOTE-
- I tried also to the second approach (2.b.) on te first NN structure (1.a.) and got medium results-
  loss on training set(MSE)- 0.07487297
  Average difference between predicted and actual tempertures in test set(in Celsius)- 0.520114643573761
  
  In a first look it can be seen that the training set loss is much lower than the two other approaches,
  although a high difference on the test set is achieved.
  This can point on an overfiting in this case. 
