# Simple Neural Network 

Learning about Neural Networks by implementing a simple one from "scratch". Based on [this tutorial](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6).

## Dependencies 
I'm using [Pipenv](https://pipenv.readthedocs.io/en/latest/) as a package/virtual-environment manager. Using that you should be able to run things with the following commands: 
```
pipenv install
pipenv shell 
python simple_neural_network.py 
```

## Structure 
This is a super simple 2-layer Neural Network (input layer, hidden layer, output layer, weights). The activation function is a [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), and the error function is a simple [Sum of Squares](https://hlab.stanford.edu/brian/error_sum_of_squares.html).