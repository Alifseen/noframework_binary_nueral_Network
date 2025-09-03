# Building Blocks of a Deep Binary Classification Network
A customizable binary classification neural network implementation built from scratch using only NumPy and Python's built-in math functions. This project demonstrates the fundamental concepts of deep learning 

### 🎯 Project Overview
This is my first project on my journey into AI and ML. By building a neural network from the ground up, I've gained deep insights into:
- Matrix multiplications and their role in neural networks
- Chain rule derivatives and backpropagation
- Parameter initialization strategies
- Optimization techniques and their mathematical foundations

### ✨ Key Features
#### Parameter Initialization
I explored various types of parameter initialization techniques. The three initialization methods available via the `init` argument are:
- random: Basic random initialization
- he: He initialization (optimal for ReLU activations)
- xavier: Xavier initialization (optimal for tanh activations)
The initialization funcitons accept the following format:
```layers_dimensions = (Input Layer shape, hidden layer 1 neurons,...., hidden layer n neurons, 1))```

#### Activation Functions
These are 'relu' and 'tanh' hidden layer activations, I initally only had deployed relu activation for hidden layer, but soon started to see dying relu problem with narrow but deep neural networks so I added tanh activation as well. 
The `hidden_activation` can be called as arguments "relu", and "tanh", and `output_activation` as "sigmoid".

#### The forward and Backward Propagation
This is where the magic happens and bulk of the helper functions exists. The forward propagation funcitons for different actitvations as well the backward propagation helper functions include linear activation, relu and sigmoid activation, linear forward activation pass, relu, tanh and sigmoid derivative functions, linear backward activation pass, log_loss cost function, and more... these were created with flat caches (except dropout, more on that later). The order in which cache is store is:
```cache order = (Z1, A1, W1, b1, Z2, A2, W2, b2......ZL, AL, WL, bL)```

#### Optimization Algorithms
I also built in support for various optimization techniques for descent ranging from normal gradient descent, to momentum and adam with adjustable beta values. By defining `optimizer=` as either "momentum" or "adam" we can change the optimation. Then these can be finetuned using `momentum_beta, adam_beta1, adam_beta2,  epsilon` arguments.

#### Learning Rate Scheduling
I also built continious decay and step decay functionality for adjusting learning rate as the model diverges, so even a larger learning rate can be used at the start. Set `decay` to either 'update_lr' for continious or 'schedule_lr_decay' for step decay, and adjust the rate using `decay_rate`

#### Mini-Batch processing
Minibatching has also been built in into the building blocks for improved efficiency and stability and can be activated by simply mentioning the size in `mini_batch_size` argument.

#### Regularization Techniques
To avoid overfitting, I built L2 regularization functionality, as well dropout functionality to the building blocks. The dropout cache is not flat and is designed to carry each layers activation's mask as well. So the forward prop and back prop for dropout are done using different helper functions. To use either L2 or Dropout (both can not be used together in this implementation) simply set a value between 0 and 1 to `lambd` for L2 or to `keep_probs` for dropout.

#### Debugging and Monitoring Tools
Finally, I built troubleshooting code which helped me build effective neural networks. This includes seeding for randomized variables, printing and plotting costs, and checking gradient during optimation. Although gradient checking is done using the final parameters after optimnization, it helps to see the difference. Arguments `print_cost, check_gradient` are boolean, while `seed` accepts an integer.

### Usage
Although there is a model function `model()` defined with all the customization features adjustable using argumetns, the building blocks are designed in such a way that any customized, usage spefic model can be built as well. 
```
# Define network architecture
layers_dims = [input_features, 64, 32, 1]

# Train the model
parameters, gradients = model(
    X_train, y_train, 
    layers_dims,
    hidden_activation="relu",
    init="he",
    optimizer="adam",
    learning_rate=0.001,
    num_epochs=1000,
    mini_batch_size=64,
    keep_probs=0.8,
    print_cost=True,
    check_gradient=True
)
```

### Ending Notes
I have tested this network multiple binary classification tasks, including sythetic data, image classification and out of the box datasets such as those from SKlearn library. All of which I tested not just with model accuracy but also using the gradient checking technique, comparing analytical gradient with numerical gradient.

In future I plan to extend these building blocks to support multiclass classification and regression tasks as well.