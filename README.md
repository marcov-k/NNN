# Neural Network Nonsense
A neural network framework created from scratch in C# implementing automatic differentiation, backpropagation, and customizable architectures for neural networks.

## Key Features
- Deep Q-Network (DQN) training capabilties
- Prioritized experience replay (PER) buffer implementation
- Reverse-mode automatic differentiation
- Dynamic computation graph reused across forward passes
- Dense and convolutional layers
- Standard activation functions (Sigmoid, Tanh, ReLU, etc.)
- Standard loss functions (MSE, pseudo-Huber Loss)
- Standard optimizers (SGD, Adam)
- Performance optimizations via SIMD vectorization and parallelization

## Results
### Gradient Correctness Tests (Autograd Verification)
| Operation | Max Relative Error |
|-----------|--------------------|
| Addition | 8.27e-8 |
| Multiplication | 6.08e-9 |
| Matrix Multiplication | 7.93e-7 |
| Pow(a, 2.0) | 1.08e-7 |
| Tanh | 4.34e-7 |

### Supervised Learning Convergence Test (XOR Classification)
Specifications:\
Architecture: 4 -> 1 (Sigmoid -> Linear)\
Optimizer: Adam\
Learning Rate: 0.01\
Target MSE: < 0.01
| Test # | Epochs Taken |
|--------|--------------|
| 1 | 788 |
| 2 | 1009 |
| 3 | 2428 |
| 4 | 309 |
| 5 | 843 |
| 6 | 841 |
| 7 | 899 |
| 8 | 600 |
| 9 | 570 |
| 10 | 497 |
| Average | 878.4 |
#### Code Used for Testing:
```
Tensor inputFormat = new([1, 2], false);
Tensor inputs = new([4, 2], false)
{
    Data = [0, 0, 1, 0, 0, 1, 1, 1]
};
Tensor targets = new([4, 1], false)
{
    Data = [0, 1, 1, 0]
};
Model testModel = new([
    new Dense(4, new Sigmoid()),
    new Dense(1, new Linear())
    ], inputFormat);
Cost testCost = new MSE();
Optimizer testOptimizer = new Adam(0.01);
Trainer testTrainer = new(testModel, testOptimizer, testCost);

int maxEpochs = 10000;
int epochs = 0;
while (epochs < maxEpochs && testCost.CalculateCost(testModel.Predict(inputs), targets)[0] >= 0.01)
{
    testTrainer.Train(inputs, targets, 1);
    epochs++;
}

Console.WriteLine($"Reached MSE below 0.01 in {epochs} epochs");
Console.WriteLine("Press any key to close...");
Console.ReadKey();
```

### Tic-Tac-Toe (DQN + Self-Play)
Architecture Used:\
128 -> 128 -> 64 -> 9 (Leaky ReLU -> Leaky ReLU -> Leaky ReLU -> Linear)
| Metric | Value |
|--------|-------|
| Training Games | 50,000|
| Training Time | 7:44.617 |
| Test Games vs Randomly-Acting Opponent | 10,000 |
| Win Rate vs Randomly-Acting Opponent | 91.15% |

## Motivation
I originally intended for this project to simply be my experimentation with implementing the systems described in Seth Weidman's _Deep Learning from Scratch_. However, after seeing my basic neural networks successfully
 train using the Boston housing dataset highlighted in Weidman's book, I became increasingly interested in creating a framework which could support Deep Q-Network training (DQN) for complex environments. After a number
 of failed attempts using the framework I had derived from the examples in _Deep Learning from Scratch_, I came to the realization that in order to support more complex networks while also maintaining sufficiently high
 performance to train such networks on my own personal computer, I would have to completely rewrite the entire framework. At this point I discovered the automatic differentiation algorithms used by libraries such as
 PyTorch, and decided that my new framework would follow a similar approach. Additionally, in order to gain as thorough of an understanding of the mathematics and logic behind these systems, I opted to avoid using
 any features not provided in the standard C# library, which I also believed would provide me with valuable experience in designing full-scale frameworks from scratch. Now, many months after I first began experimenting
 with the Neural Network Nonsense project, I can proudly say that those initial efforts have grown into something far larger than I could have ever anticipated.

## Example
```
Model network = new Model([
  new Dense(16, new ReLU()),
  new Dense(1, new Linear())
  ], InputFormat);

Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();

Trainer trainer = new Trainer(network, optimizer, cost);
trainer.Train(x, y, epochs: 1000);
```

## Architecture
### Tensor
- Stores values and gradients
- Stores parent and result tensors (for graph reuse)
- Manages autograd graph
- Provides all tensor operations (matrix multiplication, element-wise operations, etc.)
### Autograd Engine
- Topological sorting
- Reverse gradient propagation
### Layers
- Dense
- Convolutional
- Store parameters and activations
### Activation Functions
- Sigmoid
- Hyperbolic tangent (Tanh)
- Rectified Linear Unit (ReLU)
- Leaky Rectified Linear Unit (Leaky ReLU)
- Linear
### Cost Functions
- Mean Squared Error (MSE)
- Pseudo-Huber Loss
### Model
- Stores forward order of layers
- Provides access to all parameters
### Trainers
- Basic supervised trainer
- Deep Q-Network (DQN) trainer
### Saver
- Serializes trained models using JSON
- Deserializes and reconstructs models from JSON

## Automatic Differentiation (via partial derivatives)
### Example
z = x * y\
w = z + x

Partial Derivatives:\
dw/dz = 1\
dw/dx = 1\
dz/dx = y\
dz/dy = x
### Autograd Engine
The framework's equivalent of an autograd engine relies on functionality built directly into the tensor objects, rather than a separate distinct system. Each mathematical operation (element-wise addition, matrix
 multiplication, sigmoid, mean, etc.) is provided as a static function or operator in the tensor class. These functions handle the result calculation while also defining a "backward" function which calculates the
 gradients of each input using their respective partial derivative formulas multiplied by the gradient of the result to account for the chain rule. Each tensor also exposes a Backward() method which performs a topological
 of the computation graph before propagating gradients in reverse-order.

## Implementation Details
### Reverse-mode Autograd
- Computation graph dynamically constructed and updated during each forward pass
- Inference mode - avoids full graph construction on non-training forward passses
- Topological sorting before backward pass
- Gradient accumulation/chain rule application
### Memory Management
- Persistent autograd graph "node" tensor allocations
- Use of persistent buffers such as an FIFO buffer for later DQN training episode replay
- Use of tensors as the fundamental data type in all operations
- Use of "stackalloc" to reduce garbage collector overhead
- Underlying tensor data represented in linear arrays
### Performance
- Significant reduction in memory allocations through autograd graph reuse
- SIMD vectorization of most tensor operations
- CPU parallelizatoin of matrix multiplication and convolution above defined parameter thresholds
- Reuse of allocated memory where possible

## Future Work
- GPU acceleration
- Recursive neural networks
- More DQN training environments
- Proximal policy optimization (PPO)
