<div style="font-family: monospace; list-style-type: none; padding-left: 0; line-height: 1.5;">
 
# Neural Network Nonsense
A neural network framework created from scratch in C# implementing automatic differentiation, backpropagation, and customizable architectures for neural networks. The name is a relic of the project's original intended
 purpose of experimenting with neural networks.

## Key Features
- Deep Q-Network (DQN) training capabilities
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
#### Code Used for Testing (Addition Example):
```
Tensor a = new([3])
{
    Data = [1.0, 2.0, 3.0]
};
Tensor b = new([3])
{
    Data = [4.0, 5.0, 6.0]
};
Tensor[] inputs = [a, b];

Func<Tensor[], Tensor> testOp = inputs =>
{
    return inputs[0] + inputs[1];
};
Func<Tensor[], double> loss = inputs =>
{
    var result = inputs[0] + inputs[1];
    return Tensor.Mean(result)[0];
};

MathUtils.GradientTest(inputs, testOp, loss);
```
```
public static void GradientTest(Tensor[] inputs, Func<Tensor[], Tensor> testOp, Func<Tensor[], double> loss)
{
    var result = testOp(inputs);
    var mean = Tensor.Mean(result);
    mean.Backward();

    for (int i = 0; i < inputs[0].ElementCount; i++)
    {
        var (aNumerical, bNumerical) = NumericalGradient(inputs, i, loss);
        double aAnalytical = inputs[0].Grad[i];
        double bAnalytical = inputs[1].Grad[i];
        double relError = Math.Abs(aNumerical - aAnalytical) / (Math.Abs(aNumerical) + 1e-8);
        Console.WriteLine($"a[{i}]: numerical = {aNumerical}, analytical = {aAnalytical}, relError = {relError}");
        relError = Math.Abs(bNumerical - bAnalytical) / (Math.Abs(bNumerical) + 1e-8);
        Console.WriteLine($"b[{i}]: numerical = {bNumerical}, analytical = {bAnalytical}, relError = {relError}");
    }
}

static (double aNumerical, double bNumerical) NumericalGradient(Tensor[] inputs, int i, Func<Tensor[], double> loss)
{
    double eps = 1e-8;
    inputs[0][i] += eps;
    double lossPlus = loss(inputs);
    inputs[0][i] -= 2 * eps;
    double lossMinus = loss(inputs);
    inputs[0][i] += eps;
    double aNumerical = (lossPlus - lossMinus) / (2 * eps);

    inputs[1][i] += eps;
    lossPlus = loss(inputs);
    inputs[1][i] -= 2 * eps;
    lossMinus = loss(inputs);
    inputs[1][i] += eps;
    double bNumerical = (lossPlus - lossMinus) / (2 * eps);

    return (aNumerical, bNumerical);
}
```

### Supervised Learning Convergence Test (XOR Classification)
#### Specifications:
Architecture: 4 -> 1 (Sigmoid -> Linear)\
Optimizer: Adam\
Learning Rate: 0.01\
Target MSE: < 0.01
| Test # | Epochs Required |
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
#### Specifications:
Architecture: 256 -> 256 -> 128 -> 9 (Leaky ReLU -> Leaky ReLU -> Leaky ReLU -> Linear)\
Loss Function: Pseudo-Huber\
Optimizer: Adam\
Learning Rate: 0.001\
Games per Performance Test: 5000\
Opponent for Performance Tests: Randomly-Acting\
Total Time Required for Training and Evaluation: 3:15.919
| Training Episodes | Win Rate | Tie Rate | Win + Tie Rate |
|-------------------|----------|----------|----------------|
| 200* | 42.06% | 30.32% | 72.38% |
| 400* | 42.18% | 30.14% | 72.32% |
| 600* | 92.04% | 6.08% | 98.12% |
| 800 | 93.94% | 5.20% | 99.14% |
| 1000 | 93.16% | 5.50% | 98.66% |
| 1200 | 94.20% | 5.14% | 99.34% |
| 1400 | 92.96% | 6.18% | 99.14% |
| 1600 | 95.60% | 4.26% | 99.86% |
| 1800 | 95.24% | 4.54% | 99.78% |
| 2000 | 94.60% | 5.40% | 100.00% |

*Note that the majority of the first 600 episodes were used to collect initial experiences without training
#### Code Used for Testing:
##### Training Hyperparameters:
```
NNN.Environment env = new TicTacToe();
double exploration = 1.0;
double explorationDecay = 0.9995;
double minExploration = 0.01;
int trainEvery = 1;
double discount = 0.99;
Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 128;
int agentBufferSize = 2;
int opponentCopyRate = 600;
int minRandomOpponentEpisodes = 600;
double tau = 0.01;
double maxGradNorm = 1.0;
int minExperiences = 2000;
int episodeMemorySize = 100;
int testEpisodes = 5000;
```
##### Performance Evaluation (in Tic-Tac-Toe Environment):
```
public override void TestTrainingProgress(Model agent, int testEpisodes)
{
    int wins = 0;
    int ties = 0;
    for (int e = 0; e < testEpisodes; e++)
    {
        var (won, tied) = PlayRandom(agent);
        if (won) wins++;
        else if (tied) ties++;
    }

    double winPercent = ((double)wins / testEpisodes) * 100.0;
    double tiePercent = ((double)ties / testEpisodes) * 100.0;
    Console.WriteLine($"Win percentage vs randomly-acting opponent: {winPercent:F2}");
    Console.WriteLine($"Tie percentage vs randomly-acting opponent: {tiePercent:F2}");
    Console.WriteLine($"Win + tie percentage vs randomly-acting opponent: {(winPercent + tiePercent):F2}");
}

public (bool won, bool tied) PlayRandom(Model agent)
{
    Reset();

    bool agentTurn = Random.Next(2) == 1;
    while (!CheckWin() && !BoardFilled())
    {
        int action = agentTurn ? GetAgentAction(agent) : PickRandomAction();

        State[action] = State[9] == 1.0 ? 1.0 : -1.0;

        if (agentTurn && CheckWin()) return (true, false);

        State[9] *= -1.0;
        agentTurn = !agentTurn;
    }

    return (false, !CheckWin() && BoardFilled());
}
```

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

## Organization
NNN\
├── ModelTrainer (Script for training neural networks using the NNN framework)\
│\
└── Components\
&emsp;&emsp;├── Activations (Activation function classes)\
&emsp;&emsp;│&emsp;&ensp;├── Activation (Base class)\
&emsp;&emsp;│&emsp;&ensp;├── LeakyReLU\
&emsp;&emsp;│&emsp;&ensp;├── Linear\
&emsp;&emsp;│&emsp;&ensp;├── ReLU\
&emsp;&emsp;│&emsp;&ensp;├── Sigmoid\
&emsp;&emsp;│&emsp;&ensp;└── Tanh\
&emsp;&emsp;│\
&emsp;&emsp;├── Autodiff (Automatic differentation and tensor logic - implemented via a partial Tensor class)\
&emsp;&emsp;│&emsp;&ensp;├── TensorActivations (Tensor implementations of activation functions)\
&emsp;&emsp;│&emsp;&ensp;├── TensorGraph (Functions for handling computation graph)\
&emsp;&emsp;│&emsp;&ensp;├── TensorIndexing (Functions for indexing tensors)\
&emsp;&emsp;│&emsp;&ensp;├── TensorInitializations (Functions for initializing tensors)\
&emsp;&emsp;│&emsp;&ensp;├── TensorOperations (Tensor implementations of mathematical operations)\
&emsp;&emsp;│&emsp;&ensp;├── TensorProperties (All tensor class properties)\
&emsp;&emsp;│&emsp;&ensp;└── TensorUtilities (Various utility functions for tensors)\
&emsp;&emsp;│\
&emsp;&emsp;├── Buffers\
&emsp;&emsp;│&emsp;&ensp;├── FIFOBuffer (Standard First-In First-Out buffer)\
&emsp;&emsp;│&emsp;&ensp;├── ReplayBuffer (PER buffer for DQN experience replay)\
&emsp;&emsp;│&emsp;&ensp;└── SumTree (Standard sum tree data structure)\
&emsp;&emsp;│\
&emsp;&emsp;├── Costs\
&emsp;&emsp;│&emsp;&ensp;├── Cost (Base class)\
&emsp;&emsp;│&emsp;&ensp;├── Huber\
&emsp;&emsp;│&emsp;&ensp;└── MSE\
&emsp;&emsp;│\
&emsp;&emsp;├── Environments (DQN)\
&emsp;&emsp;│&emsp;&ensp;├── Environment (Base class)\
&emsp;&emsp;│&emsp;&ensp;├── MovementGrid2D\
&emsp;&emsp;│&emsp;&ensp;├── Snake\
&emsp;&emsp;│&emsp;&ensp;└── TicTacToe\
&emsp;&emsp;│\
&emsp;&emsp;├── Episodes (Data structures for DQN experience storage)\
&emsp;&emsp;│&emsp;&ensp;├── Episode (Data structure for a full DQN training episode)\
&emsp;&emsp;│&emsp;&ensp;└── Experience (Data structure for a single DQN training experience)\
&emsp;&emsp;│\
&emsp;&emsp;├── Models (Neural network functionality)\
&emsp;&emsp;│&emsp;&ensp;├── Layers\
&emsp;&emsp;│&emsp;&ensp;│&emsp;&ensp;├── Conv (Convolutional)\
&emsp;&emsp;│&emsp;&ensp;│&emsp;&ensp;├── Dense\
&emsp;&emsp;│&emsp;&ensp;│&emsp;&ensp;└── Layer (Base class)\
&emsp;&emsp;│&emsp;&ensp;│\
&emsp;&emsp;│&emsp;&ensp;└── Model (Full neural network container class)\
&emsp;&emsp;│\
&emsp;&emsp;├── Optimizers\
&emsp;&emsp;│&emsp;&ensp;├── Adam\
&emsp;&emsp;│&emsp;&ensp;├── Optimizer (Base class)\
&emsp;&emsp;│&emsp;&ensp;└── SGD\
&emsp;&emsp;│\
&emsp;&emsp;├── Trainers\
&emsp;&emsp;│&emsp;&ensp;├── DQNTrainer\
&emsp;&emsp;│&emsp;&ensp;└── Trainer (Standard supervised training)\
&emsp;&emsp;│\
&emsp;&emsp;└── Utilities\
&emsp;&emsp;&emsp;&emsp;├── SaveSystem\
&emsp;&emsp;&emsp;&emsp;│&emsp;&ensp;├── LayerData (JSON serializable class storing data for a single layer)\
&emsp;&emsp;&emsp;&emsp;│&emsp;&ensp;├── ModelData (JSON serializable class storing data for a single neural network)\
&emsp;&emsp;&emsp;&emsp;│&emsp;&ensp;└── Saver (Static class for handling model saving/loading)\
&emsp;&emsp;&emsp;&emsp;│\
&emsp;&emsp;&emsp;&emsp;├── DemoHandler (Static class for handling demonstrations)\
&emsp;&emsp;&emsp;&emsp;├── MathUtils\
&emsp;&emsp;&emsp;&emsp;└── UIUtils

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
</div>
