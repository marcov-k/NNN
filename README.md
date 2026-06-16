# Neural Network Nonsense
A neural network framework created from scratch in C# implementing automatic differentiation, backpropagation, and customizable architectures for neural networks. The name is a relic of the project's original intended
 purpose of experimenting with neural networks.

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
Specifications:\
Architecture: 128 -> 128 -> 64 -> 9 (Leaky ReLU -> Leaky ReLU -> Leaky ReLU -> Linear)\
Loss Function: Pseudo-Huber\
Optimizer: Adam\
Learning Rate: 0.001\
Games per Win Rate Test: 5000
| Training Episodes | Win Rate vs Randomly-Acting Opponent |
|-------------------|--------------------------------------|
| 300 | 52.5% |
| 600 | 80.2% |
| 900 | 89.9% |
| 1200 | 92.8% |
| 1500 | 93.5% |
| 1800 | 91.6% |
| 2100 | 92.1% |
| 2400 | 92.8% |
| 2700 | 94.2% |
| 3000 | 93.8% |

Total time required for training and evaluation: 32.574 seconds
#### Code Used for Testing:
```
Model model;
NNN.Environment env = new TicTacToe();
double exploration = 1.0;
double explorationDecay = 0.9995;
double minExploration = 0.01;
int trainEvery = 4;
double discount = 0.99;
Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 128;
int agentBufferSize = 4;
int opponentCopyRate = 600;
int minRandomOpponentEpisodes = 600;
double tau = 0.01;
double maxGradNorm = 1.0;
int minExperiences = 2000;
int episodeMemorySize = 100;
DQNTrainer dqnTrainer;
FIFOBuffer<Episode> episodeBuffer = new(episodeMemorySize);

TestTicTacToeTraining();

void TestTicTacToeTraining()
{
    // Initialize model and trainer

    model = new([
        new Dense(128, new LeakyReLU()),
        new Dense(128, new LeakyReLU()),
        new Dense(64, new LeakyReLU()),
        new Dense(env.ActionCount, new Linear())
        ], env.StateFormat);
    dqnTrainer = new(
        agent: model,
        environment: env,
        optimizer: optimizer,
        cost: cost,
        trainEvery: trainEvery,
        discount: discount,
        exploration: exploration,
        explorationDecay: explorationDecay,
        minExploration: minExploration,
        replayBufferSize: replayBufferSize,
        batchSize: batchSize,
        agentBufferSize: agentBufferSize,
        opponentCopyRate: opponentCopyRate,
        minRandomOpponentEpisodes: minRandomOpponentEpisodes,
        tau: tau,
        maxGradNorm: maxGradNorm,
        minExperiences: minExperiences
    );

    // Prepare training with periodic logs

    int totalEpisodes = 3000;
    int episodeChunks = 10; // number of logged blocks to split training into
    int episodesPerChunk = totalEpisodes / episodeChunks;
    string[] progressLogs = new string[episodeChunks]; // array of performance logs from each training chunk
    int progressTestLength = 5000; // number of games to play during performance testing
    int progressTestWins;
    Stopwatch stopwatch = new();
    stopwatch.Start();
    for (int chunk = 0; chunk < episodeChunks; chunk++)
    {
        dqnTrainer.Train(ref episodeBuffer!, episodesPerChunk);

        // Evaluate agent performance after episodesPerChunk training episodes

        Console.WriteLine($"Evaluating performance after chunk {chunk + 1}...");
        progressTestWins = 0;
        for (int t = 0; t < progressTestLength; t++)
        {
            if (env is TicTacToe tictactoe && tictactoe.PlayRandom(model)) progressTestWins++;
        }
        progressLogs[chunk] = $"Win rate after {(chunk + 1) * episodesPerChunk} episodes: {((double)progressTestWins / (double)progressTestLength) * 100.0}%";
    }
    stopwatch.Stop();
    Console.WriteLine($"Total training and evaluation time: {MathUtils.RoundToMS(stopwatch.Elapsed)}");

    // Log performance testing results

    Console.WriteLine();
    foreach (var log in progressLogs)
    {
        Console.WriteLine(log);
    }
    Console.WriteLine("\nPress any key to close...");
    Console.ReadKey();
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
