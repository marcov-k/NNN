<div style="font-family: monospace; list-style-type: none; padding-left: 0; line-height: 1.5;">
 
# Neural Network Notions
A neural network framework created from scratch in C# and C++ implementing automatic differentiation, backpropagation, and customizable architectures for neural networks.

### View the Complete Documentation [Here](https://github.com/marcov-k/NNN/wiki)

## Key Features
- Deep Q-Network (DQN) training capabilities
- Prioritized experience replay (PER) buffer implementation
- Reverse-mode automatic differentiation
- Dynamic computation graph reused across forward passes
- Dense and convolutional layers
- Standard activation functions (Sigmoid, Tanh, ReLU, etc.)
- Standard loss functions (MSE, pseudo-Huber Loss, Softmax Cross-Entropy)
- Standard optimizers (SGD, Adam)
- Powerful C++ backend with C# interop
- Performance optimizations via SIMD vectorization and parallelization
- Custom file type for saving trained models (.nnn)

## Motivation
I originally intended for this project to simply be my experimentation with implementing the systems described in Seth Weidman's _Deep Learning from Scratch_. However, after seeing my basic neural networks successfully
 train using the Boston housing dataset highlighted in Weidman's book, I became increasingly interested in creating a framework which could support Deep Q-Network training (DQN) for complex environments. After a number
 of failed attempts using the framework I had derived from the examples in _Deep Learning from Scratch_, I came to the realization that in order to support more complex networks while also maintaining sufficiently high
 performance to train such networks on my own personal computer, I would have to completely rewrite the entire framework. At this point I discovered the automatic differentiation algorithms used by libraries such as
 PyTorch, and decided that my new framework would follow a similar approach. Additionally, in order to gain as thorough of an understanding of the mathematics and logic behind these systems, I opted to avoid using
 any features not provided in the C# and C++ standard libraries, which I also believed would provide me with valuable experience in designing full-scale frameworks from scratch. Now, many months after I first began
 experimenting with the Neural Network Nonsense project, I can proudly say that those initial efforts have grown into something far larger than I could have ever anticipated.

## Installation
### Compatibility:
- Compatible with .NET Standard 2.1 and newer
- Requires 64-bit Windows with the Microsoft Visual C++ Redistributable installed

### Raw DLL Download
1. Download the NNN-vX.X.X.zip file from the GitHub release.
2. Extract the .zip file - you will see the NNN/ directory containing the runtime/ directory, as well as the LICENSE and README files.
4. Add the DLL files (NNN.dll and NNNCSharp.dll) in the extracted runtime/ directory into anywhere in your project's directory.
5. Add the following to your .csproj file:
   ```
   <ItemGroup>
     <Reference Include="NNNCSharp">
       <HintPath>Relative/Path/To/NNNCSharp.dll</HintPath>
     </Reference>
   </ItemGroup>
   ```
6. Follow the [Creating a Custom Training Environment](#creating-a-custom-training-environment), [Training a Model](#training-a-model), and/or [Saving/Loading Models](#savingloading-models) guides, or refer to the [C# API Documentation](https://github.com/marcov-k/NNN/wiki/C%23-API) to implement Neural Network Notions in your code.

### NuGet Package Install
- #### Option 1 - NuGet Package Manager
    Search for "NNN" in the Visual Studio NuGet package manager and install.
- #### Option 2 - Explicit Package Reference
    Add the following to your .csproj file:
    ```
    <ItemGroup>
      <PackageReference Include="NNN" Version="[Version you would like to use]" />
    </ItemGroup>
    ```
Follow the [Creating a Custom Training Environment](#creating-a-custom-training-environment), [Training a Model](#training-a-model), and/or [Saving/Loading Models](#savingloading-models) guides, or refer to the [C# API Documentation](https://github.com/marcov-k/NNN/wiki/C%23-API) to implement Neural Network Notions in your code.

### Unity Package Install
#### Option 1 - Via Git URL
1. Open the Unity Package Manager and click the "+" in the top left corner.
2. Select "Add package from git URL."
3. Enter the following URL: https://github.com/marcov-k/NNN.git#upm
#### Option 2 - Via Zip
1. Download the NNN-UPM-vX.X.X.zip file from the GitHub release.
2. Extract the .zip file and paste the complete extracted folder into your Unity project's Packages/ directory.

Follow the [Creating a Custom Training Environment](#creating-a-custom-training-environment), [Training a Model](#training-a-model), and/or [Saving/Loading Models](#savingloading-models) guides, or refer to the [C# API Documentation](https://github.com/marcov-k/NNN/wiki/C%23-API) to implement Neural Network Notions in your code.

## How to Use
### Using Pretrained Models
1. Locate the .nnn file containing the model you would like to use.<br>Pretrained models for Tic-Tac-Toe and the MNIST dataset can be found in the "NNNSolution/Models/" directory in the GitHub repository.
2. Copy the .nnn file into a directory in your project.
3. Specify the directory NNN should load models from:
```
using NNNCSharp.Components.Utilities.SaveSystem;

Saver.DirectoryPath = "[path to your directory containing the model]";
```
&emsp;For Unity projects add the directory to your project's "Assets/StreamingAssets" directory (create StreamingAssets manually if it does not exist) and specify the path as so:
```
Saver.DirectoryPath = Path.Combine(Application.streamingDataPath, "[path to your directory from StreamingAssetsAssets/]");
```
4. Follow the [Saving/Loading Models](#savingloading-models) guide to load the model in your code.

#### Pretrained Model Specifications
- Tic-Tac-Toe (tictactoedemo.nnn):
  - Win/Tie Rate: 100% (Based on 5000 games against randomly-acting opponent)
  - Expected Input Dimensions: [batch, 10] - batch = 1 for selecting next position while playing<br>Can use dimensions of [10] for board encoding and use Tensor.WrapBatch() when getting model predictions:
    ```
    using Tensor wrapped = Tensor.WrapBatch(state);
    Tensor qValues = model.Predict(wrapped);
    ```
  - Input Encoding:
    - Index 0-8 -> Board position values - row-major indexing with top-left being 0
    - Board Position Encoding -> 0 = Empty, 1 = X, -1 = O
    - Index 9 -> Player to act
    - Player Encoding -> 1 = X, -1 = O

### Creating a Custom Training Environment
#### DQN Environment
1. Create a class inheriting from the DQNEnvironment abstract class:
```
using NNNCSharp.Components.DQNEnvironments;

public class MyDQNEnv : DQNEnvironment {}
```
&emsp;For self-play environments also implement the ISelfPlay interface
```
public class MySelfPlayDQNEnv : DQNEnvironment, ISelfPlay {}
```
2. Override the following properties:
```
// The shape of the Tensor the environment provides to agents (first dimension represents batches)
public override Tensor StateFormat => new(new int[] { 1, [your state dimensions] });

// The number of unique actions the agent can take for a given step
public override int ActionCount => [your action count];
```
&emsp;For self-play environments also implement the following properties:
```
// Whether it is currently the agent's turn to play
public bool AgentTurn { get; set; }

// The number of opponent agents available to play against
public int OpponentCount { get; set; }

// The index of the opponent agent being played against during this episode
public int OpponentIndex { get; set; }
```
3. Override the following methods:
```
// Return the normalized form of the environment's current state to give to an agent
public override Tensor GetNormalizedState() {}

// Return the unnormalized form of the environment's current state
public override Tensor GetState() {}

// Reset the environment's state to its initial state and prepares a new episode
public override void Reset() {}

// Return the index of the highest Q-Value corresponding to a valid action in the given state
// (Default to environment's current state if no state is given)
public override int PickAgentAction(Tensor qValues, Tensor? state = null) {}

// Randomly select a valid action given the environment's current state
public override int PickRandomAction() {}

// Return whether the given action is valid in the given state
// (Default to environment's current state if no state is given)
public override bool ValidAction(int action, Tensor? state) {}

// Perform a single step in the environment using the given action
// (The steps parameter can be used terminate an episode after a fixed number of steps)
// Return the following:
//   The reward/penalty accrued by the action
//   The normalized form of the environment's state after the action is taken (identical to GetNormalizedState())
//   Whether the episode has finished
public override (double reward, Tensor nextState, bool done) Step(int action, int steps) {}

// Run the given number of episodes with the given agent
// Return a value representing the agent's average performance across the test episodes
public override double TestTrainingProgress(Model agent, int testEpisodes) {}
```
&emsp;For self-play environments also implement the following methods:
```
// Use the given agent to select an action in the given state
// (Default to environment's current state if no state is given)
public int GetAgentAction(Model agent, Tensor? state = null) {}
```
4. Refer to the [Training a DQN Agent](#dqn-training) guide to train an agent for your custom DQN environment and/or the [Set Logging Output Target](#set-logging-output-target) guide to set up logging.

### Training a Model
#### Special Cases:
- When using Model.Forward() or Model.Predict() with a single input instead of a batch, use Tensor.WrapBatch() on the input first to convert it into a batch of 1 input.
- Whenever creating a new Tensor instance through any constructor, function or operator, ensure the instance is disposed via Tensor.Dispose() once it is no longer being used - otherwise it may become a memory leak.

#### Standard Supervised Training:
```
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Buffers;
using NNNCSharp.Components.Costs;
using NNNCSharp.Components.Models;
using NNNCSharp.Componens.Models.Layers;
using NNNCSharp.Components.Optimizers;
using NNNCSharp.Components.Trainers;

Tensor[] trainData; // array containing all of your individual training inputs
Tensor[] trainTargets; // array containing all of your individual training targets

// BatchBuffer will automatically create batch tensors from your trainData and trainTargets arrays
BatchBuffer yourBatchBuffer = new(trainData, trainTargets);

Tensor inputFormat = new(new int[] { 1, [your training data dimensions] }); // specifies input shape the model should expect

// Creates a model with the following architecture:
// Convolutional layer with 8 filters, 5x5 kernels, and the ReLU activation function
// Convolutional layer with 16 filters, 5x5 kernels, the ReLU activation function, and a spatial dropout of 0.1
// Fully connected (Dense) layer with 128 neurons, the ReLU activation function, and a dropout of 0.25
// Fully connected (Dense) layer with 10 (output) neurons, and no (Linear) activation function
Model yourModel = new([
  new Conv(8, new int[] { 5, 5 }, new ReLU()),
  new Conv(16, new int[] { 5, 5 }, new ReLU(), 0.1),
  new Dense(128, new ReLU(), 0.25),
  new Dense(10, new Linear())
  ], inputFormat);

Optimizer yourOptimizer = new SGD([desired learning rate]); // stochastic gradient descent optimizer
Cost yourCost = new MSE(); // mean squared error loss

Trainer yourTrainer = new(yourModel, yourOptimizer, yourCost, [maximum gradient norm (for gradient clipping)]);

yourTrainer.Train(yourBatchBuffer, [batch size], [epochs to train for],
  [whether to train on all batches every epoch (true/false)],
  [optional function for testing performance*], [optional learning rate decay rate],
  [optional minimum learning rate fraction], [how many epochs to run between performance tests],
  [how many inputs to test per performance test]);
// *The performance test function must match the declaration 'Func<Model, int, bool>'
// receiving the model to test and the test index as inputs, and returning a boolean
// based on whether the model passed the test or not.

yourModel = yourTrainer.Model; // get the best-performing model from the trainer
```

#### DQN Training:
```
using NNNCSharp.Components.Costs;
using NNNCSharp.Components.DQNEnvironments;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Models.Layers;
using NNNCSharp.Components.Optimizers;
using NNNCSharp.Components.Trainers;

DQNEnvironment yourEnv; // the DQNEnvironment subclass you want to train in

// Creates a model with the following architecture:
// Convolutional layer with 8 filters, 3x3 kernels, and the ReLU activation function
// Convolutional layer with 16 filters, 3x3 kernels, the ReLU activation function, and a spatial dropout of 0.05
// Fully connected (Dense) layer with 128 neurons, the ReLU activation function, and a dropout of 0.1
// Fully connected (Dense) layer with 64 neurons, the ReLU activation function, and a dropout of 0.1
// Fully connected (Dense) layer 1 (output) neuron per discrete action in your DQNEnvironment, and no (Linear) activation function
Model yourModel = new([
  new Conv(8, new int[] { 3, 3 }, new ReLU()),
  new Conv(16, new int[] { 3, 3 }, new ReLU(), 0.05),
  new Dense(128, new ReLU(), 0.1),
  new Dense(64, new ReLU(), 0.1),
  new Dense(yourEnv.ActionCount, new Linear())
  ], yourEnv.StateFormat);

Optimizer yourOptimizer = new SGD([desired learning rate]); // stochastic gradient descent optimizer
Cost yourCost = new MSE() // mean squared error loss

DQNTrainer yourTrainer = new(yourModel, yourEnv, [initial exploration rate], [exploration rate decay],
  [minimum exploration rate], [how many steps to take between training on a batch], [discount factor],
  yourOptimizer, yourCost, [how many experiences to store for replay], [how many opponent agents to store (for self-play environments)],
  [batch size], [number of episodes between opponent agent copies (for self-play environments)],
  [minimum number of initial episodes against a random opponent (for self-play environments)], [tau factor to use for target model updates],
  [maximum gradient norm (for gradient clipping)], [minimum number of experiences before starting to train (must be >= batch size)]);

yourTrainer.Train([optional ref FIFOBuffer<Episode> buffer for storing past episodes], [episodes to train for],
  [number of episodes between performance tests], [number of episodes run during each performance test]);

yourModel = yourTrainer.Agent; // get the best-performing agent from the trainer
```
&emsp;Refer to the [Set Logging Output Target](#set-logging-output-target) guide to set up logging.

### Saving/Loading Models
#### Specify the directory to save/load models from:
```
using NNNCSharp.Components.Utilities.SaveSystem;

Saver.DirectoryPath = "[your target directory]";
```
&emsp;For Unity projects add the directory to your project's "Assets/StreamingAssets" directory (create StreamingAssets manually if it does not exist) and specify the path as so:
```
Saver.DirectoryPath = Path.Combine(Application.streamingDataPath, "[path to your directory from StreamingAssetsAssets/]");
```
*The framework will automatically create a directory with the given path if none exists.
   
#### Save a model to a file:
```
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Utilities.SaveSystem;

Saver.SaveModel(yourModel, "[yourfilename]", "[optional short description]"); // file name without any extension
```

#### Load a model from a file:
```
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Utilities.SaveSystem;

Model yourModel = Saver.LoadModel("[yourfilename]"); // file name without any extension
```

### Set Logging Output Target
```
using NNNCSharp.Components.Utilities

NNNLog.Output = [your target output] (eg. Console.Write, Debug.Log, etc.)
```

## Testing Results
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

    // Calculate relative error for every gradient of each input
    for (int input = 0; input < inputs.Length; input++)
    {
        for (int e = 0; e < inputs[input].ElementCount; e++)
        {
            var numerical = NumericalGradient(inputs, input, e, loss);
            double analytical = inputs[input].Grad[e];
            double relError = Math.Abs(numerical - analytical) / (Math.Abs(numerical) + 1e-8);
            Console.WriteLine($"inputs[{input}][{e}]: numerical = {numerical}, analytical = {analytical}, relError = {relError}");
        }
    }
}

static double NumericalGradient(Tensor[] inputs, int inputIndex, int e, Func<Tensor[], double> loss)
{
    // Estimate gradient via finite difference
    double eps = 1e-8;
    inputs[inputIndex][e] += eps;
    double lossPlus = loss(inputs);
    inputs[inputIndex][e] -= 2 * eps;
    double lossMinus = loss(inputs);
    inputs[inputIndex][e] += eps;
    return (lossPlus - lossMinus) / (2 * eps);
}
```

### Supervised Learning Convergence Test (XOR Classification)
#### Specifications:
Architecture: 4 -> 1 (Sigmoid -> Linear)\
Optimizer: Adam\
Learning Rate: 0.01\
Target MSE: < 0.01
#### Training Results:
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

### MNIST Dataset
#### Specifications:
Architecture: Conv(8 filters, 5x5 kernel) -> Conv(16 filters, 5x5 kernel, 0.2 dropout) -> 128 (0.5 dropout) -> 10 (Leaky ReLU, Leaky ReLU, Leaky ReLU, Linear)\
Loss Function: Softmax Cross-Entropy\
Optimizer: Adam (0.01 weight decay)\
Initial Learning Rate: 0.001 (with decay)\
Final Learning Rate: 0.0001\
Total Time Required for Training and Evaluation: 2:01:01.393\
Training Inputs: Standard 60,000 training image dataset\
Testing Inputs: Standard 10,000 testing image dataset
#### Training Results:
| Epochs | Accuracy |
|--------|----------|
| 0* | 9.82% |
| 5 | 95.30% |
| 10 | 96.00% |
| 15 | 96.36% |
| 20 | 95.98% |
| 25 | 96.80% |
| 30 | 96.85% |
| 35 | 97.16% |
| 40 | 96.94% |
| 45 | 97.26% |
| 50 | 97.42% |

*Accuracy prior to any training being done
#### Code Used for Testing:
```
Console.WriteLine("Loading MNIST dataset...");
var (trainImages, trainLabels) = MNISTLoader.GetTrainingData();
var (testImages, testLabels) = MNISTLoader.GetTestData();
Console.WriteLine("MNIST dataset loaded");

double tau = 0.05;
double convDropout = 0.15;
double denseDropout = 0.5;
Model model;
if (GetInput("Load model from file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
{
    // Load model from file
    string fileName = GetFileName();
    model = Saver.LoadModel(fileName);
}
else
{
    model = new([
        new Conv(8, [5, 5], new LeakyReLU(tau)),
        new Conv(16, [5, 5], new LeakyReLU(tau), convDropout),
        new Dense(128, new LeakyReLU(tau), denseDropout),
        new Dense(10, new Linear())
    ], new([1, 28, 28, 1]));
}

Optimizer optimizer = new Adam(0.001, weightDecay: 0.01);
double maxGradNorm = 0.5;
Cost cost = new SoftmaxCrossEntropy();
Trainer trainer = new(model, optimizer, cost, maxGradNorm);

var wrappedImages = new Tensor[testImages.Length];
for (int i = 0; i < testImages.Length; i++)
{
    wrappedImages[i] = Tensor.WrapBatch(testImages[i]);
}

Func<Model, int, bool> testFunc = (model, i) =>
{
    var predicts = model.Predict(wrappedImages[i]);
    return Tensor.ArgMax(predicts) == Tensor.ArgMax(testLabels[i]);
};

BatchBuffer batchBuffer = new(trainImages, trainLabels);
int batchSize = 128;
int testLength = testLabels.Length;
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
#### Training Results:
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
DQNEnvironment env = new TicTacToe();
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
- Serializes and saves trained models to custom .nnn format files
- Deserializes and reconstructs models from custom .nnn format files

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
NNN-Solution - Directory (Full project solution)\
├── NNN - Directory (C++ backend DLL)\
│&emsp;&ensp;├── Header Files - Solution Directory\
│&emsp;&ensp;│&emsp;&ensp;├── DataContainers - Header File (POD struct implementations)\
│&emsp;&ensp;│&emsp;&ensp;├── exports - Header File (DLL export function declarations)\
│&emsp;&ensp;│&emsp;&ensp;├── framework - Header File (Standard framework header)\
│&emsp;&ensp;│&emsp;&ensp;├── MathUtils - Header File (Math vectorization and utility function declarations)\
│&emsp;&ensp;│&emsp;&ensp;├── Models - Header File (Model-level function declarations)\
│&emsp;&ensp;│&emsp;&ensp;├── Optimizers - Header File (Optimizer function declarations)\
│&emsp;&ensp;│&emsp;&ensp;├── pch - Header File (Standard precompiled header)\
│&emsp;&ensp;│&emsp;&ensp;└── Tensor - Header File (Tensor class header and function declarations)\
│&emsp;&ensp;│\
│&emsp;&ensp;└── Source Files - Solution Directory\
│&emsp;&emsp;&emsp;&ensp;├── dllmain - C++ Script (Standard DLL boilerplate)\
│&emsp;&emsp;&emsp;&ensp;├── exports - C++ Script (DLL export function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── MathUtils - C++ Script (Math vectorization and utility function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── Models - C++ Script (Model-level function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── Optimizers - C++ Script (Optimizer function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── pch - C++ Script (Standard procompiled header script)\
│&emsp;&emsp;&emsp;&ensp;├── TensorActivations - C++ Script (Tensor activation function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorCosts - C++ Script (Tensor cost function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorGraph - C++ Script (Autograd graph function implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorIndexing - C++ Script (Tensor property access implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorInitializations - C++ Script (Tensor initialization implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorOperations - C++ Script (Tensor operation implementations)\
│&emsp;&emsp;&emsp;&ensp;├── TensorProperties - C++ Script (Tensor property initializations)\
│&emsp;&emsp;&emsp;&ensp;└── TensorUtilities - C++ Script (Tensor utility function implementations)\
│\
├── NNNCSharp - Directory (C# implementations and interop with C++ backend)\
│&emsp;&ensp;└── Components - Directory\
│&emsp;&emsp;&emsp;&ensp;├── Activations - Directory (Activation function classes)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Activation - C# Script (Base class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── LeakyReLU - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Linear - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── ReLU - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Sigmoid - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Tanh - C# Script\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Autodiff - Directory (Automatic differentation and tensor logic - implemented via a partial Tensor class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Tensor - C# Script (Wrapper for interop with C++ tensor logic)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Buffers - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── BatchBuffer - C# Script (Standard supervised training buffer)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── FIFOBuffer - C# Script (Standard First-In First-Out buffer)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── ReplayBuffer - C# Script (PER buffer for DQN experience replay)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── SumTree - C# Script (Standard sum tree data structure)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── CompilerAttributes - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── CompilerAttributes - C# Script (Applies necessary compiler attributes)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Costs - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Cost - C# Script (Base class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Huber - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── MSE - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── SoftmaxCrossEntropy - C# Script\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── DQNEnvironments - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── DQNEnvironment - C# Script (Base class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── MovementGrid2D - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Snake - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── TicTacToe - C# Script\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Episodes - Directory (Data structures for DQN experience storage)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Episode - C# Script (Data structure for a full DQN training episode)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Experience - C# Script (Data structure for a single DQN training experience)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Interop - Directory (C# interop with C++ backend)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── NativeMethods - C# Script (DLL imports for C++ backend methods)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── TensorSafeHandle - C# Script (Safe handle for C# Tensor wrapper)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Models - Directory (Neural network functionality)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Layers - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;│&emsp;&ensp;├── Conv - C# Script (Convolutional)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;│&emsp;&ensp;├── Dense - C# Script (Fully connected)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;│&emsp;&ensp;└── Layer - C# Script (Base class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Model - C# Script (Full neural network container class)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Optimizers - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Adam - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── Optimizer - C# Script (Base class)\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── SGD - C# Script\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;├── Trainers - Directory\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── DQNTrainer - C# Script\
│&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Trainer - C# Script (Standard supervised training)\
│&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&ensp;└── Utilities - Directory\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── DataLoaders - Directory\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── MNISTLoader - C# Script\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── SaveSystem - Directory\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;├── FileUtils - C# Script (Static class for reading and writing .nnn files)\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;│&emsp;&ensp;└── Saver - C# Script (Static class for handling model saving/loading)\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;│\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── ArrayUtils - C# Script\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── IDManager - C# Script (Static class for looking up IDs for file format)\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── MathUtils - C# Script\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;├── NNNLog - C# Script (Logging output target control)\
│&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;└── UIUtils - C# Script\
│\
├── NNNDemo - Directory (Program for running NNN demonstrations)\
│&emsp;&ensp;└── NNNDemo - C# Script (NNN demonstration program script)\
│\
├── NNNExplorer - Directory (NNN file viewer - for viewing .nnn files)\
│&emsp;&ensp;└── NNNExplorer - C# Script (NNN file viewer program script)\
│\
├── NNNTester - Directory (Program for testing changes made to C# and C++ implementations)\
│&emsp;&ensp;└── NNNTester - C# Script (NNN tester program script)\
│\
└── NNNTrainer - Directory (Program for training neural networks using the NNN framework)\
&emsp;&emsp;└── NNNTrainer - C# Script (NNN trainer program script)

## File Format (.nnn)
### General Formatting Notes:
- All multi-byte numbers use little-endian encoding
- Each [layer type's](#layer-data-format) encoding includes different parameters - identified by [Layer ID](#layer-ids)
- Certain [activation functions](#activation-function-data-format---found-immediately-after-activation-id-for-activation-functions-with-parameters) encode additional parameters (eg. Leaky ReLU's Tau) immediately after their ID - identified by [Activation ID](#activation-ids)
- Data appears in the file in the exact order as listed below

### File Header Format:
- Magic Number -> int32 (4 bytes) -> 776883790 (spells ".NNN" in ASCII)
- Description -> string ([see formatting](#string-format-utf8)) -> short description of the model in the file
- Parameter Count -> unsigned int64 (8 bytes) -> total number of parameter values across all parameter tensors in the model

### Model Header Format:
- Layer Count -> int32 (4 bytes) -> number of layers in the model

### Layer Header Format:
- Layer ID -> unsigned byte -> ID of the specific layer type ([see ID list](#layer-ids))

### Layer Data Format:
- #### Shared - always comes before type-specific data:
  - Activation ID -> unsigned byte -> ID of the layer's activation function ([see ID list](#activation-ids))
  - Dropout -> double (8 bytes) -> dropout parameter of the layer
  - Bias -> tensor ([see formatting](#tensor-format)) -> bias parameter of the layer

- #### Dense:
  - Neuron Count -> int32 (4 bytes) -> number of neurons in the layer
  - Weights -> tensor ([see formatting](#tensor-format)) -> weights parameter of the layer
  - Flatten -> boolean ([see formatting](#boolean-format)) -> whether the layer flattens its input prior to applying weights

- #### Conv:
  - Filter Count -> int32 (4 bytes) -> number of filters in the layer
  - Kernels -> tensor ([see formatting](#tensor-format)) -> kernels parameter of the layer ([f, h, w..., c] ordering)

### Activation Function Data Format - found immediately after Activation ID (for activation functions with parameters):
- #### Leaky ReLU:
  - Tau -> double (8 bytes) -> tau parameter of the function

### Tensor Format:
- Dimensions -> int32 array ([see formatting](#int32-array-format)) -> dimensions of the tensor
- RequiresGrad -> boolean ([see formatting](#boolean-format)) -> whether the tensor requires gradients to be calculated
- Data -> double array ([see formatting](#double-array-format)) -> linear array of the tensor's data values (row-major ordering)

### String Format (UTF8):
- Length -> int32 (4 bytes) -> number of characters in the string
- Characters -> byte[Length] -> UTF8 character bytes

### Int32 Array Format:
- Length -> int32 (4 bytes) -> number of elements in the array
- Elements -> int32[Length] (4 bytes each) -> elements in the array

### Double Array Format:
- Length -> int32 (4 bytes) -> number of elements in the array
- Elements -> double[Length] (8 bytes each) -> elements in the array

### Boolean Format:
- 1 byte -> 1 = true, 0 = false

### Layer IDs:
- 0 -> Dense
- 1 -> Conv

### Activation IDs:
- 0 -> Linear
- 1 -> Sigmoid
- 2 -> Tanh
- 3 -> ReLU
- 4 -> Leaky ReLU
- 5 -> Softmax

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
