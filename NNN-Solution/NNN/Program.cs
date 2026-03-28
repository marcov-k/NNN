using NNN;

Model model;
NNN.Environment env = new MovementGrid2D(-5, 5, -5, 5);
DQNTrainer dqnTrainer;

InteractionLoop();

void InteractionLoop()
{
    Console.WriteLine("Welcome to the DQN Training Terminal (Enter Q to quit)");

    string input = GetInput("Load model from file? y/n", ["y", "n"]);
    if (input == "y")
    {
        string fileName = GetFileName();
        model = Saver.LoadModel(fileName);
    }
    else
    {
        model = new([
            new Dense(32, new LeakyReLU()),
            new Dense(32, new LeakyReLU()),
            new Dense(4, new Linear())
        ], new Tensor(0, 4));
    }

    dqnTrainer = new DQNTrainer(
        agent: model,
        environment: env,
        actionCount: 4,
        explorationDecay: 0.999,
        discount: 0.99,
        optimizer: new Adam(),
        cost: new MSE(),
        replayBufferSize: 20000,
        batchSize: 64,
        minExperiences: 1000
    );

    TrainingLoop();

    if (GetInput("Save model to a file? y/n", ["y", "n"]) == "y")
    {
        SaveLoop();
    }

    Console.WriteLine("\nPress any key to quit...");
    Console.ReadKey();
    System.Environment.Exit(0);
}

void TrainingLoop()
{
    while (true)
    {
        string input = GetInput("Run DQN Training episodes? y/n", ["y", "n"]);
        if (input == "y")
        {
            int episodes = GetInteger("Enter number of episodes to train");
            Console.WriteLine($"Training for {episodes} episodes...");
            dqnTrainer.Train(episodes);

            TestDQNModel();
        }
        else break;
    }
}

void TestDQNModel()
{
    Console.WriteLine("\n--- Testing Agent Performance ---");
    env.Reset();
    var state = env.GetNormalizedState();
    var startState = env.GetState();
    Tensor batchState = new(1, state.Dimensions[0]);
    bool done = false;
    double totalReward = 0;
    int steps = 0;

    while (!done && steps < 50)
    {
        steps++;
        batchState.InsertSubArray(0, state);
        int action = model.Forward(batchState).MaxIndex();
        var (reward, nextState, isDone) = env.Step(action, steps);

        totalReward += reward;
        state = nextState;
        done = isDone;
    }

    var logState = env.GetState();
    Console.WriteLine($"Test Finished in {steps} steps.");
    Console.WriteLine($"Total Reward: {totalReward:F2}");
    Console.WriteLine($"Starting State: ({startState[0].Value}, {startState[1].Value})");
    Console.WriteLine($"Final State: ({logState[0].Value}, {logState[1].Value}), Target: ({logState[2].Value}, {logState[3].Value})");
}

string GetInput(string prompt, List<string>? options = null)
{
    options ??= [];
    for (int i = 0; i < options.Count; i++)
    {
        options[i] = options[i].ToLowerInvariant();
    }

    string input;
    while (true)
    {
        Console.WriteLine($"\n{prompt}");
        input = Console.ReadLine()?.ToLowerInvariant() ?? "";

        if (input == "q") System.Environment.Exit(0);
        else if (options.Count == 0 || options.Contains(input)) return input;
    }
}

string GetFileName()
{
    string input;
    while (true)
    {
        input = GetInput("Enter file name");
        if (Saver.FileExists(input)) return input;
        else Console.WriteLine("\nFile not found");
    }
}

int GetInteger(string prompt)
{
    string input;
    while (true)
    {
        input = GetInput(prompt);
        if (int.TryParse(input, out int integer)) return integer;
        else Console.WriteLine("\nNot a valid number");
    }
}

void SaveLoop()
{
    string fileName;
    string input;

    while (true)
    {
        fileName = GetInput("Enter file name");
        if (Saver.FileExists(fileName))
        {
            input = GetInput($"File with name \"{fileName}\" already exists. Overwrite existing file? y/n", ["y", "n"]);
            if (input == "y")
            {
                Saver.SaveModel(model, fileName);
                Console.WriteLine("\nModel saved");
                break;
            }
        }
        else
        {
            Saver.SaveModel(model, fileName);
            Console.WriteLine("\nModel saved");
            break;
        }
    }
}

namespace NNN
{
    using System.Diagnostics;
    using System.Text.Json;

    public abstract class Environment
    {
        public virtual Tensor GetNormalizedState()
        {
            throw new NotImplementedException();
        }

        public virtual Tensor GetState()
        {
            throw new NotImplementedException();
        }

        public virtual (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            throw new NotImplementedException();
        }

        public virtual void Reset()
        {
            throw new NotImplementedException();
        }
    }

    public class MovementGrid2D : Environment
    {
        readonly Random random = new();
        readonly Tensor State = new(4); // current x, current y, target x, target y
        readonly int[] Bounds; // xMin, xMax, yMin, yMax

        public MovementGrid2D(int xMin, int xMax, int yMin, int yMax)
        {
            State[0] = new(random.Next(xMin, xMax + 1));
            State[1] = new(random.Next(yMin, yMax + 1));
            State[2] = new(random.Next(xMin, xMax));
            State[3] = new(random.Next(yMin, yMax));
            Bounds = [xMin, xMax, yMin, yMax];
        }

        public override Tensor GetNormalizedState()
        {
            Tensor normalized = new(4);

            normalized[0] = new(State[0].Value / Math.Max(Bounds[0], Bounds[1]));
            normalized[1] = new(State[1].Value / Math.Max(Bounds[2], Bounds[3]));
            normalized[2] = new(State[2].Value / Math.Max(Bounds[0], Bounds[1]));
            normalized[3] = new(State[3].Value / Math.Max(Bounds[2], Bounds[3]));

            return normalized;
        }

        public override Tensor GetState()
        {
            return State.Copy();
        }

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            double xDiff = State[2].Value - State[0].Value;
            double yDiff = State[3].Value - State[1].Value;
            double prevDist = Math.Sqrt(Math.Pow(xDiff, 2.0) + Math.Pow(yDiff, 2.0));

            switch (action)
            {
                case 0: // left
                    State[0].Value--;
                    break;
                case 1: // right
                    State[0].Value++;
                    break;
                case 2: // up
                    State[1].Value++;
                    break;
                case 3: // down
                    State[1].Value--;
                    break;
            }

            xDiff = State[2].Value - State[0].Value;
            yDiff = State[3].Value - State[1].Value;
            double newDist = Math.Sqrt(Math.Pow(xDiff, 2.0) + Math.Pow(yDiff, 2.0));

            double reward = -0.05 + prevDist - newDist;

            bool done = false;

            bool reachedTarget = (State[0].Value == State[2].Value && State[1].Value == State[3].Value);
            bool outOfBounds = (State[0].Value < Bounds[0]) || (State[0].Value > Bounds[1]) ||
                               (State[1].Value < Bounds[2]) || (State[1].Value > Bounds[3]);

            if (newDist < 2.0) reward += 1.0;
            if (outOfBounds) reward -= 5.0;
            if (reachedTarget)
            {
                reward += 5.0;
                done = true;
            }

            if (steps >= 50)
            {
                reward -= 5.0;
                done = true;
            }

            reward = Math.Tanh(reward); // clip rewards to [-1, 1] range

            return (reward, GetNormalizedState(), done);
        }

        public override void Reset()
        {
            State[0] = new(random.Next(Bounds[0], Bounds[1] + 1));
            State[1] = new(random.Next(Bounds[2], Bounds[3] + 1));
            State[2] = new(random.Next(Bounds[0], Bounds[1] + 1));
            State[3] = new(random.Next(Bounds[2], Bounds[3] + 1));
        }
    }

    public record Experience(Tensor State, int Action, double Reward, Tensor NextState, bool Done)
    {
        public readonly Tensor State = State.Copy();
        public readonly int Action = Action;
        public readonly double Reward = Reward;
        public readonly Tensor NextState = NextState.Copy();
        public readonly bool Done = Done;
    }

    public class ReplayBuffer(int maxSize)
    {
        readonly Random random = new();
        readonly int MaxSize = maxSize;
        readonly List<Experience> Buffer = [];
        public int Count => Buffer.Count;

        public void AddExperience(Experience experience)
        {
            Buffer.Add(experience);
            if (Buffer.Count > MaxSize)
            {
                Buffer.RemoveAt(0);
            }
        }

        public List<Experience> GetBatch(int batchSize)
        {
            List<Experience> batch = [];
            for (int i = 0; i < batchSize; i++)
            {
                batch.Add(Buffer[random.Next(0, Buffer.Count)]);
            }
            return batch;
        }
    }

    public class DQNTrainer(Model agent, Environment environment, int actionCount, Optimizer optimizer, Cost cost, double discount = 0.995, double exploration = 1.0,
        double explorationDecay = 0.99, int replayBufferSize = 10000, int batchSize = 64, int minExperiences = 1000)
    {
        readonly Random random = new();
        readonly Model Agent = agent;
        Model TargetModel = agent.Copy();
        readonly Environment Environment = environment;
        readonly int ActionCount = actionCount;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;
        readonly ReplayBuffer ReplayBuffer = new(replayBufferSize);
        readonly int BatchSize = batchSize;
        readonly double Discount = discount;
        double Exploration = exploration;
        readonly double ExplorationDecay = explorationDecay;
        readonly double MinExploration = 0.01;
        int totalSteps = 0;
        int optimizerSteps = 0;
        readonly int TargetUpdateFrequency = 500;
        readonly int MinExperiences = minExperiences;

        public void Train(int episodes = 1000)
        {
            Tensor state;
            Tensor initialState;
            Tensor logState;
            bool done;
            int action;
            double reward;
            double totalReward;
            int step;
            Tensor nextState;
            TimeSpan avgElapsed = new(0);
            Stopwatch stopwatch = new();
            stopwatch.Start();
            for (int e = 0; e < episodes; e++)
            {
                Environment.Reset();
                state = Environment.GetNormalizedState();
                initialState = Environment.GetState();

                done = false;
                step = 0;
                totalReward = 0;
                while (!done)
                {
                    totalSteps++;
                    step++;
                    action = PickNextAction(state);
                    (reward, nextState, done) = Environment.Step(action, step);
                    totalReward += reward;
                    ReplayBuffer.AddExperience(new(state, action, reward, nextState, done));

                    TrainNetwork();

                    state = nextState;

                    if (totalSteps >= TargetUpdateFrequency)
                    {
                        TargetModel = Agent.Copy();
                        totalSteps = 0;
                    }
                }

                Exploration = Math.Max(Exploration * ExplorationDecay, MinExploration);

                var elapsed = stopwatch.Elapsed;
                avgElapsed += (elapsed - avgElapsed) / (e + 1);
                var eta = avgElapsed * (episodes - e - 1);

                logState = Environment.GetState();
                Console.WriteLine($"\nEpisode {e + 1}/{episodes} finished...");
                Console.WriteLine($"Initial State: ({initialState[0].Value}, {initialState[1].Value}), Final State: ({logState[0].Value}, {logState[1].Value}), Target: (" +
                    $"{initialState[2].Value}, {initialState[3].Value}), Steps Taken: {step}, Total Reward: {totalReward:F2},");
                Console.WriteLine($"Exploration Rate: {Exploration:F2}, Experience Count: {ReplayBuffer.Count}");
                Console.WriteLine($"Episode Duration: {elapsed}, Estimated Time Remaining: {eta}");
                stopwatch.Restart();
            }
        }

        int PickNextAction(Tensor state)
        {
            if (random.NextDouble() < Exploration)
            {
                return random.Next(0, ActionCount);
            }
            else
            {
                Tensor batchState = new(1, state.Dimensions[0]);
                batchState.InsertSubArray(0, state);

                return Agent.Forward(batchState).MaxIndex();
            }
        }

        void TrainNetwork()
        {
            if (ReplayBuffer.Count < MinExperiences) return;

            var batch = ReplayBuffer.GetBatch(BatchSize);
            int stateSize = batch[0].State.Dimensions[0];

            Tensor currentBatch = new(BatchSize, stateSize);
            Tensor nextBatch = new(BatchSize, stateSize);
            
            for (int i = 0; i < BatchSize; i++)
            {
                currentBatch.InsertSubArray(i, batch[i].State);
                nextBatch.InsertSubArray(i, batch[i].NextState);
            }

            var predictions = Agent.Forward(currentBatch);
            var nextAgentQs = Agent.Forward(nextBatch).Copy();
            var nextTargetQs = TargetModel.Forward(nextBatch).Copy();

            var predictedQs = MaskQValues(predictions, batch, breakGraph: false);
            var targetQs = MaskQValuesDouble(nextAgentQs, nextTargetQs, batch);

            Agent.ZeroGrad();

            var loss = Cost.CalculateCost(predictedQs, targetQs);
            loss.Backward();
            Agent.Optimize(Optimizer, optimizerSteps);
            optimizerSteps++;
        }

        Tensor MaskQValues(Tensor qValues, List<Experience> batch, bool breakGraph)
        {
            if (breakGraph)
            {
                var targetQs = qValues.ReduceDimensions();
                Tensor maskedQs = new(BatchSize, 1);

                for (int i = 0; i < BatchSize; i++)
                {
                    double qTarget = batch[i].Reward;
                    if (!batch[i].Done)
                    {
                        qTarget += Discount * targetQs[i].Max().Value;
                    }

                    maskedQs[i] = new(qTarget);
                }

                return maskedQs;
            }
            else
            {
                Tensor maskedQs = new(BatchSize, 1);

                for (int i = 0; i < BatchSize; i++)
                {
                    maskedQs[i] = qValues[i, batch[i].Action];
                }

                return maskedQs;
            }
        }

        Tensor MaskQValuesDouble(Tensor agentQValues, Tensor targetQValues, List<Experience> batch)
        {
            var agentQs = agentQValues.ReduceDimensions();
            var targetQs = targetQValues.ReduceDimensions();

            Tensor maskedQs = new(BatchSize, 1);

            for (int i = 0; i < BatchSize; i++)
            {
                double qTarget = batch[i].Reward;

                if (!batch[i].Done)
                {
                    int bestAction = agentQs[i].MaxIndex();
                    double evalQ = targetQs[i][bestAction].Value;
                    qTarget += Discount * evalQ;
                }

                maskedQs[i] = new(qTarget);
            }

            return maskedQs;
        }
    }

    public class Trainer(Model model, Optimizer optimizer, Cost cost)
    {
        readonly Model Model = model;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;

        public void Train(Tensor inputs, Tensor targets, int epochs)
        {
            Stopwatch timer = new();

            int logEvery = Math.Max(100, MathUtils.RoundToInterval(epochs / 500f, 100));
            Tensor predictions;
            Number loss;

            timer.Start();
            for (int e = 0; e < epochs; e++)
            {
                Model.ZeroGrad();
                predictions = Model.Forward(inputs);
                loss = Cost.CalculateCost(predictions, targets);
                loss.Backward();

                Model.Optimize(Optimizer, epochs);

                if (e % logEvery == 0 || e == epochs - 1)
                {
                    Console.WriteLine($"Epoch {e} : Loss = {loss.Value} : Time elapsed = {timer.ElapsedMilliseconds}ms : Time per epoch = {((float)timer.ElapsedMilliseconds / logEvery):F2}ms");
                    timer.Restart();
                }
            }
        }
    }

    public abstract class Optimizer(double learningRate)
    {
        protected readonly double LR = learningRate;

        public virtual void Step(Number parameter, int iterations)
        {
            throw new NotImplementedException();
        }
    }

    public class SGD(double learningRate) : Optimizer(learningRate)
    {
        public override void Step(Number parameter, int iterations)
        {
            parameter.Value -= parameter.Gradient * LR;
        }
    }

    public class Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : Optimizer(learningRate)
    {
        readonly double Beta1 = beta1;
        readonly double Beta2 = beta2;
        readonly double Epsilon = epsilon;

        public override void Step(Number parameter, int iteration)
        {
            parameter.FirstMoment = ((Beta1 * parameter.FirstMoment) + ((1.0 - Beta1) * parameter.Gradient)) / (1.0 - Math.Pow(Beta1, iteration));
            parameter.SecondMoment = ((Beta2 * parameter.SecondMoment) + ((1.0 - Beta2) * Math.Pow(parameter.Gradient, 2.0))) / (1.0 - Math.Pow(Beta2, iteration));

            parameter.Value -= (LR * parameter.FirstMoment) / (Math.Sqrt(parameter.SecondMoment) + Epsilon);
        }
    }

    public class Model
    {
        public Layer[] Layers { get; private set; }

        public Model(Layer[] layers)
        {
            Layers = layers;
        }

        public Model(Layer[] layers, Tensor inputFormat)
        {
            Layers = layers;
            SetUpLayers(inputFormat);
        }

        public Model(Saver.ModelData data)
        {
            Layers = new Layer[data.Layers.Length];

            BuildFromData(data);
        }

        void BuildFromData(Saver.ModelData data)
        {
            Saver.LayerData layerData;
            Type? layerType;
            for (int i = 0; i < data.Layers.Length; i++)
            {
                layerData = data.Layers[i];

                layerType = Type.GetType(layerData.LayerName);
                if (layerType != null)
                {
                    var layer = Activator.CreateInstance(layerType) as Layer;
                    layer?.BuildFromData(layerData);
                    if (layer != null) Layers[i] = layer;
                }
            }
        }

        public void SetUpLayers(Tensor inputFormat)
        {
            int inputs = inputFormat.GetLength(1);
            foreach (var layer in Layers)
            {
                layer.SetUpLayer(inputs);
                inputs = layer.NeuronCount;
            }
        }

        public Tensor Forward(Tensor input)
        {
            var output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public void Optimize(Optimizer optimizer, int iteration)
        {
            foreach (var layer in Layers)
            {
                layer.Optimize(optimizer, iteration + 1);
            }
        }

        public void ZeroGrad()
        {
            foreach (var layer in Layers)
            {
                layer.ZeroGrad();
            }
        }

        public Model Copy()
        {
            var layers = new Layer[Layers.Length];

            for (int i = 0; i < Layers.Length; i++)
            {
                layers[i] = Layers[i].Copy();
            }

            return new(layers);
        }
    }

    public abstract class Layer
    {
        public int NeuronCount { get; protected set; }

        public Layer(int neuronCount)
        {
            NeuronCount = neuronCount;
        }

        public Layer() { }

        public virtual void SetUpLayer(int inputCount)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }

        public virtual void Optimize(Optimizer optimizer, int iteration)
        {
            throw new NotImplementedException();
        }

        public virtual void ZeroGrad()
        {
            throw new NotImplementedException();
        }

        public virtual Layer Copy()
        {
            throw new NotImplementedException();
        }

        public virtual void BuildFromData(Saver.LayerData data)
        {
            throw new NotImplementedException();
        }
    }

    public class Dense : Layer
    {
        public Tensor Weights { get; private set; } = new();
        public Tensor Biases { get; private set; } = new();
        public Activation Activation { get; private set; } = new Linear();

        public Dense(int neuronCount, Activation activation)
        {
            NeuronCount = neuronCount;
            Activation = activation;
        }

        public Dense(int neuronCount, Tensor weights, Tensor biases, Activation activation)
        {
            NeuronCount = neuronCount;
            Weights = weights;
            Biases = biases;
            Activation = activation;
        }

        public Dense() { }

        public override void SetUpLayer(int inputCount)
        {
            Weights = Tensor.InitWeights(inputCount, NeuronCount);
            Biases = Tensor.InitBias(NeuronCount);
        }

        public override Tensor Forward(Tensor input)
        {
            var output = input ^ Weights;
            output += Tensor.Broadcast(Biases, output.GetLength(0));
            output = Activation.Forward(output);
            return output;
        }

        public override void Optimize(Optimizer optimizer, int iteration)
        {
            foreach (var weight in Weights.ToLinearArray())
            {
                optimizer.Step(weight, iteration);
            }
            foreach (var bias in Biases.ToLinearArray())
            {
                optimizer.Step(bias, iteration);
            }
        }

        public override void ZeroGrad()
        {
            foreach (var weight in Weights.ToLinearArray())
            {
                weight.ZeroGradient();
            }
            foreach (var bias in Biases.ToLinearArray())
            {
                bias.ZeroGradient();
            }
        }

        public override Layer Copy()
        {
            return new Dense(NeuronCount, Weights.Copy(), Biases.Copy(), Activation.Copy());
        }

        public override void BuildFromData(Saver.LayerData data)
        {
            NeuronCount = data.NeuronCount;
            Weights = data.Weights;
            Biases = data.Biases;

            var activType = Type.GetType(data.Activation);
            if (activType != null)
            {
                Activation = Activator.CreateInstance(activType) as Activation ?? new Linear();
            }
        }
    }

    public abstract class Activation
    {
        public virtual Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }

        public virtual Activation Copy()
        {
            throw new NotImplementedException();
        }
    }

    public class LeakyReLU(double tau = 0.01) : Activation
    {
        readonly double Tau = tau;

        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                if (input[i].Value >= 0) output[i] = input[i];
                else output[i] = input[i] * Tau;
            }

            return output;
        }

        public override Activation Copy()
        {
            return new LeakyReLU(Tau);
        }
    }

    public class Tanh : Activation
    {
        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Tanh(input[i]);
            }

            return output;
        }

        public override Activation Copy()
        {
            return new Tanh();
        }
    }

    public class Sigmoid : Activation
    {
        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Sigmoid(input[i]);
            }

            return output;
        }

        public override Activation Copy()
        {
            return new Sigmoid();
        }
    }

    public class Linear : Activation
    {
        public override Tensor Forward(Tensor input)
        {
            return input;
        }

        public override Activation Copy()
        {
            return new Linear();
        }
    }

    public abstract class Cost
    {
        public virtual Number CalculateCost(Tensor input, Tensor target)
        {
            throw new NotImplementedException();
        }
    }

    public class MSE : Cost
    {
        public override Number CalculateCost(Tensor input, Tensor target)
        {
            var diff = input - target;
            diff *= diff;
            return Tensor.Mean(diff);
        }
    }

    public class Huber(double delta) : Cost
    {
        readonly double Delta = delta;

        public override Number CalculateCost(Tensor input, Tensor target)
        {
            var diff = input - target;
            for (int i = 0; i < diff.ElementCount; i++)
            {
                if (Math.Abs(diff[i].Value) <= Delta)
                {
                    diff[i] = 0.5 * (diff[i] ^ 2.0);
                }
                else
                {
                    diff[i] = Delta * (Number.Abs(diff[i]) - (0.5 * Delta));
                }
            }

            return Tensor.Mean(diff);
        }
    }

    [Serializable]
    public class Tensor
    {
        public Number[] Data { get; set; }
        public int[] Dimensions { get; set; }
        public int[] Multipliers { get; set; }

        public int Rank => Dimensions.Length;
        public int ElementCount => Data.Length;

        public static Tensor operator +(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] + b[i];
            }

            return output;
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] - b[i];
            }

            return output;
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] * b[i];
            }

            return output;
        }

        public static Tensor operator ^(Tensor a, Tensor b)
        {
            AssertMultiplicationDims(a, b);

            var resultDims = new int[a.Rank];
            for (int i = 0; i < a.Rank - 1; i ++)
            {
                resultDims[i] = a.Dimensions[i];
            }
            resultDims[^1] = b.Dimensions[^1];

            Tensor output = new(resultDims);
            if (a.Rank > 2)
            {
                // Recursively reduce arrays down to batches of 2D matrices

                var reducedA = a.ReduceDimensions();
                var reducedB = b.ReduceDimensions();

                for (int i = 0; i < reducedA.Length; i++)
                {
                    output.InsertSubArray(i, reducedA[i] ^ reducedB[i]);
                }
            }
            else
            {
                // Standard 2D matrix multiplication

                for (int rowA = 0; rowA < output.GetLength(0); rowA++)
                {
                    for (int colB = 0; colB < output.GetLength(1); colB++)
                    {
                        output[rowA, colB] = new(0);
                        for (int i = 0; i < a.GetLength(1); i++)
                        {
                            output[rowA, colB] += a[rowA, i] * b[i, colB];
                        }
                    }
                }
            }

            return output;
        }

        public int MaxIndex()
        {
            int maxIndex = 0;
            double maxValue = this[0].Value;
            for (int i = 1; i < ElementCount; i++)
            {
                if (this[i].Value > maxValue)
                {
                    maxIndex = i;
                    maxValue = this[i].Value;
                }
            }
            return maxIndex;
        }

        public Number Max()
        {
            return this[MaxIndex()];
        }

        public static Number Mean(Tensor input)
        {
            return Number.Mean(input.ToLinearArray());
        }

        public static Tensor InitWeights(int inputCount, int neuronCount)
        {
            Tensor output = new(inputCount, neuronCount);

            double weight;
            for (int i = 0; i < output.ElementCount; i++)
            {
                weight = MathUtils.NextGaussian(0, Math.Sqrt(2.0 / inputCount));
                output[i] = new(weight);
            }

            return output;
        }

        public static Tensor InitBias(int neuronCount)
        {
            Tensor output = new(neuronCount);

            for (int i = 0; i < output.ElementCount; i++)
            {
                output[i] = new(0.01f);
            }

            return output;
        }

        public static Tensor Broadcast(Tensor array, int firstDimLength)
        {
            var outputDims = new int[array.Rank + 1];
            outputDims[0] = firstDimLength;
            for (int i = 1; i < outputDims.Length; i++)
            {
                outputDims[i] = array.Dimensions[i - 1];
            }

            Tensor output = new(outputDims);

            int[] inputIndices;
            int[] outputIndices;
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < array.ElementCount; j++)
                {
                    inputIndices = array.GetFullIndices(j);

                    outputIndices = new int[output.Rank];
                    outputIndices[0] = i;
                    for (int k = 1; k < outputIndices.Length; k++)
                    {
                        outputIndices[k] = inputIndices[k - 1];
                    }

                    output[outputIndices] = array[inputIndices];
                }
            }

            return output;
        }

        public Tensor(params int[] dimensions)
        {
            Dimensions = (int[])dimensions.Clone();
            Multipliers = new int[Dimensions.Length];

            int totalSize = 1;
            for (int i = Rank - 1; i >= 0; i--)
            {
                Multipliers[i] = totalSize;
                totalSize *= Dimensions[i];
            }

            Data = new Number[totalSize];

            for (int i = 0; i < ElementCount; i++)
            {
                this[i] = new(0);
            }
        }

        public int GetLinearIndex(int[] indices)
        {
            int linearIndex = 0;
            for (int i = 0; i < Rank; i++)
            {
                linearIndex += indices[i] * Multipliers[i];
            }
            return linearIndex;
        }

        public int[] GetFullIndices(int index)
        {
            var indices = new int[Rank];

            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = index % Dimensions[i];
                index /= Dimensions[i];
            }

            return indices;
        }

        public Number[] ToLinearArray()
        {
            return Data;
        }

        public override string ToString()
        {
            string output = string.Empty;
            for (int i = 0; i < ElementCount - 1; i++)
            {
                output += $"{Data[i].Value}, ";
            }
            output += Data[^1].Value;
            return output;
        }

        public Tensor[] ReduceDimensions()
        {
            var output = new Tensor[Dimensions[0]];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = ExtractSubArray(i);
            }

            return output;
        }

        Tensor ExtractSubArray(int firstDimIndex)
        {
            var extractDims = new int[Rank - 1];
            for (int i = 1; i < Rank; i++)
            {
                extractDims[i - 1] = Dimensions[i];
            }

            Tensor output = new(extractDims);

            int[] extractIndices;
            int[] parentIndices;
            for (int i = 0; i < output.ElementCount; i++)
            {
                extractIndices = output.GetFullIndices(i);

                parentIndices = new int[extractIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = extractIndices[j - 1];
                }

                output[extractIndices] = this[parentIndices];
            }

            return output;
        }

        public void InsertSubArray(int firstDimIndex, Tensor subArray)
        {
            int[] subIndices;
            int[] parentIndices;
            for (int i = 0; i < subArray.ElementCount; i++)
            {
                subIndices = subArray.GetFullIndices(i);

                parentIndices = new int[subIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = subIndices[j - 1];
                }

                this[parentIndices] = subArray[subIndices];
            }
        }

        public Tensor Copy()
        {
            Tensor output = new(Dimensions);

            for (int i = 0; i < ElementCount; i++)
            {
                output[i] = this[i].Copy();
            }

            return output;
        }

        public static Tensor Transpose(Tensor array, int[]? axes = null)
        {
            if (axes == null)
            {
                axes = new int[array.Rank];
                for (int i = 0; i < array.Rank; i++)
                {
                    axes[i] = array.Rank - 1 - i;
                }
            }

            AssertTranspositionAxes(array.Dimensions, axes);

            var outputDims = RemapIndices(array.Dimensions, axes);

            Tensor output = new(outputDims);

            for (int i = 0; i < array.ElementCount; i++)
            {
                output[RemapIndices(array.GetFullIndices(i), axes)] = array[i];
            }

            return output;
        }

        public static (Tensor normalizedArray, double normalizeFactor) Normalize(Tensor array, double? normalizeFactor = null)
        {
            double maxValue = array[0].Value;
            if (normalizeFactor == null)
            {
                foreach (var value in array.ToLinearArray())
                {
                    maxValue = Math.Max(maxValue, value.Value);
                }
            }
            else maxValue = normalizeFactor.Value;

            Tensor output = new(array.Dimensions);
            for (int i = 0; i < array.ElementCount; i++)
            {
                output[i] = new(array[i].Value / maxValue);
            }

            return (output, maxValue);
        }

        public static Tensor UnnormalizeArray(Tensor array, double normalizeFactor)
        {
            Tensor output = new(array.Dimensions);
            for (int i = 0; i < array.ElementCount; i++)
            {
                output[i] = new(array[i].Value * normalizeFactor);
            }

            return output;
        }

        public static int[] RemapIndices(int[] indices, int[] axes)
        {
            var output = new int[indices.Length];

            for (int i = 0; i < axes.Length; i++)
            {
                output[i] = indices[axes[i]];
            }

            return output;
        }

        public Number this[params int[] indices]
        {
            get => Data[GetLinearIndex(indices)];
            set => Data[GetLinearIndex(indices)] = value;
        }

        public Number this[int trueIndex]
        {
            get => Data[trueIndex];
            set => Data[trueIndex] = value;
        }

        public int GetLength(int dimension) => Dimensions[dimension];

        static void AssertElementwiseDims(Tensor a, Tensor b)
        {
            if (a.Rank != b.Rank) throw new ArgumentException("Array dimensions mismatch");
            else
            {
                for (int i = 0; i < a.Rank; i++)
                {
                    if (a.GetLength(i) != b.GetLength(i)) throw new ArgumentException("Array dimensions mismatch");
                }
            }
        }

        static void AssertMultiplicationDims(Tensor a, Tensor b)
        {
            if (a.Rank != b.Rank) throw new ArgumentException("Invalid array dimensions");
            else
            {
                if (a.Dimensions[^1] != b.Dimensions[^2]) throw new ArgumentException("Invalid array dimensions");
                for (int i = 0; i < a.Rank - 2; i++)
                {
                    if (a.GetLength(i) != b.GetLength(i)) throw new ArgumentException("Invalid array dimensions");
                }
            }
        }

        static void AssertTranspositionAxes(int[] dimensions, int[] axes)
        {
            if (axes.Length != dimensions.Length) throw new ArgumentException("Axes must match array dimensions");
            else
            {
                foreach (var axis in axes)
                {
                    if (axis >= dimensions.Length) throw new ArgumentException("Axes must match array dimensions");
                }
            }
        }
    }

    [Serializable]
    public class Number
    {
        public double Value { get; set; }
        public double Gradient = 0.0;
        public List<Number> DependsOn = [];
        public string CreationOp = "";
        public double FirstMoment = 0.0;
        public double SecondMoment = 0.0;

        public Number(double value, List<Number>? dependsOn = null, string creationOp = "")
        {
            Value = value;
            DependsOn = dependsOn ?? [];
            CreationOp = creationOp;
        }

        public Number() { }

        public static Number operator +(Number a, Number b)
        {
            return new(value: a.Value + b.Value, dependsOn: [a, b], creationOp: "+");
        }

        public static Number operator +(Number a, double b)
        {
            return a + new Number(b);
        }

        public static Number operator +(double a, Number b)
        {
            return new Number(a) + b;
        }

        public static Number operator -(Number a, Number b)
        {
            return new(value: a.Value - b.Value, dependsOn: [a, b], creationOp: "-");
        }

        public static Number operator -(Number a, double b)
        {
            return a - new Number(b);
        }

        public static Number operator -(double a, Number b)
        {
            return new Number(a) - b;
        }

        public static Number operator *(Number a, Number b)
        {
            return new(value: a.Value * b.Value, dependsOn: [a, b], creationOp: "*");
        }

        public static Number operator *(Number a, double b)
        {
            return a * new Number(b);
        }

        public static Number operator *(double a, Number b)
        {
            return new Number(a) * b;
        }

        public static Number operator /(Number a, Number b)
        {
            return new(value: a.Value / b.Value, dependsOn: [a, b], creationOp: "/");
        }

        public static Number operator /(Number a, double b)
        {
            return a / new Number(b);
        }

        public static Number operator /(double a, Number b)
        {
            return new Number(a) / b;
        }

        public static Number operator ^(Number a, Number b)
        {
            return new(value: Math.Pow(a.Value, b.Value), dependsOn: [a, b], creationOp: "^");
        }

        public static Number operator ^(Number a, double b)
        {
            return a ^ new Number(b);
        }

        public static Number operator ^(double a, Number b)
        {
            return new Number(a) ^ b;
        }

        public static Number Abs(Number a)
        {
            return new(value: Math.Abs(a.Value), dependsOn: [a], creationOp: "abs");
        }

        public void Backward()
        {
            List<Number> topography = [];
            HashSet<Number> visited = [];

            BuildTopography(topography, visited);

            foreach (var node in topography) node.Gradient = 0.0;
            Gradient = 1.0;

            for (int i = topography.Count - 1; i >= 0; i--)
            {
                topography[i].ApplyBackward();
            }
        }

        void BuildTopography(List<Number> topography, HashSet<Number> visited)
        {
            if (!visited.Contains(this))
            {
                visited.Add(this);

                foreach (var parent in DependsOn)
                {
                    parent.BuildTopography(topography, visited);
                }

                topography.Add(this);
            }
        }

        void ApplyBackward()
        {
            switch (CreationOp)
            {
                case "+":
                    DependsOn[0].Gradient += Gradient;
                    DependsOn[1].Gradient += Gradient;
                    break;
                case "-":
                    DependsOn[0].Gradient += Gradient;
                    DependsOn[1].Gradient -= Gradient;
                    break;
                case "*":
                    DependsOn[0].Gradient += Gradient * DependsOn[1].Value;
                    DependsOn[1].Gradient += Gradient * DependsOn[0].Value;
                    break;
                case "/":
                    DependsOn[0].Gradient += Gradient * (1.0 / DependsOn[1].Value);
                    DependsOn[1].Gradient += Gradient * (-DependsOn[0].Value / Math.Pow(DependsOn[1].Value, 2.0));
                    break;
                case "^":
                    DependsOn[0].Gradient += Gradient * DependsOn[1].Value * Math.Pow(DependsOn[0].Value, DependsOn[1].Value - 1.0);
                    DependsOn[1].Gradient += Gradient * Math.Pow(DependsOn[0].Value, DependsOn[1].Value) * Math.Log(DependsOn[0].Value);
                    break;
                case "abs":
                    DependsOn[0].Gradient += Gradient * Math.Sign(DependsOn[0].Value);
                    break;
            }
        }

        public void ZeroGradient()
        {
            Gradient = 0;
        }

        public override string ToString()
        {
            return $"Value = {Value}; Gradient = {Gradient}";
        }

        public Number Copy()
        {
            return new(value: Value);
        }

        public static Number Max(Number a, Number b)
        {
            return (a.Value > b.Value) ? a : b;
        }

        public static Number Min(Number a, Number b)
        {
            return (a.Value < b.Value) ? a : b;
        }

        public static Number Mean(Number[] inputs)
        {
            Number sum = new(0);
            foreach (var input in inputs)
            {
                sum += input;
            }
            return sum * (1f / inputs.Length);
        }

        public static Number Tanh(Number x)
        {
            return ((MathF.E ^ (2 * x)) - 1) / ((MathF.E ^ (2 * x)) + 1);
        }

        public static Number Sigmoid(Number x)
        {
            return 1 / (1 + (MathF.E ^ (-1 * x)));
        }
    }

    public static class MathUtils
    {
        static readonly Random random = new();

        public static double NextGaussian()
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return randStdNormal;
        }

        public static double NextGaussian(double mean, double stdDev)
        {
            double randStdNormal = NextGaussian();
            double randNormal = mean + stdDev * randStdNormal;
            return randNormal;
        }

        public static int RoundToInterval(double value, int interval)
        {
            return (int)Math.Round(value / interval, MidpointRounding.AwayFromZero) * interval;
        }
    }

    public static class Saver
    {
        const string Extension = ".nnn";

#pragma warning disable CS8604 // Possible null reference argument.
        public static void SaveModel(Model model, string fileName)
        {
            fileName += Extension;

            var layers = new LayerData[model.Layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                var layer = model.Layers[i];

                switch (layer)
                {
                    case Dense dense:
                        LayerData layerData = new(dense.NeuronCount, dense.GetType().AssemblyQualifiedName, dense.Weights,
                            dense.Biases, dense.Activation.GetType().AssemblyQualifiedName);
                        layers[i] = layerData;

                        break;
                }
            }

            ModelData modelData = new(layers);

            string json = JsonSerializer.Serialize(modelData);

            File.WriteAllText(fileName, json);
        }

        public static Model LoadModel(string fileName)
        {
            fileName += Extension;

            string json = File.ReadAllText(fileName);

            var modelData = JsonSerializer.Deserialize<ModelData>(json);

            Model model = new(modelData);

            return model;
        }
#pragma warning restore CS8604 // Possible null reference argument.

        public static bool FileExists(string fileName)
        {
            fileName += Extension;

            if (File.Exists(fileName)) return true;
            else return false;
        }

        [Serializable]
        public class ModelData(LayerData[] layers)
        {
            public LayerData[] Layers { get; set; } = layers;
        }

        [Serializable]
        public class LayerData(int neuronCount, string layerName, Tensor weights, Tensor biases, string activation)
        {
            public int NeuronCount { get; set; } = neuronCount;
            public string LayerName { get; set; } = layerName;
            public Tensor Weights { get; set; } = weights;
            public Tensor Biases { get; set; } = biases;
            public string Activation { get; set; } = activation;
        }
    }
}