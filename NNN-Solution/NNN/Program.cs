using NNN;

Model model;
NNN.Environment env = new MovementGrid2D(-5, 5, -5, 5);
double exploration = 1.0;
double explorationDecay = 0.995;
double minExploration = 0.1;
double discount = 0.99;
Optimizer optimizer = new Adam(0.0001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 64;
double tau = 0.005;
double maxGradNorm = 1.0;
int minExperiences = 1000;
int episodeMemorySize = 100;
DQNTrainer dqnTrainer;
FIFOBuffer<Episode> episodeBuffer = new(episodeMemorySize);

Dictionary<UserInput, string> userInputs = new() { { UserInput.Yes, "y" }, { UserInput.No, "n" }, { UserInput.Quit, "q" } };

InteractionLoop();

#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning disable CS8602 // Dereference of a possibly null reference.
void InteractionLoop()
{
    Console.WriteLine("Welcome to the DQN Training Terminal (Enter Q to quit)");

    if (GetInput("Load model from file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        string fileName = GetFileName();
        model = Saver.LoadModel(fileName);
        exploration = 0.01;
    }
    else
    {
        model = new([
            new Dense(32, new LeakyReLU()),
            new Dense(32, new LeakyReLU()),
            new Dense(4, new Linear())
        ], new Tensor(1, env.StateSize));
    }

    dqnTrainer = new(
        agent: model,
        environment: env,
        actionCount: env.ActionCount,
        exploration: exploration,
        explorationDecay: explorationDecay,
        minExploration: minExploration,
        discount: discount,
        optimizer: optimizer,
        cost: cost,
        replayBufferSize: replayBufferSize,
        batchSize: batchSize,
        tau: tau,
        maxGradNorm: maxGradNorm,
        minExperiences: minExperiences
    );

    TrainingLoop();

    if (GetInput("Save model to a file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
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
        if (GetInput("Run DQN Training episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            int episodes = GetInteger("Enter number of episodes to train");
            Console.WriteLine($"Training for {episodes} episodes...");
            dqnTrainer.Train(ref episodeBuffer, episodes);

            TestDQNModel();

            ViewEpisodes();
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
    Tensor trueState;
    Tensor batchState = new(1, state.Dimensions[0]);
    bool done = false;
    double totalReward = 0;
    int steps = 0;

    List<Experience> episodeExperiences = [];
    while (!done && steps < 50)
    {
        steps++;
        trueState = env.GetState();
        batchState.InsertSubArray(0, state);
        int action = model.Forward(batchState).MaxIndex();
        var (reward, nextState, isDone) = env.Step(action, steps);

        totalReward += reward;
        state = nextState;
        done = isDone;

        episodeExperiences.Add(new(trueState, action, reward, env.GetState(), done));
    }

    episodeBuffer?.Add(new(episodeExperiences));

    var logState = env.GetState();
    Console.WriteLine($"Test Finished in {steps} steps.");
    Console.WriteLine($"Total Reward: {totalReward:F2}");
    Console.WriteLine($"Starting State: ({startState[0].Value}, {startState[1].Value})");
    Console.WriteLine($"Final State: ({logState[0].Value}, {logState[1].Value}), Target: ({logState[2].Value}, {logState[3].Value})");
}

void ViewEpisodes()
{
    if (GetInput("Replay past episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        while (true)
        {
            Console.WriteLine();
            int episode = GetEpisodeSelection();

            int step = 0;
            bool viewingEpisode = true;
            while (viewingEpisode)
            {
                Console.Clear();
                env.Render(episodeBuffer[episode], step);

                var input = Console.ReadKey(true).Key;

                switch(input)
                {
                    case (ConsoleKey)EpisodeNavigation.Next:
                        step = Math.Min(step + 1, episodeBuffer[episode].Experiences.Count);
                        break;
                    case (ConsoleKey)EpisodeNavigation.Previous:
                        step = Math.Max(step - 1, 0);
                        break;
                    case (ConsoleKey)EpisodeNavigation.Exit:
                        Console.Clear();
                        viewingEpisode = false;
                        break;
                    case (ConsoleKey)EpisodeNavigation.Quit:
                        System.Environment.Exit(0);
                        break;
                }
            }

            if (GetInput("View another episode? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) break;
        }
    }
}

int GetEpisodeSelection()
{
    while (true)
    {
        int index = GetInteger($"Enter episode number ({episodeBuffer.Count} episodes cached)");
        if (index > 0 && index <= episodeBuffer.Count) return index - 1;
        else Console.WriteLine("Invalid episode number");
    }
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

        if (input == userInputs[UserInput.Quit]) System.Environment.Exit(0);
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
    while (true)
    {
        if (int.TryParse(GetInput(prompt), out int integer)) return integer;
        else Console.WriteLine("\nNot a valid number");
    }
}

void SaveLoop()
{
    string fileName;

    while (true)
    {
        fileName = GetInput("Enter file name");
        if (Saver.FileExists(fileName))
        {
            if (GetInput($"File with name \"{fileName}\" already exists. Overwrite existing file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
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
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning restore CS8602 // Dereference of a possibly null reference.

enum UserInput { Yes, No, Quit }

enum EpisodeNavigation
{
    Previous = ConsoleKey.LeftArrow,
    Next = ConsoleKey.RightArrow,
    Exit = ConsoleKey.Escape,
    Quit = ConsoleKey.Q
}

namespace NNN
{
    using System.Diagnostics;
    using System.Text.Json;
    using System.Linq;

    public abstract class Environment
    {
        public virtual int StateSize => throw new NotImplementedException();
        public virtual int ActionCount => throw new NotImplementedException();

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

        public virtual void Render(Episode episode, int step)
        {
            throw new NotImplementedException();
        }
    }

    public class MovementGrid2D : Environment
    {
        public override int StateSize => 4;
        public override int ActionCount => 4;
        readonly Random random = new();
        readonly Tensor State = new(4); // current x, current y, target x, target y
        readonly int[] Bounds; // xMin, xMax, yMin, yMax
        readonly int MaxSteps;
        readonly double XRange;
        readonly double YRange;

        public MovementGrid2D(int xMin, int xMax, int yMin, int yMax, int maxSteps = 50)
        {
            State[0] = new(random.Next(xMin, xMax + 1));
            State[1] = new(random.Next(yMin, yMax + 1));
            State[2] = new(random.Next(xMin, xMax + 1));
            State[3] = new(random.Next(yMin, yMax + 1));
            Bounds = [xMin, xMax, yMin, yMax];
            XRange = xMax - xMin;
            YRange = yMax - yMin;
            MaxSteps = maxSteps;
        }

        public override Tensor GetNormalizedState()
        {
            Tensor normalized = new(4);

            double xPos = State[0].Value / (XRange / 2.0);
            double yPos = State[1].Value / (YRange / 2.0);

            double xTarget = State[2].Value / (XRange / 2.0);
            double yTarget = State[3].Value / (YRange / 2.0);

            (normalized[0], normalized[1], normalized[2], normalized[3]) =
                (new(xPos), new(yPos), new(xTarget), new(yTarget));

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
                case (int)Action.Left: // left
                    State[0].Value--;
                    break;
                case (int)Action.Right: // right
                    State[0].Value++;
                    break;
                case (int)Action.Up: // up
                    State[1].Value++;
                    break;
                case (int)Action.Down: // down
                    State[1].Value--;
                    break;
            }

            xDiff = State[2].Value - State[0].Value;
            yDiff = State[3].Value - State[1].Value;
            double newDist = Math.Sqrt(Math.Pow(xDiff, 2.0) + Math.Pow(yDiff, 2.0));
            double deltaDist = prevDist - newDist;

            bool reachedTarget = (State[0].Value == State[2].Value && State[1].Value == State[3].Value);
            bool outOfBounds = (State[0].Value < Bounds[0]) || (State[0].Value > Bounds[1]) ||
                               (State[1].Value < Bounds[2]) || (State[1].Value > Bounds[3]);
            bool outOfSteps = steps >= MaxSteps && !reachedTarget;

            if (outOfBounds)
            {
                State[0].Value = Math.Clamp(State[0].Value, Bounds[0], Bounds[1]);
                State[1].Value = Math.Clamp(State[1].Value, Bounds[2], Bounds[3]);
            }

            bool done = reachedTarget || outOfBounds || outOfSteps;

            double reward = 2.0 * deltaDist;
            reward += reachedTarget ? 50.0 : 0.0;
            reward -= outOfBounds ? 5.0 : 0.0;
            reward -= outOfSteps ? 5.0 : 0.0;

            return (reward, GetNormalizedState(), done);
        }

        public override void Reset()
        {
            State[0] = new(random.Next(Bounds[0], Bounds[1] + 1));
            State[1] = new(random.Next(Bounds[2], Bounds[3] + 1));
            State[2] = new(random.Next(Bounds[0], Bounds[1] + 1));
            State[3] = new(random.Next(Bounds[2], Bounds[3] + 1));
        }

        public override void Render(Episode episode, int step)
        {
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
            var state = step == episode.Experiences.Count ? exp.NextState : exp.State;
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);

            for (int y = Bounds[3] + 1; y >= Bounds[2] - 1; y--)
            {
                bool yEdge = y == Bounds[3] + 1 || y == Bounds[2] - 1;

                for (int x = Bounds[0] - 1; x <= Bounds[1] + 1; x++)
                {
                    bool xEdge = x == Bounds[0] - 1 || x == Bounds[1] + 1;

                    if (xEdge && yEdge)
                    {
                        Console.Write("+");
                        continue;
                    }
                    else if (xEdge)
                    {
                        Console.Write("|");
                        continue;
                    }
                    else if (yEdge)
                    {
                        Console.Write("-");
                        continue;
                    }

                    if (x == state[0].Value && y == state[1].Value)
                    {
                        Console.Write("A");
                    }
                    else if (x == state[2].Value && y == state[3].Value)
                    {
                        Console.Write("T");
                    }
                    else
                    {
                        Console.Write(" ");
                    }
                }
                Console.Write("\n");
            }

            Console.Write($"Step: {step}, Action: {(Enum.IsDefined(typeof(Action), action) ? ((Action)action).ToString() : "None")}, Reward: {reward:F3}");
        }

        enum Action { Left, Right, Up, Down }
    }

    public record Episode
    {
        public List<Experience> Experiences { get; init; }

        public Episode(List<Experience> experiences)
        {
            Experiences = [.. experiences.Select(e => new Experience(e.State.Copy(), e.Action, e.Reward, e.NextState.Copy(), e.Done))];
        }
    }

    public record Experience
    {
        public Tensor State { get; init; }
        public int Action { get; init; } 
        public double Reward { get; init; }
        public Tensor NextState { get; init; }
        public bool Done { get; init; }
        public double Priority { get; set; }

        public Experience(Tensor state, int action, double reward, Tensor nextState, bool done, double priority = 1.0)
        {
            State = state.Copy();
            Action = action;
            Reward = reward;
            NextState = nextState.Copy();
            Done = done;
            Priority = Math.Max(priority, 1e-8);
        }
    }

    public class FIFOBuffer<T>(int maxSize)
    {
        readonly protected int MaxSize = maxSize;
        readonly protected List<T> Buffer = [];
        protected int FirstIndex = 0;
        public int Count => Buffer.Count;

        public virtual void Add(T item)
        {
            if (Count < MaxSize) Buffer.Add(item);
            else
            {
                Buffer[FirstIndex] = item;
                FirstIndex = (FirstIndex + 1) % MaxSize;
            }
        }

        public T this[int index]
        {
            get => Buffer[(FirstIndex + index) % Count];
        }
    }

    public class ReplayBuffer(int maxSize, double alpha = 0.6) : FIFOBuffer<Experience>(maxSize)
    {
        readonly Random random = new();
        readonly double Alpha = alpha;
        double Beta = 0.4;
        readonly double BetaIncrement = 0.001;

        public override void Add(Experience item)
        {
            double maxPriority = Count > 0 ? Buffer.Max(e => e.Priority) : 1.0;
            item.Priority = maxPriority;

            base.Add(item);
        }

        public (List<Experience> batch, double[] weights) GetBatch(int batchSize)
        {
            double[] priorities = [.. Buffer.Select(e => Math.Pow(e.Priority, Alpha))];
            double sum = priorities.Sum();

            List<Experience> batch = [];
            var weights = new double[batchSize];

            double maxWeight = 0.0;

            double r;
            double cumulative;
            for (int i = 0; i < batchSize; i++)
            {
                r = random.NextDouble() * sum;
                cumulative = 0.0;

                for (int j = 0; j < Count; j++)
                {
                    cumulative += priorities[j];
                    if (cumulative >= r)
                    {
                        batch.Add(Buffer[j]);

                        double prob = priorities[j] / sum;
                        double weight = Math.Pow(1.0 / (Count * prob), Beta);

                        weights[i] = weight;
                        maxWeight = Math.Max(maxWeight, weight);
                        break;
                    }
                }
            }

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] /= maxWeight;
            }

            Beta = Math.Min(1.0, Beta + BetaIncrement);

            return (batch, weights);
        }
    }

    public class DQNTrainer(Model agent, Environment environment, int actionCount, Optimizer optimizer, Cost cost, double discount = 0.995, double exploration = 1.0,
        double explorationDecay = 0.99, double minExploration = 0.01, int replayBufferSize = 10000, int batchSize = 64, double tau = 0.005, double maxGradNorm = 1.0,
           int minExperiences = 1000)
    {
        readonly Random random = new();
        readonly Model Agent = agent;
        readonly Model TargetModel = agent.Copy();
        readonly Environment Environment = environment;
        readonly int ActionCount = actionCount;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;
        readonly ReplayBuffer ReplayBuffer = new(replayBufferSize);
        readonly int BatchSize = batchSize;
        readonly double Discount = discount;
        double Exploration = exploration;
        readonly double ExplorationDecay = explorationDecay;
        readonly double MinExploration = minExploration;
        int optimizerSteps = 0;
        readonly double Tau = tau;
        readonly double MaxNorm = maxGradNorm;
        readonly int MinExperiences = minExperiences;
        double totalLoss = 0.0;

        public void Train(ref FIFOBuffer<Episode>? episodeBuffer, int episodes = 1000)
        {
            List<Experience> episodeExperiences = [];
            Tensor state;
            Tensor initialState;
            Tensor trueState;
            Tensor logState;
            bool done;
            int action;
            double reward;
            double totalReward;
            int step;
            Tensor nextState;
            int[] movementMagnitude = new int[2];
            TimeSpan avgElapsed = new(0);
            Stopwatch stopwatch = new();
            stopwatch.Start();
            for (int e = 0; e < episodes; e++)
            {
                totalLoss = 0.0;
                episodeExperiences.Clear();
                Environment.Reset();
                (movementMagnitude[0], movementMagnitude[1]) = (0, 0);
                state = Environment.GetNormalizedState();
                initialState = Environment.GetState();

                done = false;
                step = 0;
                totalReward = 0;
                while (!done)
                {
                    step++;
                    trueState = Environment.GetState();
                    action = PickNextAction(state);

                    switch (action)
                    {
                        case 0:
                        case 1:
                            movementMagnitude[0]++;
                            break;
                        case 2:
                        case 3:
                            movementMagnitude[1]++;
                            break;
                    }

                    (reward, nextState, done) = Environment.Step(action, step);
                    totalReward += reward;
                    ReplayBuffer.Add(new(state, action, reward, nextState, done));
                    episodeExperiences.Add(new(trueState, action, reward, Environment.GetState(), done));

                    TrainNetwork();

                    state = nextState;
                }

                episodeBuffer?.Add(new(episodeExperiences));

                Exploration = Math.Max(Exploration * ExplorationDecay, MinExploration);

                var elapsed = stopwatch.Elapsed;
                avgElapsed += (elapsed - avgElapsed) / (e + 1);
                var eta = avgElapsed * (episodes - e - 1);

                logState = Environment.GetState();
                Console.WriteLine($"\nEpisode {e + 1}/{episodes} finished...");
                Console.WriteLine($"Initial State: ({initialState[0].Value}, {initialState[1].Value}), Final State: ({logState[0].Value}, {logState[1].Value}), Target: (" +
                    $"{initialState[2].Value}, {initialState[3].Value}), Steps Taken: {step}, Total Reward: {totalReward:F2},");
                Console.WriteLine($"Total Movement Magnitude: ({movementMagnitude[0]}, {movementMagnitude[1]}), Average Loss: {(totalLoss / step):F3}");
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

            var (batch, weights) = ReplayBuffer.GetBatch(BatchSize);
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

            var predictedQs = MaskQValues(predictions, batch);
            var targetQs = MaskQValuesDouble(nextAgentQs, nextTargetQs, batch);

            Agent.ZeroGrad();

            var lossResult = Cost.CalculateCostWithPriority(predictedQs, targetQs, weights);
            for (int i = 0; i < BatchSize; i++)
            {
                batch[i].Priority = lossResult.Priorities[i];
            }
            var loss = Tensor.Mean(lossResult.Losses);
            totalLoss += loss.Value;

            loss.Backward();
            Agent.ClipGradients(MaxNorm);

            for (int i = 0; i < Agent.ParameterCount; i++)
            {
                Optimizer.Step(Agent.Parameters[i], optimizerSteps);
            }

            for (int i = 0; i < TargetModel.ParameterCount; i++)
            {
                TargetModel.Parameters[i].Value = (Tau * Agent.Parameters[i].Value) + ((1.0 - Tau) * TargetModel.Parameters[i].Value);
            }

            optimizerSteps++;
        }

        Tensor MaskQValues(Tensor qValues, List<Experience> batch)
        {
            Tensor maskedQs = new(BatchSize, 1);

            for (int i = 0; i < BatchSize; i++)
            {
                maskedQs[i] = qValues[i, batch[i].Action];
            }

            return maskedQs;
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

                foreach (var param in Model.Parameters)
                {
                    Optimizer.Step(param, epochs);
                }

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
            iteration++;
            parameter.FirstMoment = (Beta1 * parameter.FirstMoment) + ((1.0 - Beta1) * parameter.Gradient);
            parameter.SecondMoment = (Beta2 * parameter.SecondMoment) + ((1.0 - Beta2) * Math.Pow(parameter.Gradient, 2.0));

            double mHat = parameter.FirstMoment / (1.0 - Math.Pow(Beta1, iteration));
            double vHat = parameter.SecondMoment / (1.0 - Math.Pow(Beta2, iteration));

            parameter.Value -= (LR * mHat) / (Math.Sqrt(vHat) + Epsilon);
        }
    }

    public class Model
    {
        public Layer[] Layers { get; private set; }
        public List<Number> Parameters
        {
            get
            {
                parameters ??= [.. GetParameters()];
                return parameters;
            }
        }
        List<Number>? parameters;
        public int ParameterCount => Parameters.Count;

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
            InvalidateParameters();
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

        public IEnumerable<Number> GetParameters()
        {
            foreach (var layer in Layers)
            {
                foreach (var param in layer.GetParameters())
                {
                    yield return param;
                }
            }
        }

        public void ClipGradients(double maxNorm)
        {
            double totalNorm = 0.0;

            foreach (var param in Parameters)
            {
                totalNorm += Math.Pow(param.Gradient, 2.0);
            }

            totalNorm = Math.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / (totalNorm + 1e-8);

                foreach (var param in Parameters)
                {
                    param.Gradient *= scale;
                }
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

        void InvalidateParameters()
        {
            parameters = null;
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

        public virtual IEnumerable<Number> GetParameters()
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

        public override IEnumerable<Number> GetParameters()
        {
            foreach (var weight in Weights.ToLinearArray())
            {
                yield return weight;
            }
            
            foreach (var bias in Biases.ToLinearArray())
            {
                yield return bias;
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

    public class LeakyReLU : Activation
    {
        readonly double Tau;

        public LeakyReLU(double tau = 0.01)
        {
            Tau = tau;
        }

        public LeakyReLU() { }

        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.LeakyReLU(input[i], Tau);
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

    public record CostResult(Tensor Losses, double[] Priorities);

    public abstract class Cost
    {
        public virtual Number CalculateCost(Tensor input, Tensor target)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor CalculatePerSampleCost(Tensor input, Tensor target)
        {
            throw new NotImplementedException();
        }

        public virtual CostResult CalculateCostWithPriority(Tensor input, Tensor target, double[]? weights = null)
        {
            var losses = CalculatePerSampleCost(input, target);
            var priorities = new double[losses.ElementCount];

            for (int i = 0; i < losses.ElementCount; i++)
            {
                double value = losses[i].Value;
                priorities[i] = Math.Abs(value) + 1e-8;

                if (weights != null) losses[i] *= weights[i];
            }

            return new(losses, priorities);
        }
    }

    public class MSE : Cost
    {
        public override Number CalculateCost(Tensor input, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(input, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor input, Tensor target)
        {
            var diff = input - target;
            diff *= diff;
            return diff;
        }
    }

    public class Huber(double delta = 1.0) : Cost
    {
        readonly double Delta = delta;

        public override Number CalculateCost(Tensor input, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(input, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor input, Tensor target)
        {
            var diff = input - target;

            Tensor costs = new(diff.Dimensions);
            for (int i = 0; i < diff.ElementCount; i++)
            {
                costs[i] = Math.Pow(Delta, 2.0) * (((1.0 + ((diff[i] / Delta) ^ 2.0)) ^ 0.5) - 1.0);
            }

            return costs;
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
        public Func<List<double>, double>? Op;
        public Func<List<double>, List<double>>? BackwardOp;
        public double FirstMoment = 0.0;
        public double SecondMoment = 0.0;

        public Number(double value, List<Number>? dependsOn = null, Func<List<double>, double>? op = null, Func<List<double>, List<double>>? backwardOp = null)
        {
            Value = value;
            DependsOn = dependsOn ?? [];
            Op = op;
            BackwardOp = backwardOp;
        }

        public Number() { }

        public static Number operator +(Number a, Number b)
        {
            return new(value: a.Value + b.Value, dependsOn: [a, b], op: inputs => inputs[0] + inputs[1], backwardOp: inputs => [1.0, 1.0]);
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
            return new(value: a.Value - b.Value, dependsOn: [a, b], op: inputs => inputs[0] - inputs[1], backwardOp: inputs => [1.0, -1.0]);
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
            return new(value: a.Value * b.Value, dependsOn: [a, b], op: inputs => inputs[0] * inputs[1], backwardOp: inputs => [inputs[1], inputs[0]]);
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
            return new(value: a.Value / b.Value, dependsOn: [a, b], op: inputs => inputs[0] / inputs[1],
                backwardOp: inputs => [1.0 / inputs[1], -inputs[0] / Math.Pow(inputs[1], 2.0)]);
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
            return new(value: Math.Pow(a.Value, b.Value), dependsOn: [a, b], op: inputs => Math.Pow(inputs[0], inputs[1]),
                backwardOp: inputs => [inputs[1] * Math.Pow(inputs[0], inputs[1] - 1.0), Math.Pow(inputs[0], inputs[1]) * Math.Log(inputs[0])]);
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
            return new(value: Math.Abs(a.Value), dependsOn: [a], op: inputs => Math.Abs(inputs[0]), backwardOp: inputs => [Math.Sign(inputs[0])]);
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
            return sum * (1.0 / inputs.Length);
        }

        public static Number Tanh(Number x)
        {
            double value = MathUtils.Tanh(x.Value);

            return new(
                value: value,
                dependsOn: [x],
                op: inputs => MathUtils.Tanh(inputs[0]),
                backwardOp: inputs => [1.0 - Math.Pow(MathUtils.Tanh(inputs[0]), 2.0)]
            );
        }

        public static Number Sigmoid(Number x)
        {
            double value = MathUtils.Sigmoid(x.Value);

            return new(
                value: value,
                dependsOn: [x],
                op: inputs => MathUtils.Sigmoid(inputs[0]),
                backwardOp: inputs =>
                {
                    double sig = MathUtils.Sigmoid(inputs[0]);
                    return [sig * (1.0 - sig)];
                }
            );
        }

        public static Number LeakyReLU(Number x, double tau)
        {
            double value = Math.Max(tau * x.Value, x.Value);

            return new(
                value: value,
                dependsOn: [x],
                op: inputs => Math.Max(tau * inputs[0], inputs[0]),
                backwardOp: inputs => [inputs[0] > 0.0 ? 1.0 : tau]
            );
        }

        public static Number Log(Number logBase, Number arg)
        {
            double value = Math.Log(arg.Value, logBase.Value);
            return new(
                value: value,
                dependsOn: [logBase, arg],
                op: inputs => Math.Log(inputs[1], inputs[0]),
                backwardOp: inputs => [-(Math.Log(inputs[1]) / (inputs[0] * Math.Pow(Math.Log(inputs[0]), 2.0))),
                    1.0 / (inputs[1] * Math.Log(inputs[0]))]
            );
        }

        public void Backward()
        {
            List<Number> topography = [];
            HashSet<Number> visited = [];

            BuildTopography(topography, visited);

            foreach (var node in topography) node.Gradient = 0.0;
            Gradient = 1.0;

            topography.Reverse();

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
            if (DependsOn.Count > 0 && BackwardOp != null)
            {
                var grads = BackwardOp(DependsOn.ConvertAll(d => d.Value));

                for (int i = 0; i < DependsOn.Count; i++)
                {
                    DependsOn[i].Gradient += Gradient * grads[i];
                }
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

        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public static double Tanh(double value)
        {
            return Math.Tanh(value);
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