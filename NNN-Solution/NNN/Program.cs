using NNN;
using static NNN.UIUtils;

bool demoMode = false;

Model model;
NNN.Environment env = new Snake();
double exploration = 1.0;
double explorationDecay = 0.9995;
double minExploration = 0.01;
int trainEvery = 4;
double discount = 0.99;
Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 64;
int agentBufferSize = 5;
int opponentCopyRate = 600;
int minRandomOpponentEpisodes = 600;
double tau = 0.01;
double maxGradNorm = 1.0;
int minExperiences = 2000;
int episodeMemorySize = 100;
DQNTrainer dqnTrainer;
FIFOBuffer<Episode> episodeBuffer = new(episodeMemorySize);

if (demoMode) DemoHandler.RunDemo();
else InteractionLoop();

#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning disable CS8602 // Dereference of a possibly null reference.
// Primary loop for model training UI - entry point
void InteractionLoop()
{
    Console.WriteLine("Welcome to the DQN Training Terminal (Enter Q to quit)");

    if (GetInput("Load model from file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        // Load model from file
        string fileName = GetFileName();
        model = Saver.LoadModel(fileName);
        exploration = minExploration;
    }
    else
    {
        // Create a new model
        model = new([
            new Conv(16, [3, 3], new LeakyReLU()),
            new Conv(32, [3, 3], new LeakyReLU()),
            new Dense(256, new LeakyReLU()),
            new Dense(env.ActionCount, new Linear())
        ], env.StateFormat);
    }

    dqnTrainer = new(
        agent: model,
        environment: env,
        exploration: exploration,
        explorationDecay: explorationDecay,
        minExploration: minExploration,
        trainEvery: trainEvery,
        discount: discount,
        optimizer: optimizer,
        cost: cost,
        replayBufferSize: replayBufferSize,
        agentBufferSize: agentBufferSize,
        batchSize: batchSize,
        opponentCopyRate: opponentCopyRate,
        minRandomOpponentEpisodes: minRandomOpponentEpisodes,
        tau: tau,
        maxGradNorm: maxGradNorm,
        minExperiences: minExperiences
    );

    TrainingLoop();

    if (GetInput("Save model to a file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        SaveLoop(model);
    }

    Console.WriteLine("\nPress any key to quit...");
    Console.ReadKey();
    System.Environment.Exit(0);
}

// UI loop for training models for a specified number of episodes
void TrainingLoop()
{
    // Train agent until user indicates to stop
    while (true)
    {
        if (GetInput("Run DQN Training episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            // Train agent for a given number of episodes
            int episodes = GetInteger("Enter number of episodes to train");
            Console.WriteLine($"Training for {episodes} episodes...");
            dqnTrainer.Train(ref episodeBuffer, episodes);

            TestDQNModel();

            ViewEpisodes();
        }
        else break;
    }
}

// Test the performance of the current trained model through running a single episode with no exploration
void TestDQNModel()
{
    Console.WriteLine("\n--- Testing Agent Performance ---");
    env.Reset();
    var state = env.GetNormalizedState();
    Tensor trueState;
    bool done = false;
    double totalReward = 0;
    int steps = 0;

    List<Experience> episodeExperiences = [];

    // Run a single complete episode with no exploration
    while (!done)
    {
        steps++;
        trueState = env.GetState();
        int action = env.PickAgentAction(model.Predict(Tensor.WrapBatch(state)));
        var (reward, nextState, isDone) = env.Step(action, steps);

        totalReward += reward;
        state = nextState;
        done = isDone;

        episodeExperiences.Add(new(trueState, action, reward, env.GetState(), done));
    }

    episodeBuffer?.Add(new(episodeExperiences));

    Console.WriteLine($"Total Reward: {totalReward:F2}");
    env.Render(episodeBuffer[^1], steps);

    if (env is TicTacToe ticTacToe && GetInput("Play against model? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        ticTacToe.Play(model);
    }

    if (env is Snake snake && GetInput("Watch agent play? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        snake.Play(model);
    }
}

// UI loop for viewing and navigating through past training episodes
void ViewEpisodes()
{
    if (GetInput("Replay past episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        // Allow user to view past episodes until user indicates to stop
        while (true)
        {
            Console.WriteLine();
            int episode = GetEpisodeSelection(episodeBuffer!);

            int step = 0;
            bool viewingEpisode = true;
            while (viewingEpisode)
            {
                // Render current selected step of the episode
                Console.Clear();
                env.Render(episodeBuffer[episode], step);

                // Navigate through episode based on user input
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

#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning restore CS8602 // Dereference of a possibly null reference.

/// <summary>
/// Key mappings for user navigation during training episode review.
/// </summary>
enum EpisodeNavigation
{
    Previous = ConsoleKey.LeftArrow,
    Next = ConsoleKey.RightArrow,
    Exit = ConsoleKey.Escape,
    Quit = ConsoleKey.Q
}

// Neural Network Nonsense library for training neural networks
namespace NNN
{
    using ILGPU.IR.Types;
    using ILGPU.Runtime;
    using ILGPU.Runtime.Cuda;
    using System.Buffers;
    using System.Data;
    using System.Diagnostics;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.InteropServices;
    using System.Text.Json;
    using System.Threading.Tasks.Dataflow;
    using TorchSharp.Modules;

    /// <summary>
    /// Interface for DQN environments in which the agent plays against itself during training.
    /// </summary>
    public interface ISelfPlay
    {
        // Base environment data

        /// <summary>
        /// Current environment state.
        /// </summary>
        public Tensor State { get; init; }

        // Opponent agent data

        /// <summary>
        /// Whether the agent is to move this turn.
        /// </summary>
        public bool AgentTurn { get; set; }
        /// <summary>
        /// Number of stored opponent agents.
        /// </summary>
        public int OpponentCount { get; set; }
        /// <summary>
        /// Index of the current opponent agent.
        /// </summary>
        public int OpponentIndex { get; set; } // index >= opponent count -> random actions

        // Utilities

        /// <summary>
        /// Environment's Random instance.
        /// </summary>
        public Random Random { get; init; }

        /// <summary>
        /// Picks the action to be taken by the current opponent.
        /// </summary>
        /// <param name="agents">Buffer of all stored opponent agents.</param>
        /// <returns>Index of the action taken by the opponent.</returns>
        public int PickOpponentAction(FIFOBuffer<Model> agents)
        {
            if (agents.Count != OpponentCount) UpdateOpponentIndex(agents.Count); // update opponent data if buffer expanded

            // Get action from current opponent
            if (OpponentIndex >= OpponentCount)
            {
                return PickRandomAction();
            }
            else
            {
                return GetAgentAction(agents[OpponentIndex]);
            }
        }

        /// <summary>
        /// Updates opponent data and selects a new opponent.
        /// </summary>
        /// <param name="newOpponentCount">Optional number of currently stored opponents.</param>
        void UpdateOpponentIndex(int? newOpponentCount = null)
        {
            OpponentCount = newOpponentCount is not null ? newOpponentCount.Value : OpponentCount;
            OpponentIndex = Random.Next(OpponentCount + 1); // +1 to allow for an opponent which always chooses actions randomly
        }

        /// <summary>
        /// Gets the action taken by the given agent given the environment state.
        /// </summary>
        /// <param name="agent">Agent being used to select the action.</param>
        /// <param name="state">Optional tensor representing the given state if context differs from environment's current state.</param>
        /// <returns>Index of the action selected by the agent.</returns>
        public int GetAgentAction(Model agent, Tensor? state = null);

        /// <summary>
        /// Picks the action the agent will take given predicted Q-Values and the current state.
        /// </summary>
        /// <param name="qValues">Tensor representing the Q-Values predicted by the agent.</param>
        /// <param name="state">Optional tensor representing the state if context differs from the environment's current state.</param>
        /// <returns>Index of the action the agent will take.</returns>
        public int PickAgentAction(Tensor qValues, Tensor? state = null);

        /// <summary>
        /// Picks a random valid action given the environment's current state.
        /// </summary>
        /// <returns>Index of the chosen action.</returns>
        public int PickRandomAction();
    }

    /// <summary>
    /// Base class for DQN training environments.
    /// </summary>
    public abstract class Environment
    {
        /// <summary>
        /// The number of elements present in the environment's state representation.
        /// </summary>
        public int StateSize => StateFormat.ElementCount;
        /// <summary>
        /// The format of the tensor used to represent a batch of the environment's states.
        /// </summary>
        public abstract Tensor StateFormat { get; }
        /// <summary>
        /// The number of discrete actions the agent can make.
        /// </summary>
        public abstract int ActionCount { get; }
        /// <summary>
        /// The name of the environment to display to the user.
        /// </summary>
        public abstract string EnvironmentName { get; }

        /// <summary>
        /// Get the normalized form of the environment's current state.
        /// </summary>
        /// <returns>Tensor containing the current normalized state.</returns>
        public abstract Tensor GetNormalizedState();

        /// <summary>
        /// Get the raw current state of the environment.
        /// </summary>
        /// <returns>Tensor containing the current raw state.</returns>
        public abstract Tensor GetState();

        /// <summary>
        /// Resets the environment to its initial state and prepares it for a new episode.
        /// </summary>
        public abstract void Reset();

        /// <summary>
        /// Picks the action the agent will take given predicted Q-Values and the current state.
        /// </summary>
        /// <param name="qValues">Tensor representing the Q-Values predicted by the agent.</param>
        /// <param name="state">Optional tensor representing the state if context differs from the environment's current state.</param>
        /// <returns>Index of the action the agent will take.</returns>
        public abstract int PickAgentAction(Tensor qValues, Tensor? state = null);

        /// <summary>
        /// Picks a random valid action given the environment's current state.
        /// </summary>
        /// <returns>Index of the chosen action.</returns>
        public abstract int PickRandomAction();

        /// <summary>
        /// Checks whether the given action is valid in the context of the state.
        /// </summary>
        /// <param name="action">Index of the action being checked.</param>
        /// <param name="state">Optional tensor representing the state if context differs from the environment's current state.</param>
        /// <returns>Whether the action is valid in the given state.</returns>
        public abstract bool ValidAction(int action, Tensor? state);

        /// <summary>
        /// Performs a single step in the environment.
        /// </summary>
        /// <param name="action">The action being taken.</param>
        /// <param name="steps">The total number of steps which have been taken.</param>
        /// <returns>Reward from the action, tensor reprsenting the normalized state after the action, and whether the current episode has finished.</returns>
        public abstract (double reward, Tensor nextState, bool done) Step(int action, int steps);

        /// <summary>
        /// Displays a previous recorded state in the console.
        /// </summary>
        /// <param name="episode">Episode in which the state is recorded.</param>
        /// <param name="step">Step at which the state occurred in the episode.</param>
        public abstract void Render(Episode episode, int step);

        /// <summary>
        /// Plays a demonstration of the model trained on the environment.
        /// </summary>
        public abstract void PlayDemo();
    }

    /// <summary>
    /// 2D grid environment for an agent to navigate between two points.
    /// </summary>
    public class MovementGrid2D : Environment
    {
        // Base Environment API overrides
        public override Tensor StateFormat => new([1, 4]); // encodes agent's and target's grid positions
        public override int ActionCount => 4; // agent can move in one of the 4 cardinal directions
        public override string EnvironmentName => "2D Movement Grid";

        // Internal grid environment representation
        /// <summary>
        /// State encoding agent and target grid positions.
        /// </summary>
        readonly Tensor State = new([4]); // agent x, agent y, target x, target y
        /// <summary>
        /// Bounds of the environment's grid.
        /// </summary>
        readonly int[] Bounds; // xMin, xMax, yMin, yMax
        /// <summary>
        /// Width of the environment's grid.
        /// </summary>
        readonly double XRange;
        /// <summary>
        /// Height of the environment's grid.
        /// </summary>
        readonly double YRange;

        // Training parameters
        /// <summary>
        /// Maximum number of steps the agent can take during an episode.
        /// </summary>
        readonly int MaxSteps;

        // Utilities
        /// <summary>
        /// Environment's Random instance.
        /// </summary>
        readonly Random Random = new();
        /// <summary>
        /// Action index mapping.
        /// </summary>
        enum Action { Left, Right, Up, Down }

        /// <summary>
        /// Creates a new 2D movement grid environment instance.
        /// </summary>
        /// <param name="xMin">Left edge bound of the grid.</param>
        /// <param name="xMax">Right edge bound of the grid.</param>
        /// <param name="yMin">Bottom edge bound of the grid.</param>
        /// <param name="yMax">Top edge bound of the grid.</param>
        /// <param name="maxSteps">Maximum number of steps the agent can take during an episode.</param>
        public MovementGrid2D(int xMin, int xMax, int yMin, int yMax, int maxSteps = 50)
        {
            Bounds = [xMin, xMax, yMin, yMax];
            XRange = xMax - xMin;
            YRange = yMax - yMin;
            MaxSteps = maxSteps;
            Reset(); // generate the initial state for the first episode
        }

        // Base Environment API overrides

        public override Tensor GetNormalizedState()
        {
            Tensor normalized = new([4]);

            // Normalize positions to ~ between -1 and 1 for a symmetrical grid

            double xPos = State[0] / (XRange / 2.0);
            double yPos = State[1] / (YRange / 2.0);

            double xTarget = State[2] / (XRange / 2.0);
            double yTarget = State[3] / (YRange / 2.0);

            (normalized[0], normalized[1], normalized[2], normalized[3]) =
                (xPos, yPos, xTarget, yTarget);

            return normalized;
        }

        public override Tensor GetState()
        {
            return State.Copy();
        }

        public override void Reset()
        {
            // Randomly generate initial agent and target positions
            State[0] = Random.Next(Bounds[0], Bounds[1] + 1);
            State[1] = Random.Next(Bounds[2], Bounds[3] + 1);
            State[2] = Random.Next(Bounds[0], Bounds[1] + 1);
            State[3] = Random.Next(Bounds[2], Bounds[3] + 1);
        }

        public override int PickAgentAction(Tensor qValues, Tensor? state = null) => Tensor.ArgMax(qValues); // no invalid actions - return highest Q-Value

        public override int PickRandomAction() => Random.Next(ActionCount); // no invalid actions - return random action index

        public override bool ValidAction(int action, Tensor? state) => true; // no invalid actions

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            // Calculate distance to target before moving
            double xDiff = State[2] - State[0];
            double yDiff = State[3] - State[1];
            double prevDist = Math.Sqrt(xDiff * xDiff + yDiff * yDiff);

            // Move the agent's position based on action mapping
            switch (action)
            {
                case (int)Action.Left:
                    State[0]--;
                    break;
                case (int)Action.Right:
                    State[0]++;
                    break;
                case (int)Action.Up:
                    State[1]++;
                    break;
                case (int)Action.Down:
                    State[1]--;
                    break;
                default:
                    throw new ArgumentException("Invalid Action");
            }

            // Calculate distance to target after moving
            xDiff = State[2] - State[0];
            yDiff = State[3] - State[1];
            double newDist = Math.Sqrt(xDiff * xDiff + yDiff * yDiff);
            double deltaDist = prevDist - newDist; // change in distance to target

            bool reachedTarget = (State[0] == State[2] && State[1] == State[3]);
            bool outOfBounds = (State[0] < Bounds[0]) || (State[0] > Bounds[1]) ||
                               (State[1] < Bounds[2]) || (State[1] > Bounds[3]);
            bool outOfSteps = steps >= MaxSteps && !reachedTarget;

            bool done = reachedTarget || outOfBounds || outOfSteps;

            // Calculate reward for the action
            double reward = 5.0 * deltaDist; // shaped reward based on change in distance to target
            reward += reachedTarget ? 100.0 : 0.0; // reward for reaching the target
            reward -= outOfBounds ? 100.0 : 0.0; // penalty for going out of bounds
            reward -= outOfSteps ? 5.0 : 0.0; // penalty for exceeding step limit

            return (reward, GetNormalizedState(), done);
        }

        public override void Render(Episode episode, int step)
        {
            // Extract the state at the given step from the episode
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
            var state = step == episode.Experiences.Count ? exp.NextState : exp.State;
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);

            // Draw the grid for the given state
            for (int y = Bounds[3] + 1; y >= Bounds[2] - 1; y--)
            {
                bool yEdge = y == Bounds[3] + 1 || y == Bounds[2] - 1;

                for (int x = Bounds[0] - 1; x <= Bounds[1] + 1; x++)
                {
                    bool xEdge = x == Bounds[0] - 1 || x == Bounds[1] + 1;

                    if (xEdge && yEdge)
                    {
                        Console.Write("+"); // draw corner
                        continue;
                    }
                    else if (xEdge)
                    {
                        Console.Write("|"); // draw left/right edge
                        continue;
                    }
                    else if (yEdge)
                    {
                        Console.Write("-"); // draw top/bottom edge
                        continue;
                    }

                    if (x == state[0] && y == state[1])
                    {
                        Console.Write("A"); // draw agent
                    }
                    else if (x == state[2] && y == state[3])
                    {
                        Console.Write("T"); // draw target
                    }
                    else
                    {
                        Console.Write(" "); // draw empty space
                    }
                }
                Console.Write("\n");
            }

            Console.Write($"Step: {step}, Action: {(Enum.IsDefined(typeof(Action), action) ? ((Action)action).ToString() : "None")}, Reward: {reward:F3}");
        }

        public override void PlayDemo() => throw new NotImplementedException();
    }

    /// <summary>
    /// Tic-Tac-Toe game environment.
    /// </summary>
    public class TicTacToe : Environment, ISelfPlay
    {
        // Base Environment API overrides
        public override Tensor StateFormat => new([1, 10]); // encodes state of all 9 board positions + current player to move
        public override int ActionCount => 9; // agent can select one of the 9 board positions
        public override string EnvironmentName => "Tic-Tac-Toe";

        // Self-play interface API overrides
        public bool AgentTurn { get; set; } = true;
        public int OpponentCount { get; set; }
        public int OpponentIndex { get; set; }

        // Internal board representation
        /// <summary>
        /// State encoding the positions of the board and current player to move.
        /// </summary>
        public Tensor State { get; init; } = new([10]); // positions index 0-8, index 9 - current player to move: 1 -> X, 0 -> empty, -1 -> O
        /// <summary>
        /// Array of all possible winning position index combinations.
        /// </summary>
        static readonly int[][] WinOrients = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

        // Training parameters
        /// <summary>
        /// Maximum number of steps the agent can take during an episode.
        /// </summary>
        readonly int MaxSteps = 9; // at most 9 turns can happen in one game of Tic-Tac-Toe
        /// <summary>
        /// Base reward for a winning action.
        /// </summary>
        const double WinRewardBase = 1.0;
        /// <summary>
        /// Base reward for an action which prevents the opponent from winning.
        /// </summary>
        const double BlockRewardBase = 0.8;
        /// <summary>
        /// Base reward for an action which ties the game - optimal outcome if both players play optimally.
        /// </summary>
        const double DrawRewardBase = 0.15;
        /// <summary>
        /// Penalty for actions which disadvantage the agent.
        /// </summary>
        const double Penalty = 0.0;

        // Utilities
        /// <summary>
        /// Environment's Random instance.
        /// </summary>
        public Random Random { get; init; } = new();

        // Demo
        /// <summary>
        /// Name of the file containing the demonstration agent for the environment.
        /// </summary>
        const string DemoFileName = "tictactoedemo";

        /// <summary>
        /// Creates a new TicTacToe environment instance.
        /// </summary>
        public TicTacToe() { }

        // Base Environment API overrides

        public override Tensor GetNormalizedState()
        {
            return GetState(); // no normalization necessary - all values already between -1 and 1
        }

        public override Tensor GetState()
        {
            return State.Copy();
        }

        public override void Reset()
        {
            AgentTurn = Random.Next(2) == 1; // randomly pick agent to play X or O
            (this as ISelfPlay).UpdateOpponentIndex(); // select a new opponent agent for the next episode

            // Reset all positions to empty
            for (int i = 0; i < State.ElementCount - 1; i++)
            {
                State[i] = 0.0;
            }
            State[9] = 1.0; // set player to move to X
        }

        public override int PickAgentAction(Tensor agentQValues, Tensor? state = null)
        {
            state ??= State; // assume current environment state if no state is given

            var qValues = agentQValues.Copy();

            // Find valid action with highest Q-Value
            int action = Tensor.ArgMax(qValues);
            while (!ValidAction(action, state) && !BoardFilled(state))
            {
                qValues[action] = double.MinValue;
                action = Tensor.ArgMax(qValues);
            }
            return action;
        }

        public override int PickRandomAction()
        {
            // Find indices of all empty positions
            List<int> validActions = [];
            for (int i = 0; i < State.ElementCount - 1; i++)
            {
                if (State[i] == 0.0) validActions.Add(i);
            }
            return validActions[Random.Next(validActions.Count)]; // select a random empty position
        }

        public override bool ValidAction(int action, Tensor? state = null)
        {
            state ??= State; // assume current environment state if no state is given
            return (action != state.ElementCount - 1) && (state[action] == 0.0); // ensure action index is within valid range and the position at the action index is empty
        }

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            if (!ValidAction(action)) throw new ArgumentException("Invalid Action"); // ensure action being taken is valid

            State[action] = State[9] == 1.0 ? 1.0 : -1.0; // fill position at the action index with current player's encoding
            var (reward, done) = EvaluateAction(action); // evaluate the reward of the action
            
            // Flip current player
            AgentTurn = !AgentTurn;
            State[9] *= -1.0;

            var nextState = GetNormalizedState();

            done = done || BoardFilled() || steps >= MaxSteps;

            return (reward, nextState, done);
        }

        public override void Render(Episode episode, int step)
        {
            // Extract the state at the given step from the episode
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
            var state = step == episode.Experiences.Count ? exp.NextState : exp.State;
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);

            DrawState(state);

            Console.WriteLine($"\n\nPosition Taken: {action}, Reward: {reward}");
        }

        public override void PlayDemo()
        {
            ShowDemoInstructions();
            var agent = Saver.LoadModel(DemoFileName);
            Play(agent);
        }

        // Additional self-play interface API overrides

        public int GetAgentAction(Model agent, Tensor? state = null)
        {
            state ??= State; // assume current environment state if no state is given
            return PickAgentAction(agent.Predict(Tensor.WrapBatch(state)), state);
        }

        // Additional environment-specific functionality

        /// <summary>
        /// Evaluates the reward of the given action.
        /// </summary>
        /// <param name="action">Index of action which was taken.</param>
        /// <returns>Reward of the action and whether the episode has finished.</returns>
        (double reward, bool won) EvaluateAction(int action)
        {
            var relevantOrients = WinOrients.Where(o => o.Contains(action)).ToArray(); // find all winning orientations affected by the action

            // Extract position encoding values of all relevant winning orientations
            double[][] orientValues = new double[relevantOrients.Length][];
            for (int orient = 0; orient < relevantOrients.Length; orient++)
            {
                orientValues[orient] = new double[relevantOrients[orient].Length];
                for (int pos = 0; pos < relevantOrients[orient].Length; pos++)
                {
                    orientValues[orient][pos] = State[relevantOrients[orient][pos]];
                }
            }

            // Determine encoding values of current acting player and opponent
            double ownValue = State[9] == 1.0 ? 1.0 : -1.0;
            double oppValue = -ownValue;

            var advantOrients = orientValues.Where(o => !o.Contains(oppValue)); // find all orientations where the acting player can win
            var blockOrients = orientValues.Where(o => o.Contains(oppValue) && !o.Contains(ownValue)); // find all orientations where the acting player blocked the opponent from winning
            var falseOrients = orientValues.Where(o => o.Contains(ownValue) && o.Contains(oppValue)); // find all orientations where the action had no impact

            bool boardFilled = BoardFilled();
            double reward = boardFilled ? 0.0 : Penalty * falseOrients.Count(); // add penalty per unimpactful orientations
            bool won = false;

            // Calculate reward for orientations where the acting player is closer to winning
            foreach (var orient in advantOrients)
            {
                int ownPositions = orient.Count(p => p == ownValue);
                reward += ownPositions switch
                {
                    2 => 0.1 * WinRewardBase, // small reward for filling 2/3 positions in the orientation
                    3 => WinRewardBase, // reward for winning the orientation
                    _ => 0.0
                };
                won = won || ownPositions == 3; // track whether acting player has won
            }

            // Calculate reward for orientations where the acting player blocked the opponent from winning
            foreach (var orient in blockOrients)
            {
                int oppPositions = orient.Count(p => p == oppValue);
                reward += oppPositions == 2 ? BlockRewardBase : 0.0;
            }

            reward += (boardFilled && !won) ? DrawRewardBase : 0.0; // add reward if the game has ended with a draw

            return (reward, won);
        }

        /// <summary>
        /// Plays a game between the user and the given agent.
        /// </summary>
        /// <param name="agent">Agent for the user to play against.</param>
        public void Play(Model agent)
        {
            bool playing = true;
            while (playing)
            {
                Reset();
                bool playerTurn = Random.Next(2) == 0; // randomly select user to play X or O

                string winner = "Draw";

                bool done = false;
                while (!done)
                {
                    Console.Clear();
                    DrawState(State);

                    // Get the current acting player's action
                    int action = playerTurn ? GetPlayerAction() : GetAgentAction(agent);
                    if (action == -1) break;

                    State[action] = State[9] == 1.0 ? 1.0 : -1.0; // update board encoding with acting player's action

                    // Check whether the acting player has won
                    if (CheckWin())
                    {
                        winner = State[9] == 1.0 ? "X" : "O";
                        break;
                    }

                    done = BoardFilled();

                    // Flip current acting player
                    State[9] *= -1.0;
                    playerTurn = !playerTurn;
                }

                // Draw final board state
                Console.Clear();
                DrawState(State);
                Console.WriteLine($"\n\nWinner: {winner}");

                playing = GetInput("Play again? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes];
            }
        }

#pragma warning disable CS8602 // Dereference of a possibly null reference.
        /// <summary>
        /// Gets the action which the user has chosen to take.
        /// </summary>
        /// <returns>Index of the action taken by the user.</returns>
        int GetPlayerAction()
        {
            string input = string.Empty;
            Console.WriteLine();
            while (input != "q")
            {
                Console.WriteLine("\nEnter desired position:");
                input = Console.ReadLine().ToLowerInvariant();

                // Ensure input is a valid action index
                if (int.TryParse(input, out int action) && ValidAction(action)) return action;
                else if (input == "q") break;

                Console.WriteLine("Invalid position...");
            }
            return -1;
        }
#pragma warning restore CS8602 // Dereference of a possibly null reference.

        /// <summary>
        /// Checks whether the board is completely filled.
        /// </summary>
        /// <param name="state">Optional tensor representing the state if context differs from the environment's current state.</param>
        /// <returns>Whether the board is completely filled.</returns>
        bool BoardFilled(Tensor? state = null)
        {
            state ??= State; // assume current environment state if no state is given
            return !state.Data.Any(p => p == 0.0);
        }

        /// <summary>
        /// Checks whether either player has won.
        /// </summary>
        /// <param name="state">Optional tensor representing the state if context differs from the environment's current state.</param>
        /// <returns>Whether either player has won.</returns>
        bool CheckWin(Tensor? state = null)
        {
            state ??= State; // assume current environment state if no state is given
            return WinOrients.Any(o => o.All(p => state[p] == (state[9] == 1.0 ? 1.0 : -1.0))); // check whether any winning orientation is fully filled with either player's encoding
        }

        /// <summary>
        /// Draws the given state to the console.
        /// </summary>
        /// <param name="state">Tensor representing the state to be drawn.</param>
        static void DrawState(Tensor state)
        {
            for (int i = 0; i < state.ElementCount - 1; i++) // ignore last state encoding (current acting player encoding)
            {
                if (i % 3 == 0) Console.WriteLine(); // start a new row after every 3 columns

                // Fill the position based on its encoding
                string fill = state[i] switch
                {
                    1.0 => " X ",
                    -1.0 => " O ",
                    _ => "   "
                };
                Console.Write(fill);
            }
        }

        static void ShowDemoInstructions()
        {
            Console.WriteLine("Welcome to the Tic-Tac-Toe agent demonstration.");
            Console.WriteLine("The agent contains a total of 329 neurons.");
            Console.WriteLine("These are arranged in two layers of 128 neurons each, one layer of 64 neurons, and an output layer of 9 neurons - one for each position on the board.");
            Console.WriteLine("The agent receives a total of 10 inputs, one encoding each position on the board and a tenth encoding whether X or O is to move.");
            Console.WriteLine("This agent was trained over the course of around 50,000 games of Tic-Tac-Toe.");
            Console.WriteLine("During these games, it learned to play both X and O, and trained by playing against past versions of itself.");
            Console.WriteLine("It is by no means perfect, but it can play relatively well in most situations.");
            Console.WriteLine("I have personally found a few board states in which it simply fails.");
            Console.WriteLine("Can you find them as well?\n");
            Console.WriteLine("The positions on the game board are represented by the indices 0-8.");
            Console.WriteLine("The indices increase across rows and then down columns, with position 0 being in the top left and position 8 in the bottom right.");
            Console.WriteLine("When playing, simply select the position you would like to take.\n");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }

    /// <summary>
    /// Snake game environment.
    /// </summary>
    public class Snake : Environment
    {
        // Base Environment API overrides
        public override Tensor StateFormat => new([1, GridDims.Y, GridDims.X, 8]); // 20 x 20 grid with one-hot encoding for contents of each cell and the direction each cell's content will move in
        public override int ActionCount => 3; // agent can move forward, left, or right
        public override string EnvironmentName => "Snake";

        // Internal game representation
        /// <summary>
        /// Dimensions of the environment's game grid.
        /// </summary>
        Int2 GridDims;
        /// <summary>
        /// Head node of the snake.
        /// </summary>
        SnakeNode SnakeHead = new(); // snake represented as a linked list
        /// <summary>
        /// Current length of the snake.
        /// </summary>
        int SnakeLength = 1;
        /// <summary>
        /// Current position of the apple.
        /// </summary>
        Int2 ApplePosition = new();

        // Training parameters
        /// <summary>
        /// Maximum steps an episode can last.
        /// </summary>
        const int MaxSteps = 10000;
        /// <summary>
        /// Steps since an apple was last eaten.
        /// </summary>
        int StepsWithoutApple = 0;
        /// <summary>
        /// Current maximum number of steps the agent can go without eating an apple.
        /// </summary>
        int MaxStepsWithoutApple = InitMaxStepsWithoutApple;
        /// <summary>
        /// Initial maximum number of steps the agent can go without eating an apple.
        /// </summary>
        const int InitMaxStepsWithoutApple = 200;
        /// <summary>
        /// Reward for eating an apple.
        /// </summary>
        const double AppleReward = 20.0;
        /// <summary>
        /// Multiplier for the additional apple reward based on snake length.
        /// </summary>
        const double LengthRewardMult = 1.0;
        /// <summary>
        /// Multiplier for the shaped distance to apple reward.
        /// </summary>
        const double DistRewardMult = 0.2;
        /// <summary>
        /// Multiplier for the shaped reward based on number of reachable positions.
        /// </summary>
        const double ReachableRewardMult = 0.4;
        /// <summary>
        /// Penalty for not reaching the next apple in time.
        /// </summary>
        const double TimeoutPenalty = -1.0;
        /// <summary>
        /// Penalty for colliding with the border or snake body.
        /// </summary>
        const double CollisionPenalty = -5.0;
        /// <summary>
        /// Penalty for each step taken.
        /// </summary>
        const double StepPenalty = -0.1;

        // Utilities
        /// <summary>
        /// Environment's Random instance.
        /// </summary>
        readonly Random Random = new();
        /// <summary>
        /// Time per frame when showing the agent playing the game.
        /// </summary>
        const int FrameTime = 100; // in milliseconds

        // Demo
        /// <summary>
        /// Name of the file containing the demonstration agent for the environment.
        /// </summary>
        const string DemoFileName = "snakedemo";

        /// <summary>
        /// Encoding of agent's actions.
        /// </summary>
        enum Actions { Left, Forward, Right }

        /// <summary>
        /// Encoding of cardinal directions within the game's grid.
        /// </summary>
        enum Movements { Left, Up, Right, Down }

        /// <summary>
        /// Encoding of board elements in the state representation.
        /// </summary>
        struct BoardEncoding
        {
            public const double Empty = 0.0;
            public const double Head = 1.0;
            public const double Body = 2.0;
            public const double Tail = 3.0;
            public const double Apple = 4.0;
        }

        /// <summary>
        /// One-hot encoding indices of board position data.
        /// </summary>
        enum BoardEncodingOneHot { Head, Body, Tail, Apple, Left, Up, Right, Down }

        /// <summary>
        /// Creates a new Snake environment instance.
        /// </summary>
        /// <param name="width">Width of the game's grid.</param>
        /// <param name="height">Height of the game's grid.</param>
        public Snake(int width = 20, int height = 20)
        {
            GridDims = new(width, height);
            Reset(); // prepare the first episode
        }

        // Base Environment API overrides

        public override Tensor GetNormalizedState()
        {
            return GetState();
        }

        public override Tensor GetState()
        {
            Tensor state = new(StateFormat.Dimensions[1..]);

            // Encode the state based on the internal game representation
            int[] contentIndices = new int[3];
            var contSpan = contentIndices.AsSpan();

            int[] directionIndices = new int[3];
            var dirSpan = directionIndices.AsSpan();

            SnakeNode? node = SnakeHead;
            while (node is not null)
            {
                if (ValidPosition(node.Position))
                {
                    OneHotEncodeNode(node, contSpan, dirSpan);
                    state[contentIndices] = 1.0;
                    state[directionIndices] = 1.0;
                }

                node = node.Child;
            }

            state[ApplePosition.Y, ApplePosition.X, (int)BoardEncodingOneHot.Apple] = 1.0;

            return state;
        }

        public override void Reset()
        {
            StepsWithoutApple = 0;
            MaxStepsWithoutApple = InitMaxStepsWithoutApple;
            SnakeLength = 1;

            // Generate new starting snake position and direction
            int startX = Random.Next(0, GridDims.X);
            int startY = Random.Next(0, GridDims.Y);
            int startDir = Random.Next(0, 4);

            SnakeHead = new(x: startX, y: startY) { Direction = startDir };

            GenerateApple(); // generate first apple
        }

        public override int PickAgentAction(Tensor qValues, Tensor? state = null) => Tensor.ArgMax(qValues); // no invalid actions - return highest Q-Value

        public override int PickRandomAction() => Random.Next(ActionCount); // no invalid actions - return random action index

        public override bool ValidAction(int action, Tensor? state) => true; // no invalid actions

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            if (steps >= MaxSteps) return (0.0, GetNormalizedState(), true); // end episode if step limit exceeded

            // Calculate distance from agent to apple before moving
            double xDiff = ApplePosition.X - SnakeHead.Position.X;
            double yDiff = ApplePosition.Y - SnakeHead.Position.Y;
            double prevDist = Math.Sqrt(xDiff * xDiff + yDiff * yDiff);

            var prevState = GetNormalizedState();

            // Move the snake in the given direction
            int dir = MapAction(action);
            SnakeHead.Move(dir);

            // Calculate reward of the action

            double reward = StepPenalty; // apply per-step penalty

            // Add reward for eating the apple
            if (AteApple())
            {
                reward += AppleReward + LengthRewardMult * SnakeLength;
                StepsWithoutApple = 0;
                return (reward, GetNormalizedState(), false);
            }
            else if (Collided()) // add penalty for colliding with the border or snake body and end the episode
            {
                reward += CollisionPenalty;
                return (reward, prevState, true);
            }

            StepsWithoutApple++;

            // End episode if agent has taken too many steps to eat the apple
            if (StepsWithoutApple >= MaxStepsWithoutApple) return (TimeoutPenalty, GetNormalizedState(), true);

            // Calculate distance from agent to the apple after moving
            xDiff = ApplePosition.X - SnakeHead.Position.X;
            yDiff = ApplePosition.Y - SnakeHead.Position.Y;
            double newDist = Math.Sqrt(xDiff * xDiff + yDiff * yDiff);

            reward += DistRewardMult * (prevDist - newDist); // add shaped reward based on change in distance

            // Find number of reachable positions
            double reachable = ReachablePositions(SnakeHead.Position, BlockedCells()) / (double)(GridDims.X * GridDims.Y - SnakeLength);
            reward += ReachableRewardMult * reachable; // add shaped reward based on number of reachable positions

            return (reward, GetNormalizedState(), false);
        }

        public override void Render(Episode episode, int step)
        {
            // Extract the state at the given step from the episode
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);
            string dirMoved = action switch
            {
                (int)Actions.Forward => "Forward",
                (int)Actions.Left => "Left",
                (int)Actions.Right => "Right",
                _ => "Invalid Action Made"
            };

            // Only display basic information about the action taken due to state representation being insufficient to reconstruct the exact board state
            Console.WriteLine($"\nDirection Moved: {dirMoved}, Step: {step}, Reward: {reward}");
        }

        public override void PlayDemo()
        {
            ShowDemoInstructions();
            var agent = Saver.LoadModel(DemoFileName);
            Play(agent);
        }

        // Additional environment-specific functionality

        /// <summary>
        /// Maps the agent's forward/left/right action index to a cardinal direction on the game's grid.
        /// </summary>
        /// <param name="action">Action index to map.</param>
        /// <returns>Index of the corresponding cardinal direction on the game's grid.</returns>
        /// <exception cref="ArgumentException">Action index outside valid range.</exception>
        int MapAction(int action)
        {
            return action switch
            {
                (int)Actions.Forward => SnakeHead.Direction,
                (int)Actions.Left => (SnakeHead.Direction + 3) % 4,
                (int)Actions.Right => (SnakeHead.Direction + 1) % 4,
                _ => throw new ArgumentException("Invalid Action")
            };
        }

        /// <summary>
        /// Finds the one-hot encoding of a snake node.
        /// </summary>
        /// <param name="node">Snake node to encode.</param>
        /// <param name="contentIndices">Span to write one-hot encoding indices for the state content encoding to.</param>
        /// <param name="directionIndices">Span to write one-hot encoding indices for the state direction encoding to.</param>
        static void OneHotEncodeNode(SnakeNode node, Span<int> contentIndices, Span<int> directionIndices)
        {
            contentIndices[0] = node.Position.Y;
            contentIndices[1] = node.Position.X;
            contentIndices.CopyTo(directionIndices);

            if (node.Parent is not null && node.Child is not null)
            {
                contentIndices[2] = (int)BoardEncodingOneHot.Body;
            }
            else if (node.Parent is not null)
            {
                contentIndices[2] = (int)BoardEncodingOneHot.Tail;
            }
            else
            {
                contentIndices[2] = (int)BoardEncodingOneHot.Head;
            }

            directionIndices[2] = node.Direction switch
            {
                (int)Movements.Left => (int)BoardEncodingOneHot.Left,
                (int)Movements.Up => (int)BoardEncodingOneHot.Up,
                (int)Movements.Right => (int)BoardEncodingOneHot.Right,
                (int)Movements.Down => (int)BoardEncodingOneHot.Down,
                _ => throw new Exception("Cannot encode - invalid direction")
            };
        }

        /// <summary>
        /// Generates a new apple on the game's board.
        /// </summary>
        void GenerateApple()
        {
            var state = GetBoardState(); // get the current game board

            // Find all positions not occupied by the snake
            List<int> validPositions = [];
            for (int i = 0; i < state.ElementCount; i++)
            {
                if (state[i] == BoardEncoding.Empty) validPositions.Add(i);
            }

            // Select a random empty position index
            int linearPos = validPositions[Random.Next(validPositions.Count)]; // generate random index in linear position list
            var arrayPos = state.GetFullIndices(linearPos); // convert linear index to grid coordinates -> (row, column)

            ApplePosition = new(arrayPos[1], arrayPos[0]); // column -> x, row -> y
        }

        /// <summary>
        /// Checks whether the snake has eaten the current apple.
        /// </summary>
        /// <returns>Whether the snake ate the current apple.</returns>
        bool AteApple()
        {
            if (SnakeHead.Position == ApplePosition)
            {
                SnakeHead.Grow();
                SnakeLength++;
                MaxStepsWithoutApple++;
                GenerateApple();
                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Checks whether the snake has collided with an obstacle.
        /// </summary>
        /// <returns>Whether the snake has collided with an obstacle.</returns>
        bool Collided()
        {
            return HitWall() || HitBody();
        }

        /// <summary>
        /// Checks whether the snake has collided with the game grid's border.
        /// </summary>
        /// <returns>Whether the snake has collided with a border.</returns>
        bool HitWall()
        {
            return SnakeHead.Position.X < 0 || SnakeHead.Position.X >= GridDims.X || SnakeHead.Position.Y < 0 || SnakeHead.Position.Y >= GridDims.Y;
        }

        /// <summary>
        /// Checks whether the snake has collided with its own body.
        /// </summary>
        /// <returns>Whether the snake has collided with itself.</returns>
        bool HitBody()
        {
            // Compare the head's position with all body positions
            SnakeNode? node = SnakeHead.Child;
            while (node is not null)
            {
                if (SnakeHead.Position == node.Position) return true;
                node = node.Child;
            }
            return false;
        }

        /// <summary>
        /// Counts the number of positions reachable from the given position.
        /// </summary>
        /// <param name="from">Position from which to search.</param>
        /// <param name="blocked">HashSet of all blocked positions.</param>
        /// <returns>Number of unique reachable positions.</returns>
        int ReachablePositions(Int2 from, HashSet<Int2> blocked)
        {
            HashSet<Int2> visited = []; // HashSet of unique visited positions
            Queue<Int2> queue = []; // Queue of positions to visit
            queue.Enqueue(from); // add starting position to the queue

            // Iterate through all positions in the Queue
            while (queue.Count > 0)
            {
                var pos = queue.Dequeue(); // get the next position from the queue
                if (!visited.Add(pos)) continue; // skip if already visited position

                // Add each adjacent unblocked position to the queue
                foreach (var neighbor in GetNeighbors(pos))
                {
                    if (!blocked.Contains(neighbor)) queue.Enqueue(neighbor);
                }
            }

            return visited.Count - 1; // discard initial position
        }

        /// <summary>
        /// Get all of the positions adjacent to a given position.
        /// </summary>
        /// <param name="pos">Center position of which to get neighbors.</param>
        /// <returns>List of all adjacent positions.</returns>
        List<Int2> GetNeighbors(Int2 pos)
        {
            // Generate all coordinates adjacent linearly
            List<Int2> neighbors = [new(pos.X - 1, pos.Y), new(pos.X, pos.Y - 1), new(pos.X + 1, pos.Y), new(pos.X, pos.Y + 1)];

            // Remove all neighbors with invalid positions
            foreach (var neighbor in neighbors.ToList())
            {
                if (!ValidPosition(neighbor)) neighbors.Remove(neighbor);
            }

            return neighbors;
        }

        /// <summary>
        /// Checks whether a position falls within the game's grid.
        /// </summary>
        /// <param name="pos">Position to check.</param>
        /// <returns>Whether the position is within the game's grid.</returns>
        bool ValidPosition(Int2 pos) => pos.X >= 0 && pos.X < GridDims.X && pos.Y >= 0 && pos.Y < GridDims.Y;

        /// <summary>
        /// Finds all grid cells currently being blocked by the snake.
        /// </summary>
        /// <returns>HashSet of all blocked grid positions.</returns>
        HashSet<Int2> BlockedCells()
        {
            HashSet<Int2> blocked = [];

            // Iterate through all of the snake's nodes
            SnakeNode? node = SnakeHead.Child;
            while (node is not null)
            {
                if (node.Child is not null) blocked.Add(node.Position); // ignore the tail - cannot be collided with
                else break;

                node = node.Child;
            }

            return blocked;
        }

        /// <summary>
        /// Generates a 2D tensor representing the state of every cell within the game's grid.
        /// </summary>
        /// <returns>Tensor encoding each grid position's state.</returns>
        Tensor GetBoardState()
        {
            Tensor state = new([GridDims.Y, GridDims.X]);

            // Encode all positions occupied by the snake
            SnakeNode node = SnakeHead;
            state[[node.Position.Y, node.Position.X]] = BoardEncoding.Head;
            while (node.Child is not null)
            {
                node = node.Child;
                state[[node.Position.Y, node.Position.X]] = (node.Child is not null) ? BoardEncoding.Body : BoardEncoding.Tail;
            }

            // Encode the position occupied by the apple
            state[[ApplePosition.Y, ApplePosition.X]] = BoardEncoding.Apple;

            return state;
        }

        /// <summary>
        /// Has the given agent play a complete episode of the game.
        /// </summary>
        /// <param name="agent">Agent to be used to play the game.</param>
        public void Play(Model agent)
        {
            bool playing = true;
            while (playing)
            {
                Reset();

                // Play until the agent collides or fails to reach an apple in time
                int stepsWithoutApple = 0;
                while (!Collided())
                {
                    int action = PickAgentAction(agent.Predict(Tensor.WrapBatch(GetNormalizedState())));
                    SnakeHead.Move(MapAction(action));

                    if (Collided()) break;

                    if (AteApple()) stepsWithoutApple = 0;
                    else stepsWithoutApple++;

                    Console.Clear();
                    DrawSnake();

                    if (stepsWithoutApple >= MaxStepsWithoutApple) break;

                    Thread.Sleep(FrameTime);
                }

                Console.WriteLine("\nAgent collided or timed out!");

                playing = GetInput("Watch agent play again? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes];
            }
        }

        /// <summary>
        /// Draws the current game board to the console.
        /// </summary>
        void DrawSnake()
        {
            var state = GetBoardState(); // get the current state of the board

            Console.WriteLine("Key: A - Apple, H - Snake Head, B - Snake Body, T - Snake Tail\n");
            Console.WriteLine($"Snake Length: {SnakeLength}");

            for (int row = -1; row <= GridDims.Y; row++)
            {
                for (int col = -1; col <= GridDims.X; col++)
                {
                    bool rowEdge = row == -1 || row == GridDims.Y;
                    bool colEdge = col == -1 || col == GridDims.X;
                    if (rowEdge && colEdge) Console.Write("+"); // draw corner
                    else if (rowEdge) Console.Write("-"); // draw top/bottom edge
                    else if (colEdge) Console.Write("|"); // draw left/right edge
                    else
                    {
                        // Fill the cell based on its encoding
                        string fill = state[[row, col]] switch
                        {
                            BoardEncoding.Head => "H",
                            BoardEncoding.Body => "B",
                            BoardEncoding.Tail => "T",
                            BoardEncoding.Apple => "A",
                            _ => " "
                        };
                        Console.Write(fill);
                    }
                }

                Console.Write("\n");
            }
        }

        /// <summary>
        /// Shows the instructions for the environment's demonstration.
        /// </summary>
        static void ShowDemoInstructions()
        {
            Console.WriteLine("Welcome to the Snake agent demonstration.");
            Console.WriteLine("The agent contains a total of 131 neurons.");
            Console.WriteLine("These are arranged in two layers of 64 neurons each and an output layer of 3 neurons - one each for moving forward, left, and right.");
            Console.WriteLine("The agent receives 7 inputs.");
            Console.WriteLine("These include: the X and Y distances to the apple, the direction the snake's head is currently facing,");
            Console.WriteLine("The distances to the nearest obstacle to the front, left, and right, and the proportion of empty spaces which it can currently reach.");
            Console.WriteLine("This agent was trained over the course of roughly 40,000 games of Snake.");
            Console.WriteLine("It is nowhere near perfect, and is unlikely to reach high scores.");
            Console.WriteLine("But this seems to be approaching the limit of what this specific architecture is able to achieve with its limited view of the game.");
            Console.WriteLine("My next plan is to implement support for convolutional neural networks, which will be far more capable of understanding the full board.");
            Console.WriteLine("But, until that happens, enjoy watching this current limited version.");
            Console.WriteLine("I still find it impressive that it was able to learn how to survive as long as it usually does, given it started out being completely random.");
            Console.WriteLine("Keep in mind that there may be certain initial starting layouts in which it may just simply fail.");
            Console.WriteLine("Feel free to run the demo again if that were to happen.\n");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        /// <summary>
        /// Class representing a single node of the linked list representing the snake.
        /// </summary>
        /// <param name="parent">Node ahead of the new node in the linked list.</param>
        /// <param name="x">X coordinate of the new node.</param>
        /// <param name="y">Y coordinate of the new node.</param>
        class SnakeNode(SnakeNode? parent = null, int x = 0, int y = 0)
        {
            // Linked list properties
            /// <summary>
            /// Node immediately ahead of this snake node.
            /// </summary>
            public SnakeNode? Parent { get; } = parent;
            /// <summary>
            /// Node immediately behind this snake node.
            /// </summary>
            public SnakeNode? Child { get; private set; } = null;

            // Position properties
            /// <summary>
            /// Direction the snake node is currently facing.
            /// </summary>
            public int Direction { get; set; }
            /// <summary>
            /// Current position of the snake node.
            /// </summary>
            public Int2 Position { get; set; } = new(x, y);
            /// <summary>
            /// Previous position of the snake node.
            /// </summary>
            Int2 PrevPosition { get; set; } = new();

            /// <summary>
            /// Moves the snake node in the given direction.
            /// </summary>
            /// <param name="dir">Cardinal direction in which to move.</param>
            /// <exception cref="ArgumentException">Direction index outside valid range.</exception>
            public void Move(int dir)
            {
                PrevPosition = Position;

                Position = dir switch
                {
                    (int)Movements.Left => new(Position.X - 1, Position.Y),
                    (int)Movements.Up => new(Position.X, Position.Y - 1),
                    (int)Movements.Right => new(Position.X + 1, Position.Y),
                    (int)Movements.Down => new(Position.X, Position.Y + 1),
                    _ => throw new ArgumentException("Invalid Movement")
                };

                Child?.Move(Direction); // recursively move all snake nodes in the linked list

                Direction = dir;
            }

            /// <summary>
            /// Extends the snake by one node.
            /// </summary>
            public void Grow()
            {
                if (Child is not null)
                {
                    Child.Grow(); // recurse to the tail node through the linked list
                }
                else
                {
                    Child = new(this, PrevPosition.X, PrevPosition.Y); // add a new node at the current snake node's previous position
                }
            }
        }

        /// <summary>
        /// Struct representing a position within the game's grid.
        /// </summary>
        /// <param name="x">X coordinate of the new position within the grid.</param>
        /// <param name="y">Y coordinate of the new position within the grid.</param>
        struct Int2(int x = 0, int y = 0)
        {
            /// <summary>
            /// X coordinate of the position within the grid.
            /// </summary>
            public int X { get; set; } = x;
            /// <summary>
            /// Y coordinate of the position within the grid.
            /// </summary>
            public int Y { get; set; } = y;

            // Equality overrides

            /// <summary>
            /// Compares the X and Y coordinates of two positions.
            /// </summary>
            /// <param name="a">First position to compare.</param>
            /// <param name="b">Second position to compare.</param>
            /// <returns>Whether the X and Y coordinates of the two positions are equal.</returns>
            public static bool operator ==(Int2 a, Int2 b)
            {
                return a.X == b.X && a.Y == b.Y;
            }

            /// <summary>
            /// Compares the X and Y coordinates of two positions.
            /// </summary>
            /// <param name="a">First position to compare.</param>
            /// <param name="b">Second position to compare.</param>
            /// <returns>Whether the X and Y coordinates of the two positions are not equal.</returns>
            public static bool operator !=(Int2 a, Int2 b)
            {
                return !(a == b);
            }

            /// <summary>
            /// Compares an object to this position.
            /// </summary>
            /// <param name="obj">Object to compare.</param>
            /// <returns>Whether the object is also a position and it and this position have equal X and Y coordinates.</returns>
            public override readonly bool Equals(object? obj) => obj is Int2 other && Equals(other);

            /// <summary>
            /// Compares another position to this position.
            /// </summary>
            /// <param name="other">Position to compare.</param>
            /// <returns>Whether the positions have equal X and Y coordinates.</returns>
            public readonly bool Equals(Int2 other) => this == other;

            /// <summary>
            /// Generates a HashCode for this position based on its X and Y coordinates.
            /// </summary>
            /// <returns>HashCode for this node.</returns>
            public override readonly int GetHashCode() => HashCode.Combine(X, Y);
        }
    }

    /// <summary>
    /// Record containing all of the experiences within a single DQN training episode.
    /// </summary>
    public record Episode
    {
        /// <summary>
        /// List of experiences contained in the episode.
        /// </summary>
        public List<Experience> Experiences { get; init; }

        /// <summary>
        /// Creates a new episode instance.
        /// </summary>
        /// <param name="experiences">List of experiences within the episode.</param>
        public Episode(List<Experience> experiences)
        {
            Experiences = [.. experiences.Select(e => new Experience(e.State, e.Action, e.Reward, e.NextState, e.Done))]; // create a new copy of each experience
        }
    }

    /// <summary>
    /// Record of a single DQN training experience.
    /// </summary>
    public record Experience
    {
        /// <summary>
        /// Initial environment state.
        /// </summary>
        public Tensor State { get; init; }
        /// <summary>
        /// Action selected by the agent.
        /// </summary>
        public int Action { get; init; }
        /// <summary>
        /// Reward of the selected action.
        /// </summary>
        public double Reward { get; init; }
        /// <summary>
        /// Environment state following the selected action.
        /// </summary>
        public Tensor NextState { get; init; }
        /// <summary>
        /// Whether the episode terminated.
        /// </summary>
        public bool Done { get; init; }
        /// <summary>
        /// Replay priority of the experience - temporal difference error.
        /// </summary>
        public double Priority { get; set; }

        /// <summary>
        /// Creates a new experience instance.
        /// </summary>
        /// <param name="state">Initial environment state.</param>
        /// <param name="action">Action selected by the agent.</param>
        /// <param name="reward">Reward of the selected action.</param>
        /// <param name="nextState">Environment state following the selected action.</param>
        /// <param name="done">Whether the episode terminated.</param>
        /// <param name="priority">Replay priority of the experience - temporal difference error.</param>
        public Experience(Tensor state, int action, double reward, Tensor nextState, bool done, double priority = 1.0)
        {
            State = state.Copy();
            Action = action;
            Reward = reward;
            NextState = nextState.Copy();
            Done = done;
            Priority = Math.Max(priority, 1e-8); // ensure non-zero priority
        }
    }

    /// <summary>
    /// Generic First-In First-Out buffer.
    /// </summary>
    /// <typeparam name="T">Type of the elements contained in the buffer.</typeparam>
    /// <param name="maxSize">Maximum size of the buffer.</param>
    public class FIFOBuffer<T>(int maxSize)
    {
        /// <summary>
        /// Maximum size of the buffer.
        /// </summary>
        public int MaxSize { get; init; } = maxSize;
        /// <summary>
        /// List of elements stored in the buffer.
        /// </summary>
        readonly protected List<T> Buffer = [];
        /// <summary>
        /// Index of the oldest added elements.
        /// </summary>
        protected int FirstIndex = 0;
        /// <summary>
        /// Number of elements stored in the buffer.
        /// </summary>
        public int Count => Buffer.Count;

        /// <summary>
        /// Appends an element to the end of the buffer.
        /// </summary>
        /// <param name="item">Element to append.</param>
        public virtual void Add(T item)
        {
            if (Count < MaxSize) Buffer.Add(item);
            else
            {
                Buffer[FirstIndex] = item; // replace oldest element
                FirstIndex = (FirstIndex + 1) % MaxSize; // increment index of oldest element
            }
        }

        /// <summary>
        /// Returns the element at the given index in the buffer.
        /// </summary>
        /// <param name="index">Index of the element being accessed.</param>
        /// <returns>Element at the given index.</returns>
        public T this[int index]
        {
            get => Buffer[(FirstIndex + index) % Count]; // adjust the element index based on current index of the oldest element
        }
    }

    public class SumTree<T>(int capacity)
    {
        readonly int Capacity = capacity;
        readonly double[] Tree = new double[2 * capacity - 1];
        readonly T[] Data = new T[capacity];
        int WritePointer = 0;
        public int Count { get; private set; } = 0;
        public double TotalPriority => Tree[0];

        public void Add(T item, double priority)
        {
            Data[WritePointer] = item;

            int treeIndex = WritePointer + Capacity - 1;
            Update(treeIndex, priority);

            WritePointer = (WritePointer + 1) % Capacity;
            if (Count < Capacity) Count++;
        }

        public void Update(int treeIndex, double priority)
        {
            double change = priority - Tree[treeIndex];
            Tree[treeIndex] = priority;

            while (treeIndex > 0)
            {
                treeIndex = (treeIndex - 1) / 2;
                Tree[treeIndex] += change;
            }
        }

        public (int treeIndex, double priority, T item) Get(double value)
        {
            int parentIndex = 0;

            while (parentIndex < Capacity - 1)
            {
                int leftChild = 2 * parentIndex + 1;
                int rightChild = leftChild + 1;

                if (value <= Tree[leftChild])
                {
                    parentIndex = leftChild;
                }
                else
                {
                    value -= Tree[leftChild];
                    parentIndex = rightChild;
                }
            }

            int dataIndex = parentIndex - (Capacity - 1);
            return (parentIndex, Tree[parentIndex], Data[dataIndex]);
        }
    }

    public class ReplayBuffer(int capacity, double alpha = 0.6)
    {
        readonly SumTree<Experience> SumTree = new(capacity);
        readonly Random Random = new();
        double MaxPriority = 1.0;
        readonly double Alpha = alpha;
        double Beta = 0.4;
        const double BetaIncrement = 0.001;
        public int Count => SumTree.Count;

        public void Add(Experience experience)
        {
            double priority = Math.Pow(MaxPriority, Alpha);
            SumTree.Add(experience, priority);
        }

        public (List<Experience> batch, int[] indices, double[] weights) GetBatch(int batchSize)
        {
            List<Experience> batch = new(batchSize);
            var indices = new int[batchSize];
            var weights = new double[batchSize];

            double totalPriority = SumTree.TotalPriority;
            double segment = totalPriority / batchSize;
            double maxWeight = 0;

            for (int i = 0; i < batchSize; i++)
            {
                double low = segment * i;
                double high = segment * (i + 1);
                double sampleVal = low + (Random.NextDouble() * (high - low));

                var (treeIndex, priority, item) = SumTree.Get(sampleVal);

                batch.Add(item);
                indices[i] = treeIndex;
                weights[i] = priority;
            }

            for (int i = 0; i < batchSize; i++)
            {
                double prob = weights[i] / totalPriority;
                if (prob == 0) prob = 1e-8;

                double weight = Math.Pow(1.0 / (SumTree.Count * prob), Beta);
                weights[i] = weight;
                if (weight > maxWeight) maxWeight = weight;
            }

            if (maxWeight > 0)
            {
                for (int i = 0; i < batchSize; i++)
                {
                    weights[i] /= maxWeight;
                }
            }

            Beta = Math.Min(1.0, Beta + BetaIncrement);

            return (batch, indices, weights);
        }

        public void UpdatePriorities(int[] indices, double[] priorities)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                if (priorities[i] > MaxPriority)
                {
                    MaxPriority = priorities[i];
                }

                double powerPriority = Math.Pow(priorities[i], Alpha);
                SumTree.Update(indices[i], powerPriority);
            }
        }
    }

    /// <summary>
    /// Deep Q-Network (DQN) trainer class.
    /// </summary>
    /// <param name="agent">DQN agent to be trained.</param>
    /// <param name="environment">Environment to train in.</param>
    /// <param name="optimizer">Optimizer to use for parameter updates.</param>
    /// <param name="cost">Cost function to use for loss calculation.</param>
    /// <param name="discount">Discount factor for future rewards.</param>
    /// <param name="exploration">Initial exploration rate of the agent.</param>
    /// <param name="explorationDecay">Per-episode exponential decay factor of exploration rate.</param>
    /// <param name="minExploration">Minimum exploration rate of the agent.</param>
    /// <param name="replayBufferSize">Size of the experience replay buffer.</param>
    /// <param name="batchSize">Number of experiences in each training batch.</param>
    /// <param name="agentBufferSize">Size of the opponent agent buffer for self-play environments.</param>
    /// <param name="opponentCopyRate">Number of episodes between opponent agents being frozen for self-play environments.</param>
    /// <param name="minRandomOpponentEpisodes">Minimum number of episodes with a randomly acting opponent for self-play environments.</param>
    /// <param name="tau">Target model parameter update factor.</param>
    /// <param name="maxGradNorm">Maximum total magnitude of gradients without normalization.</param>
    /// <param name="minExperiences">Minimum number of experiences before training can begin.</param>
    public class DQNTrainer(Model agent, Environment environment, Optimizer optimizer, Cost cost, int trainEvery = 4, double discount = 0.995,
        double exploration = 1.0, double explorationDecay = 0.99, double minExploration = 0.01, int replayBufferSize = 10000, int batchSize = 64,
        int agentBufferSize = 5, int opponentCopyRate = 100, int minRandomOpponentEpisodes = 200, double tau = 0.005, double maxGradNorm = 1.0,
        int minExperiences = 1000)
    {
        // Agent and environment parameters
        /// <summary>
        /// Agent being trained.
        /// </summary>
        readonly Model Agent = agent;
        /// <summary>
        /// Target prediction model.
        /// </summary>
        readonly Model TargetModel = agent.Copy();
        /// <summary>
        /// Environment agent is being trained in.
        /// </summary>
        readonly Environment Environment = environment;
        /// <summary>
        /// Optimizer used to update parameters.
        /// </summary>
        readonly Optimizer Optimizer = optimizer;
        /// <summary>
        /// Cost function used to calculate loss.
        /// </summary>
        readonly Cost Cost = cost;

        // Experience buffer parameters
        /// <summary>
        /// Buffer containing past experiences.
        /// </summary>
        readonly ReplayBuffer ReplayBuffer = new(replayBufferSize);
        /// <summary>
        /// Minimum number of stored experiences to begin training.
        /// </summary>
        readonly int MinExperiences = minExperiences;
        /// <summary>
        /// Number of experiences in each training batch.
        /// </summary>
        readonly int BatchSize = batchSize;

        // Training parameters
        /// <summary>
        /// Number of environment steps between DQN training passes.
        /// </summary>
        readonly int TrainEvery = trainEvery;
        /// <summary>
        /// Discount factor of future rewards.
        /// </summary>
        readonly double Discount = discount;
        /// <summary>
        /// Current exploration rate of the agent.
        /// </summary>
        double Exploration = exploration;
        /// <summary>
        /// Per-episode exponential decay factor of the exploration rate.
        /// </summary>
        readonly double ExplorationDecay = explorationDecay;
        /// <summary>
        /// Minimum exploration rate of the agent.
        /// </summary>
        readonly double MinExploration = minExploration;
        /// <summary>
        /// Maximum magnitude of gradients without normalization.
        /// </summary>
        readonly double MaxNorm = maxGradNorm;
        /// <summary>
        /// Total number of times the agent's parameters have been optimized.
        /// </summary>
        int optimizerSteps = 0;
        /// <summary>
        /// Target model parameter update factor.
        /// </summary>
        readonly double Tau = tau;
        readonly double OneMinusTau = 1.0 - tau;
        readonly Vector<double> TauVec = new(tau);
        readonly Vector<double> OneMinusTauVec = new(1.0 - tau);

        // Self-play parameters
        /// <summary>
        /// Whether the training environment requires self-play.
        /// </summary>
        readonly bool SelfPlay = environment is ISelfPlay;
        /// <summary>
        /// Buffer storing frozen opponents for self-play.
        /// </summary>
        readonly FIFOBuffer<Model> AgentBuffer = new(agentBufferSize);
        /// <summary>
        /// Number of episodes between opponent agents being frozen and stored for self-play.
        /// </summary>
        readonly int OpponentCopyRate = opponentCopyRate;
        /// <summary>
        /// Minimum number of episodes with a randomly acting opponent for self-play.
        /// </summary>
        readonly int MinRandomOppEpisodes = minRandomOpponentEpisodes;

        // Utilities
        /// <summary>
        /// Trainer's Random instance.
        /// </summary>
        readonly Random Random = new();
        /// <summary>
        /// Total loss accumulated during the episode.
        /// </summary>
        double totalLoss = 0.0;
        static readonly int VectorSize = Vector<double>.Count;

        // Persistent training buffers
        /// <summary>
        /// Persistent buffer for each training batch of current states.
        /// </summary>
        Tensor? _currentBatch;
        /// <summary>
        /// Persistent buffer for each training batch of next states.
        /// </summary>
        Tensor? _nextBatch;
        /// <summary>
        /// Persistent buffer for next states during future value prediction.
        /// </summary>
        Tensor? _nextState;
        /// <summary>
        /// Persistent buffer for target Q-Values during training.
        /// </summary>
        Tensor? _targetQs;

        /// <summary>
        /// Trains the agent for a given number of episodes.
        /// </summary>
        /// <param name="episodeBuffer">Buffer in which to store episodes for reviewing.</param>
        /// <param name="episodes">Number of episodes to train for.</param>
        public void Train(ref FIFOBuffer<Episode>? episodeBuffer, int episodes = 1000)
        {
            List<Experience> episodeExperiences = [];
            Tensor state; // normalized state
            Tensor trueState; // unnormalized state
            bool done;
            bool learnerTurn;
            int action;
            double reward;
            double totalReward;
            int step;
            int trainSteps;
            Tensor nextState;
            TimeSpan avgElapsed = new(0);
            Stopwatch stopwatch = new();
            stopwatch.Start();
            for (int e = 0; e < episodes; e++)
            {
                // Freeze new opponent agent for self-play every OpponentCopyRate episodes
                if (SelfPlay && ((e + 1) >= MinRandomOppEpisodes) && ((e + 1) % OpponentCopyRate == 0)) AgentBuffer.Add(Agent.Copy());

                totalLoss = 0.0;
                episodeExperiences.Clear();
                Environment.Reset();
                state = Environment.GetNormalizedState();

                // Run full episode until it has finished
                done = false;
                step = 0;
                trainSteps = 0;
                totalReward = 0;
                while (!done)
                {
                    step++;
                    trueState = Environment.GetState();
                    learnerTurn = Environment is not ISelfPlay sp || sp.AgentTurn; // agent acts on every step unless in self-play
                    action = PickNextAction(state);

                    (reward, nextState, done) = Environment.Step(action, step);
                    totalReward += reward;

                    // Store experience for training and episode review
                    if (learnerTurn) ReplayBuffer.Add(new(state, action, reward, nextState, done));
                    episodeExperiences.Add(new(trueState, action, reward, done ? trueState : Environment.GetState(), done));

                    if ((step - 1) % TrainEvery == 0)
                    {
                        TrainNetwork();
                        trainSteps++;
                    }

                    state = nextState;
                }

                episodeBuffer?.Add(new(episodeExperiences));

                Exploration = Math.Max(Exploration * ExplorationDecay, MinExploration); // exponentially decay exploration rate

                // Calculate diagnostic data
                var elapsed = stopwatch.Elapsed;
                avgElapsed += (elapsed - avgElapsed) / (e + 1);
                var eta = avgElapsed * (episodes - e - 1);

                // Log episode diagnostics in the console
                Console.WriteLine($"\nEpisode {e + 1}/{episodes} finished...");
                Console.WriteLine($"Total Reward: {totalReward:F2},");
                Console.WriteLine($"Average Loss: {(totalLoss / trainSteps):F3}");
                Console.WriteLine($"Exploration Rate: {Exploration:F2}, Experience Count: {ReplayBuffer.Count}");
                if (Environment is ISelfPlay selfPlayEnv)
                {
                    Console.WriteLine($"Opponent agent: {(selfPlayEnv.OpponentIndex < selfPlayEnv.OpponentCount ?
                        $"{selfPlayEnv.OpponentIndex + 1}/{selfPlayEnv.OpponentCount}" : "Random")}");
                }
                Console.WriteLine($"Final State:");
                if (episodeBuffer is not null) Environment.Render(episodeBuffer[^1], step);
                Console.WriteLine($"Ended on step: {step}");
                Console.WriteLine($"Episode Duration: {MathUtils.RoundToMS(elapsed):g}, Estimated Time Remaining: {MathUtils.RoundToMS(eta):g}");
                stopwatch.Restart();
            }
        }

        /// <summary>
        /// Picks the next action to be taken.
        /// </summary>
        /// <param name="state">Tensor representing the environment's current state.</param>
        /// <returns>Index of the action to be taken.</returns>
        int PickNextAction(Tensor state)
        {
            if (Environment is ISelfPlay selfPlayEnv && !selfPlayEnv.AgentTurn)
            {
                return PickOpponentAction();
            }
            else
            {
                return PickAgentAction(state);
            }
        }

        /// <summary>
        /// Picks the next action to be taken by the agent.
        /// </summary>
        /// <param name="state">Tensor representing the environment's current state.</param>
        /// <returns>Index of the action to be taken.</returns>
        int PickAgentAction(Tensor state)
        {
            if (Random.NextDouble() < Exploration) // pick random action for exploration
            {
                return Environment.PickRandomAction();
            }
            else // pick action based on predicted Q-Values
            {
                return Environment.PickAgentAction(Agent.Predict(Tensor.WrapBatch(state)));
            }
        }

        /// <summary>
        /// Picks the next action to be taken by the opponent in self-play.
        /// </summary>
        /// <returns>Index of the action to be taken.</returns>
        /// <exception cref="Exception">Training environment is not self-play.</exception>
        int PickOpponentAction()
        {
            if (Environment is ISelfPlay selfPlayEnv)
            {
                return selfPlayEnv.PickOpponentAction(AgentBuffer);
            }
            else throw new Exception("Environment not self-play");
        }

        /// <summary>
        /// Trains the agent using a batch of stored experiences.
        /// </summary>
        void TrainNetwork()
        {
            if (ReplayBuffer.Count < MinExperiences) return; // skip training until minimum number of experiences are stored

            var (batch, indices, weights) = ReplayBuffer.GetBatch(BatchSize);

            // Initialize persistent buffers if not yet initialized
            if (_currentBatch is null || _nextBatch is null)
            {
                var stateDims = batch[0].State.Dimensions;
                var batchDims = new int[stateDims.Length + 1];
                batchDims[0] = BatchSize;
                stateDims.CopyTo(batchDims, 1);
                _currentBatch = new(batchDims);
                _nextBatch = new(batchDims);
            }

            // Copy current training batch into batch buffers
            int batchOffset;
            for (int b = 0; b < BatchSize; b++)
            {
                batchOffset = b * Environment.StateSize;
                Array.Copy(batch[b].State.Data, 0, _currentBatch.Data, batchOffset, Environment.StateSize);
                Array.Copy(batch[b].NextState.Data, 0, _nextBatch.Data, batchOffset, Environment.StateSize);
            }

            // Predict future values of actions
            var nextAgentQs = Agent.Predict(_nextBatch).Copy(); // select actions for experience's next states
            var nextTargetQs = TargetModel.Predict(_nextBatch).Copy(); // predict Q-Values of actions
            var targetQs = MaskQValuesDouble(nextAgentQs, nextTargetQs, batch);

            // Predict Q-Values of actions in the batch
            var predictions = Agent.Forward(_currentBatch); // predicted after next states to avoid overwriting autograd graph
            var predictedQs = Tensor.MaskActions(predictions, batch);

            // Calculate the agent's loss and new priorities of each experience
            var lossResult = Cost.CalculateCostWithPriority(predictedQs, targetQs, weights);
            ReplayBuffer.UpdatePriorities(indices, lossResult.Priorities);
            var loss = Tensor.Mean(lossResult.Losses);
            totalLoss += loss[0];

            // Calculate parameter gradients
            loss.Backward();
            Agent.ClipGradients(MaxNorm);

            // Update parameters based on gradients
            for (int i = 0; i < Agent.ParameterCount; i++)
            {
                Optimizer.Step(Agent.Parameters[i], optimizerSteps);
            }

            // Gradually update target model parameters
            for (int i = 0; i < TargetModel.ParameterCount; i++)
            {
                var agentParamVecs = MemoryMarshal.Cast<double, Vector<double>>(Agent.Parameters[i].Data.AsSpan());
                var targetParamVecs = MemoryMarshal.Cast<double, Vector<double>>(TargetModel.Parameters[i].Data.AsSpan());
                for (int j = 0; j < agentParamVecs.Length; j++)
                {
                    targetParamVecs[j] = (TauVec * agentParamVecs[j]) + (OneMinusTauVec * targetParamVecs[j]);
                }

                for (int j = agentParamVecs.Length * VectorSize; j < TargetModel.Parameters[i].ElementCount; j++)
                {
                    TargetModel.Parameters[i][j] = (Tau * Agent.Parameters[i][j]) + (OneMinusTau * TargetModel.Parameters[i][j]);
                }
            }

            optimizerSteps++;
        }

        /// <summary>
        /// Masks Q-Values based on agent's selected action and calculates target Q-Values using the Bellman equation.
        /// </summary>
        /// <param name="agentQValues">Q-Values predicted by the agent used for action select.</param>
        /// <param name="targetQValues">Q-Values predicted by the target model for target Q-Values calculation.</param>
        /// <param name="batch">Experience batch corresponding to the given Q-Values.</param>
        /// <returns>Target Q-Values for each experience in the batch.</returns>
        Tensor MaskQValuesDouble(Tensor agentQValues, Tensor targetQValues, List<Experience> batch)
        {
            // Initialize persistent buffers if not yet initialized
            _targetQs ??= new([BatchSize, 1]);
            _nextState ??= new(batch[0].NextState.Dimensions);

            // Ensure persistent buffers do not store unnecessary graphs
            _targetQs.ClearGraph();
            _nextState.ClearGraph();

            int actionCount = agentQValues.Dimensions[^1];
            int stateSize = Environment.StateSize;

            // Calculate target Q-Value for each experience in the batch using the Bellman equation -> Q(s, a) = R(s) + γ * maxQ(s', a')
            for (int i = 0; i < BatchSize; i++)
            {
                double qTarget = batch[i].Reward; // add immediate reward of the action

                // Calculate predicted future value of the action
                if (!batch[i].Done)
                {
                    Array.Copy(_nextBatch!.Data, i * stateSize, _nextState.Data, 0, stateSize); // update the next state buffer with relevant data from the next batch buffer

                    // Find the best valid action predicted by the agent for the next state
                    int bestAction = -1;
                    double bestQ = double.MinValue;
                    for (int a = 0; a < actionCount; a++)
                    {
                        if (!Environment.ValidAction(a, _nextState)) continue;
                        double q = agentQValues[i * actionCount + a];
                        if (q > bestQ)
                        {
                            bestQ = q;
                            bestAction = a;
                        }
                    }

                    if (bestAction != -1)
                    {
                        double evalQ = targetQValues[i * actionCount + bestAction]; // get future value predicted by target model
                        qTarget += Discount * evalQ * (SelfPlay ? -1.0 : 1.0); // add future value to the target Q-Value using the Bellman equation
                    }
                }

                _targetQs[i] = qTarget;
            }

            return _targetQs;
        }
    }

    /// <summary>
    /// Basic non-experiencial neural network model trainer.
    /// </summary>
    /// <param name="model">Model to be trained.</param>
    /// <param name="optimizer">Optimizer to use for parameter updates.</param>
    /// <param name="cost">Cost function to use for loss calculation.</param>
    public class Trainer(Model model, Optimizer optimizer, Cost cost)
    {
        /// <summary>
        /// Model being trained.
        /// </summary>
        readonly Model Model = model;
        /// <summary>
        /// Optimizer being used for parameter updates.
        /// </summary>
        readonly Optimizer Optimizer = optimizer;
        /// <summary>
        /// Cost function being used for loss calculation.
        /// </summary>
        readonly Cost Cost = cost;

        /// <summary>
        /// Trains the model for the given number of epochs using the given dataset.
        /// </summary>
        /// <param name="inputs">Tensor representing the input values of the dataset.</param>
        /// <param name="targets">Tensor representing the target outputs of the dataset.</param>
        /// <param name="epochs">Number of epochs to train for.</param>
        public void Train(Tensor inputs, Tensor targets, int epochs)
        {
            Stopwatch timer = new();

            int logEvery = Math.Max(100, MathUtils.RoundToInterval(epochs / 500f, 100));
            Tensor predictions;
            Tensor loss;

            // Train for the given number of epochs
            timer.Start();
            for (int e = 0; e < epochs; e++)
            {
                predictions = Model.Forward(inputs);
                loss = Cost.CalculateCost(predictions, targets);
                loss.Backward();

                foreach (var param in Model.Parameters)
                {
                    Optimizer.Step(param, epochs);
                }

                // Log diagnostic data to the console
                if (e % logEvery == 0 || e == epochs - 1)
                {
                    Console.WriteLine($"Epoch {e} : Loss = {loss[0]} : Time elapsed = {timer.ElapsedMilliseconds}ms : Time per epoch = {((float)timer.ElapsedMilliseconds / logEvery):F2}ms");
                    timer.Restart();
                }
            }
        }
    }

    /// <summary>
    /// Base class for optimizers used for updating model parameters.
    /// </summary>
    /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
    public abstract class Optimizer(double learningRate)
    {
        /// <summary>
        /// Gradient scaling factor for parameter updates.
        /// </summary>
        protected readonly double LR = learningRate;
        protected readonly Vector<double> LRVec = new(learningRate);
        protected static readonly int VectorSize = Vector<double>.Count;

        /// <summary>
        /// Updates the parameter based on its gradient.
        /// </summary>
        /// <param name="parameter">Parameter tensor to be updated.</param>
        /// <param name="iterations">Number of training iterations which have been run.</param>
        public abstract void Step(Tensor parameter, int iterations);
    }

    /// <summary>
    /// Stochastic Gradient Descent optimizer.
    /// </summary>
    /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
    public class SGD(double learningRate) : Optimizer(learningRate)
    {
        public override void Step(Tensor parameter, int iterations)
        {
            var paramVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Data.AsSpan());
            var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Grad.AsSpan());
            for (int i = 0; i < paramVecs.Length; i++)
            {
                paramVecs[i] -= gradVecs[i] * LRVec;
            }
            
            for (int i = paramVecs.Length * VectorSize; i < parameter.ElementCount; i++)
            {
                parameter[i] -= parameter.Grad[i] * LR;
            }
        }
    }

    /// <summary>
    /// Adaptive Moment Estimation optimizer.
    /// </summary>
    /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
    /// <param name="beta1">Exponential decay rate of first moment estimates.</param>
    /// <param name="beta2">Exponential decay rate of second moment estimates.</param>
    /// <param name="epsilon">Epsilon value to use.</param>
    public class Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : Optimizer(learningRate)
    {
        /// <summary>
        /// Exponential decay rate of first moment estimates.
        /// </summary>
        readonly double Beta1 = beta1;
        readonly double OneMinusBeta1 = 1.0 - beta1;
        readonly Vector<double> Beta1Vec = new(beta1);
        readonly Vector<double> OneMinusBeta1Vec = new(1.0 - beta1);
        /// <summary>
        /// Exponential decay rate of second moment estimates.
        /// </summary>
        readonly double Beta2 = beta2;
        readonly double OneMinusBeta2 = 1.0 - beta2;
        readonly Vector<double> Beta2Vec = new(beta2);
        readonly Vector<double> OneMinusBeta2Vec = new(1.0 - beta2);
        /// <summary>
        /// Epsilon value to use.
        /// </summary>
        readonly double Epsilon = epsilon;
        readonly Vector<double> EpsilonVec = new(epsilon);

        /// <summary>
        /// Dictionary of per-parameter persistent buffers for first and second moments.
        /// </summary>
        readonly Dictionary<Tensor, (double[] m, double[] v)> _state = [];

        public override void Step(Tensor parameter, int iteration)
        {
            // Create a new persistent moment buffer if necessary
            if (!_state.TryGetValue(parameter, out var moments))
            {
                moments = (new double[parameter.ElementCount], new double[parameter.ElementCount]);
                _state[parameter] = moments;
            }

            // Update moments and parameter values
            double biasCorrection1 = 1.0 - Math.Pow(Beta1, iteration + 1);
            var biasCorr1Vec = new Vector<double>(biasCorrection1);

            double biasCorrection2 = 1.0 - Math.Pow(Beta2, iteration + 1);
            var biasCorr2Vec = new Vector<double>(biasCorrection2);
            var (m, v) = moments;

            var mVecs = MemoryMarshal.Cast<double, Vector<double>>(m.AsSpan());
            var vVecs = MemoryMarshal.Cast<double, Vector<double>>(v.AsSpan());
            var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Grad.AsSpan());
            var paramVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Data.AsSpan());

            for (int i = 0; i < paramVecs.Length; i++)
            {
                mVecs[i] = (Beta1Vec * mVecs[i]) + (OneMinusBeta1Vec * gradVecs[i]);
                vVecs[i] = (Beta2Vec * vVecs[i]) + (OneMinusBeta2Vec * (gradVecs[i] * gradVecs[i]));

                var mHatVec = mVecs[i] / biasCorr1Vec;
                var vHatVec = vVecs[i] / biasCorr2Vec;

                paramVecs[i] -= (LRVec * mHatVec) / (Vector.SquareRoot(vHatVec) + EpsilonVec);
            }

            for (int i = paramVecs.Length * VectorSize; i < parameter.ElementCount; i++)
            {
                double grad = parameter.Grad[i];

                // Update parameter moments
                m[i] = Beta1 * m[i] + (1.0 - Beta1) * grad;
                v[i] = Beta2 * v[i] + (1.0 - Beta2) * (grad * grad);

                // Correct moment estimate bias due to 0 initialization
                double mHat = m[i] / biasCorrection1;
                double vHat = v[i] / biasCorrection2;

                parameter.Data[i] -= (LR * mHat) / (Math.Sqrt(vHat) + Epsilon); // update parameter based on moments
            }
        }
    }

    /// <summary>
    /// Base class for cost functions used for calculating model losses.
    /// </summary>
    public abstract class Cost
    {
        /// <summary>
        /// Calculates the mean loss of model predictions.
        /// </summary>
        /// <param name="predictions">Predicted values from the model.</param>
        /// <param name="target">Target outputs.</param>
        /// <returns>Tensor storing the mean loss of the predictions.</returns>
        public abstract Tensor CalculateCost(Tensor predictions, Tensor target);

        /// <summary>
        /// Calculates the loss of each model prediction.
        /// </summary>
        /// <param name="predictions">Predicted values from the model.</param>
        /// <param name="target">Target outputs.</param>
        /// <returns>Tensor storing the loss of each prediction.</returns>
        public abstract Tensor CalculatePerSampleCost(Tensor predictions, Tensor target);

        /// <summary>
        /// Calculates the loss of each model prediction from a PER sampled batch.
        /// </summary>
        /// <param name="predictions">Predicted values from the model.</param>
        /// <param name="target">Target values.</param>
        /// <param name="weights">Sampling weights of the predictions' associated experiences.</param>
        /// <returns>CostResult storing Tensor of per-prediction losses and corresponding PER sampling priorites.</returns>
        public virtual CostResult CalculateCostWithPriority(Tensor predictions, Tensor target, double[]? weights = null)
        {
            var losses = CalculatePerSampleCost(predictions, target); // calculate loss of each prediction

            // Update PER sampling priorities and scale losses based on sampling bias
            var priorities = new double[losses.ElementCount];
            for (int i = 0; i < losses.ElementCount; i++)
            {
                priorities[i] = Math.Abs(losses[i]) + 1e-8;

                if (weights is not null) losses[i] *= weights[i];
            }

            return new(losses, priorities);
        }
    }

    /// <summary>
    /// Mean Squared Error cost function.
    /// </summary>
    public class MSE : Cost
    {
        public override Tensor CalculateCost(Tensor predictions, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(predictions, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
        {
            return Tensor.Pow(target - predictions, 2.0); // MSE function -> (y - y_hat)^2
        }
    }

    /// <summary>
    /// Pseudo-Huber (smoothed) cost function.
    /// </summary>
    /// <param name="delta">Linear transition threshold.</param>
    public class Huber(double delta = 1.0) : Cost
    {
        /// <summary>
        /// Linear transition threshold.
        /// </summary>
        readonly double Delta = delta;

        public override Tensor CalculateCost(Tensor predictions, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(predictions, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
        {
            var diff = predictions - target; // calculate per-prediction error

            // Apply pseudo-Huber function -> delta^2 * (sqrt(1 + (diff/delta)^2) - 1)
            var scaled = diff / Delta;
            var inner = Tensor.Pow(scaled, 2.0) + 1.0;
            return (Tensor.Pow(inner, 0.5) - 1.0) * (Delta * Delta);
        }
    }

    /// <summary>
    /// Record storing PER loss and priority pairs.
    /// </summary>
    /// <param name="Losses">Tensor storing per-prediction losses.</param>
    /// <param name="Priorities">Array storing per-prediction PER sampling priorities.</param>
    public record CostResult(Tensor Losses, double[] Priorities);

    /// <summary>
    /// Neural network model class.
    /// </summary>
    public class Model
    {
        // Externally accessible properties
        /// <summary>
        /// Hidden layers and output layer of the model.
        /// </summary>
        public Layer[] Layers { get; private set; }
        /// <summary>
        /// List of all layer parameters.
        /// </summary>
        public List<Tensor> Parameters
        {
            get
            {
                _parameters ??= [.. GetParameters()]; // initialize internal property if not yet initialized
                return _parameters;
            }
        }
        /// <summary>
        /// Number of parameters across all layers.
        /// </summary>
        public int ParameterCount => Parameters.Count;

        // Internal properties
        /// <summary>
        /// Internal list of all player parameters.
        /// </summary>
        List<Tensor>? _parameters;

        // Utilities
        static readonly int VectorSize = Vector<double>.Count;

        /// <summary>
        /// Initializes a new model with the given layers.
        /// </summary>
        /// <param name="layers">List of hidden and output layers.</param>
        public Model(Layer[] layers)
        {
            Layers = layers;
        }

        /// <summary>
        /// Initializes a new model with the given layers and initializes parameters for the given input format.
        /// </summary>
        /// <param name="layers">List of hidden and output layers.</param>
        /// <param name="inputFormat">Tensor representing the expected input format.</param>
        public Model(Layer[] layers, Tensor inputFormat)
        {
            Layers = layers;
            SetUpLayers(inputFormat);
        }

        /// <summary>
        /// Constructs a new model instance from the given save data.
        /// </summary>
        /// <param name="data">Save data representing the model architecture and training.</param>
        public Model(Saver.ModelData data)
        {
            Layers = new Layer[data.Layers.Length];

            BuildFromData(data);
        }

        /// <summary>
        /// Reconstructs the model to match the architecture and training of the model stored in the save data.
        /// </summary>
        /// <param name="data"></param>
        void BuildFromData(Saver.ModelData data)
        {
            InvalidateParameters(); // ensure any previous parameters are cleared

            // Reconstruct each layer from the save data
            Saver.LayerData layerData;
            Type? layerType;
            for (int i = 0; i < data.Layers.Length; i++)
            {
                layerData = data.Layers[i];

                // Create new instance of saved layer type
                layerType = Type.GetType(layerData.LayerName);
                if (layerType is not null)
                {
                    var layer = Activator.CreateInstance(layerType) as Layer;
                    layer?.BuildFromData(layerData);
                    if (layer is not null) Layers[i] = layer;
                }
            }
        }

        /// <summary>
        /// Initializes layer parameters to support the given input format.
        /// </summary>
        /// <param name="inputFormat">Tensor representing the expected input format.</param>
        public void SetUpLayers(Tensor inputFormat)
        {
            var format = inputFormat.Copy();

            // Initialize parameters of each layer
            foreach (var layer in Layers)
            {
                layer.SetUpLayer(format);
                format = layer.OutputFormat; // every subsequent layer receives output dimensions of the previous layer
            }
        }

        /// <summary>
        /// Sends the input forward through the neural network while building autograd graph.
        /// </summary>
        /// <param name="input">Input tensor to be processed.</param>
        /// <returns>Tensor containing the model's predicted outputs.</returns>
        public Tensor Forward(Tensor input)
        {
            Tensor.BeginForward();

            var output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        /// <summary>
        /// Sends the input forward through the neural network without building an autograd graph.
        /// </summary>
        /// <param name="input">Input tensor to be processed.</param>
        /// <returns>Tensor containing the model's predicted outputs.</returns>
        public Tensor Predict(Tensor input)
        {
            Tensor.Inference = true;
            var output = Forward(input);
            Tensor.Inference = false;
            return output;
        }

        /// <summary>
        /// Gets all of the parameters in the model's hidden layers and output layer.
        /// </summary>
        /// <returns>IEnumerable containing all parameter tensors.</returns>
        public IEnumerable<Tensor> GetParameters()
        {
            foreach (var layer in Layers)
            {
                foreach (var param in layer.GetParameters())
                {
                    yield return param;
                }
            }
        }

        /// <summary>
        /// Clips all of the model's parameters if necessary.
        /// </summary>
        /// <param name="maxNorm">Maximum total magnitude of gradients without clipping.</param>
        public void ClipGradients(double maxNorm)
        {
            // Calculate total magnitude of gradients
            double totalNorm = 0.0;
            foreach (var param in Parameters)
            {
                var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(param.Grad.AsSpan());
                var acc = Vector<double>.Zero;
                for (int i = 0; i < gradVecs.Length; i++)
                {
                    acc += gradVecs[i] * gradVecs[i];
                }
                totalNorm += Vector.Sum(acc);

                for (int i = gradVecs.Length * VectorSize; i < param.GradCount; i++)
                {
                    double grad = param.Grad[i];

                    totalNorm += grad * grad;
                }
            }
            totalNorm = Math.Sqrt(totalNorm);

            // Scale (clip) gradients if necessary
            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / (totalNorm + 1e-8);
                var scaleVec = new Vector<double>(scale);

                foreach (var param in Parameters)
                {
                    var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(param.Grad.AsSpan());
                    for (int i = 0; i < gradVecs.Length; i++)
                    {
                        gradVecs[i] *= scaleVec;
                    }

                    for (int i = gradVecs.Length * VectorSize; i < param.GradCount; i++)
                    {
                        param.Grad[i] *= scale;
                    }
                }
            }
        }

        /// <summary>
        /// Creates an identical deep copy of the model.
        /// </summary>
        /// <returns>New model instance with identical layers and parameters.</returns>
        public Model Copy()
        {
            var layers = new Layer[Layers.Length];

            for (int i = 0; i < Layers.Length; i++)
            {
                layers[i] = Layers[i].Copy();
            }

            return new(layers);
        }

        /// <summary>
        /// Clears the model's current stored parameter references.
        /// </summary>
        void InvalidateParameters() => _parameters = null;
    }

    /// <summary>
    /// Base class for neural network layers.
    /// </summary>
    public abstract class Layer
    {
        /// <summary>
        /// Tensor containing the bias parameters of the layer.
        /// </summary>
        public Tensor Biases { get; protected set; } = new();
        /// <summary>
        /// Activation function used by the layer.
        /// </summary>
        public Activation Activation { get; protected set; } = new Linear();
        public Tensor OutputFormat { get; protected set; } = new();

        /// <summary>
        /// Parameterless constructor for model reconstruction from save data.
        /// </summary>
        public Layer() { }

        /// <summary>
        /// Initializes the layer's parameters for the given number of inputs.
        /// </summary>
        /// <param name="inputFormat">Expected format of input tensors.</param>
        public abstract void SetUpLayer(Tensor inputFormat);

        /// <summary>
        /// Processes the input using the layer's parameters and activation function.
        /// </summary>
        /// <param name="input">Input tensor to be processed.</param>
        /// <returns>Tensor containing the outputs of the layer's operations.</returns>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Gets all of the layer's parameters.
        /// </summary>
        /// <returns>IEnumerable containing all of the layer's parameter tensors.</returns>
        public abstract IEnumerable<Tensor> GetParameters();

        /// <summary>
        /// Creates a new deep copy of the layer with the same parameters and activation function.
        /// </summary>
        /// <returns>New layer instance with identical parameters and activation function.</returns>
        public abstract Layer Copy();

        /// <summary>
        /// Reconstructs the layer from the save data.
        /// </summary>
        /// <param name="data">Save data containing the layer's saved parameters.</param>
        public abstract void BuildFromData(Saver.LayerData data);
    }

    /// <summary>
    /// Fully connected neural network layer.
    /// </summary>
    public class Dense : Layer
    {
        public int NeuronCount { get; private set; }
        /// <summary>
        /// Tensor containing the weights parameters of the layer.
        /// </summary>
        public Tensor Weights { get; private set; } = new();

        /// <summary>
        /// Creates a new dense neural network layer instance.
        /// </summary>
        /// <param name="neuronCount">Number of neurons in the new layer.</param>
        /// <param name="activation">Activation function of the new layer.</param>
        public Dense(int neuronCount, Activation activation)
        {
            NeuronCount = neuronCount;
            Activation = activation;
        }

        /// <summary>
        /// Creates a new dense neural network layer instance.
        /// </summary>
        /// <param name="neuronCount">Number of neurons in the new layer.</param>
        /// <param name="weights">Weights tensor of the new layer.</param>
        /// <param name="biases">Bias tensor of the new layer.</param>
        /// <param name="activation">Activation function of the new layer.</param>
        public Dense(int neuronCount, Tensor weights, Tensor biases, Activation activation)
        {
            NeuronCount = neuronCount;
            Weights = weights;
            Biases = biases;
            Activation = activation;
        }

        /// <summary>
        /// Parameterless constructor for model reconstruction from save data.
        /// </summary>
        public Dense() { }

        // Base Layer API overrides

        public override void SetUpLayer(Tensor inputFormat)
        {
            Weights = Tensor.InitWeights(inputFormat.ElementCount, NeuronCount);
            Biases = Tensor.InitBiases(NeuronCount);
            OutputFormat = new([1, NeuronCount]);
        }

        public override Tensor Forward(Tensor input)
        {
            var flatInput = input.Rank > 2 ? Tensor.Flatten(input, 1) : input; // flatten input to a 2D (batch * state elements) tensor
            var output = flatInput ^ Weights;
            output += Tensor.Broadcast(Biases, output.Dimensions);
            output = Activation.Forward(output);
            return output;
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            yield return Weights;
            yield return Biases;
        }

        public override Layer Copy()
        {
            return new Dense(NeuronCount, Weights.Copy(), Biases.Copy(), Activation.Copy());
        }

        public override void BuildFromData(Saver.LayerData data)
        {
            if (data.NeuronCount is not null) NeuronCount = data.NeuronCount.Value;

            if (data.Weights is not null) Weights = data.Weights;
            Weights.RestoreGrad();

            Biases = data.Biases;
            Biases.RestoreGrad();

            var activType = Type.GetType(data.Activation);
            if (activType is not null)
            {
                Activation = Activator.CreateInstance(activType) as Activation ?? new Linear();
            }
        }
    }

    public class Conv : Layer
    {
        public int FilterCount { get; private set; }
        public int[] KernelDims { get; private set; } = [];
        public Tensor Kernels { get; private set; } = new();

        public Conv(int filterCount, int[] kernelDims, Activation activation)
        {
            FilterCount = filterCount;
            KernelDims = kernelDims;
            Activation = activation;
        }

        public Conv(int filterCount, int[] kernelDims, Tensor kernels, Tensor biases, Activation activation)
        {
            FilterCount = filterCount;
            KernelDims = kernelDims;
            Kernels = kernels;
            Biases = biases;
            Activation = activation;
        }

        public Conv() { }

        public override void SetUpLayer(Tensor inputFormat)
        {
            Kernels = Tensor.InitKernels(FilterCount, KernelDims, inputFormat.Dimensions[^1]);
            Biases = Tensor.InitBiases(FilterCount);

            var outputDims = new int[inputFormat.Rank];
            outputDims[0] = 1;
            for (int i = 0; i < KernelDims.Length; i++)
            {
                outputDims[i + 1] = inputFormat.Dimensions[i + 1] - KernelDims[i] + 1;
            }
            outputDims[^1] = FilterCount;
            OutputFormat = new(outputDims);
        }

        public override Tensor Forward(Tensor input)
        {
            var output = Tensor.Convolve(input, Kernels, Biases);
            return Activation.Forward(output);
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            yield return Kernels;
            yield return Biases;
        }

        public override Layer Copy()
        {
            return new Conv(FilterCount, [.. KernelDims], Kernels.Copy(), Biases.Copy(), Activation.Copy());
        }

        public override void BuildFromData(Saver.LayerData data)
        {
            if (data.FilterCount is not null) FilterCount = data.FilterCount.Value;
            if (data.KernelDims is not null) KernelDims = data.KernelDims;

            if (data.Kernels is not null) Kernels = data.Kernels;
            Kernels.RestoreGrad();

            if (data.Biases is not null) Biases = data.Biases;
            Biases.RestoreGrad();

            var activType = Type.GetType(data.Activation);
            if (activType is not null)
            {
                Activation = Activator.CreateInstance(activType) as Activation ?? new Linear();
            }
        }
    }

    /// <summary>
    /// Base class for neural network activation functions.
    /// </summary>
    public abstract class Activation
    {
        /// <summary>
        /// Applies the activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the activation function to.</param>
        /// <returns>Tensor containing the result of applying the activation function to the input tensor.</returns>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Creates a new instance of the same activation function.
        /// </summary>
        /// <returns>New instance of the same activation function.</returns>
        public abstract Activation Copy(); // used to ensure function-specific parameters are persisted during copies
    }

    /// <summary>
    /// Rectified Linear Unit activation function.
    /// </summary>
    public class ReLU : Activation
    {
        // Base Activation API overrides

        /// <summary>
        /// Applies the ReLU activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the ReLU activation function to.</param>
        /// <returns>Tensor containing the result of applying the ReLU activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.ReLU(input);

        /// <summary>
        /// Creates a new ReLU activation function instance.
        /// </summary>
        /// <returns>New ReLU activation function instance.</returns>
        public override Activation Copy() => new ReLU();
    }

    /// <summary>
    /// Leaky Rectified Linear Unit activation function.
    /// </summary>
    public class LeakyReLU : Activation
    {
        /// <summary>
        /// Coefficient for negative inputs.
        /// </summary>
        readonly double Tau;

        /// <summary>
        /// Creates a new Leaky ReLU activation function instance.
        /// </summary>
        /// <param name="tau">Coefficient for negative inputs.</param>
        public LeakyReLU(double tau = 0.01)
        {
            Tau = tau;
        }

        /// <summary>
        /// Parameterless constructor for model reconstruction from save data.
        /// </summary>
        public LeakyReLU() { }

        /// <summary>
        /// Applies the Leaky ReLU activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the Leaky ReLU activation function to.</param>
        /// <returns>Tensor containing the result of applying the Leaky ReLU activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.LeakyReLU(input, Tau);

        /// <summary>
        /// Creates a new Leaky ReLU activation function instance with the same Tau value.
        /// </summary>
        /// <returns>New Leaky ReLU activation function instance with the same Tau value.</returns>
        public override Activation Copy() => new LeakyReLU(Tau);
    }

    /// <summary>
    /// Hyperbolic tangent activation function.
    /// </summary>
    public class Tanh : Activation
    {
        /// <summary>
        /// Applies the Tanh activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the Tanh activation function to.</param>
        /// <returns>Tensor containing the result of applying the Tanh activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.Tanh(input);

        /// <summary>
        /// Creates a new Tanh activation function instance.
        /// </summary>
        /// <returns>New Tanh activation function instance.</returns>
        public override Activation Copy() => new Tanh();
    }

    /// <summary>
    /// Sigmoid activation function.
    /// </summary>
    public class Sigmoid : Activation
    {
        /// <summary>
        /// Applies the sigmoid activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the sigmoid activation function to.</param>
        /// <returns>Tensor containing the result of applying the sigmoid activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.Sigmoid(input);

        /// <summary>
        /// Creates a new sigmoid activation function instance.
        /// </summary>
        /// <returns>New sigmoid activation function instance.</returns>
        public override Activation Copy() => new Sigmoid();
    }

    /// <summary>
    /// Linear activation function (does not modify input).
    /// </summary>
    public class Linear : Activation
    {
        /// <summary>
        /// Applies the linear activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the linear activation function to.</param>
        /// <returns>Tensor containing the result of applying the linear activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => input; // does not modify input -> linear function

        /// <summary>
        /// Creates a new linear activation function instance.
        /// </summary>
        /// <returns>New linear activation function instance.</returns>
        public override Activation Copy() => new Linear();
    }

    /// <summary>
    /// Class representing n-dimensional data and autograd graph nodes.
    /// </summary>
    [Serializable]
    public class Tensor
    {
        // Linear value storage
        /// <summary>
        /// Linear array storing all of the tensor's values.
        /// </summary>
        public double[] Data { get; init; } = [];
        /// <summary>
        /// Linear array storing the gradient of each corresponding value.
        /// </summary>
        public double[] Grad { get; private set; } = [];

        // Shape properties
        /// <summary>
        /// Number of dimensions of the tensor.
        /// </summary>
        public int Rank => Dimensions.Length;
        /// <summary>
        /// Number of individual values in the tensor.
        /// </summary>
        public int ElementCount => Data.Length;
        /// <summary>
        /// Number of individual gradients in the tensor.
        /// </summary>
        public int GradCount => Grad.Length;
        /// <summary>
        /// Array containing the length of each of the tensor's dimensions.
        /// </summary>
        public int[] Dimensions { get; init; } = [];

        // Index mapping
        /// <summary>
        /// Array containing the strides over the linear array represented by increments in the coordinate indices along each dimension.
        /// </summary>
        public int[] Strides { get; init; } = [];

        // AutoGrad graph
        /// <summary>
        /// Whether the autograd engine is in graphing or inference mode.
        /// </summary>
        public static bool Inference { get; set; } = false; // controls whether the backward graph is generated during a forward pass
        /// <summary>
        /// Whether it is necessary to calculate the tensor's gradient.
        /// </summary>
        public bool RequiresGrad { get; set; }
        /// <summary>
        /// List of tensors involved in the operation which created this instance.
        /// </summary>
        readonly List<Tensor> _parents = [];
        /// <summary>
        /// List of tensors which were created through operations in which this instance was involved.
        /// </summary>
        readonly List<Tensor> _results = [];
        /// <summary>
        /// Index of the next performed operation involving this tensor in the current autograd graph.
        /// </summary>
        int _opIndex = 0;
        /// <summary>
        /// Gradient calculation function for the parents of this instance.
        /// </summary>
        Action _backward = delegate { };
        /// <summary>
        /// Index of the current autograd graph.
        /// </summary>
        static int _forwardGen = 0;
        /// <summary>
        /// Index of the last autograd graph for which this instance was prepared.
        /// </summary>
        int _lastGen = -1;
        /// <summary>
        /// List of tensors representing the topography of the complete autograd graph.
        /// </summary>
        List<Tensor>? _topo = null;
        /// <summary>
        /// HashSet of all unique autograd tensor nodes visited during topography construction.
        /// </summary>
        HashSet<Tensor>? _visited = null;

        // Optimization parameters
        /// <summary>
        /// Size of vectors in the current CPU architecture.
        /// </summary>
        static readonly int VectorSize = Vector<double>.Count;
        /// <summary>
        /// Minimum tensor element threshold for parallelizing matrix multiplication operator.
        /// </summary>
        const long ParallelThreshold = 500_000;

        // Initialization functions

        /// <summary>
        /// Creates a new tensor instance with the given dimensions.
        /// </summary>
        /// <param name="dims">Array containing the dimensions of the new tensor.</param>
        /// <param name="requiresGrad">Whether it is necessary to calculate the tensor's gradient.</param>
        public Tensor(int[] dims, bool requiresGrad = false)
        {
            // Calculate shape data
            Dimensions = (int[])dims.Clone();
            Strides = ComputeStrides(dims);

            // Calculate linear size
            int size = 1;
            foreach (var dim in dims) size *= dim;

            Data = new double[size]; // initialize linear array of values

            // Initialize linear array of gradients if necessary
            RequiresGrad = requiresGrad;
            if (RequiresGrad) Grad = new double[size];
        }

        /// <summary>
        /// Parameterless constructor for Json serializer.
        /// </summary>
        public Tensor() { }

        /// <summary>
        /// Initializes a new weights tensor using He Initialization.
        /// </summary>
        /// <param name="inputCount">Number of inputs to the layer.</param>
        /// <param name="neuronCount">Number of neurons in the layer.</param>
        /// <returns>Weights tensor with values initialized using He Initialization.</returns>
        public static Tensor InitWeights(int inputCount, int neuronCount)
        {
            Tensor weights = new([inputCount, neuronCount], true);

            double stdDev = Math.Sqrt(2.0 / inputCount);
            for (int i = 0; i < weights.ElementCount; i++)
            {
                weights[i] = MathUtils.NextGaussian(0, stdDev);
            }

            return weights;
        }

        /// <summary>
        /// Initializes a new bias tensor with non-zero values.
        /// </summary>
        /// <param name="neuronCount">Number of neurons in the layer.</param>
        /// <returns>Bias tensor with non-zero values.</returns>
        public static Tensor InitBiases(int neuronCount) => Scalar(0.01, [neuronCount], true);

        public static Tensor InitKernels(int filterCount, int[] kernelDims, int inputChannels)
        {
            var dims = new int[kernelDims.Length + 2];
            dims[0] = filterCount;
            Array.Copy(kernelDims, 0, dims, 1, kernelDims.Length);
            dims[^1] = inputChannels;

            int fanIn = inputChannels;
            foreach (var dim in kernelDims)
            {
                fanIn *= dim;
            }

            Tensor kernels = new(dims, true);

            double stdDev = Math.Sqrt(2.0 / fanIn);
            for (int i = 0; i < kernels.ElementCount; i++)
            {
                kernels[i] = MathUtils.NextGaussian(0, stdDev);
            }

            return kernels;
        }

        /// <summary>
        /// Creates a deep copy of the current tensor.
        /// </summary>
        /// <returns>Deep copy tensor detached from the existing autograd graph.</returns>
        public Tensor Copy()
        {
            Tensor copy = new(Dimensions, false);
            Array.Copy(Data, copy.Data, ElementCount);
            return copy;
        }

        // Indexing functions

        /// <summary>
        /// Calculates the strides over the linear array represented by increments in the indices of each dimension.
        /// </summary>
        /// <param name="dims">Array containing the length of each dimension.</param>
        /// <returns>Array containing the strides over the linear array represented by increments in the indices of each dimension.</returns>
        static int[] ComputeStrides(int[] dims)
        {
            int n = dims.Length;
            var strides = new int[n];

            strides[n - 1] = 1;

            for (int i = n - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * dims[i + 1];
            }

            return strides;
        }

        /// <summary>
        /// Gets the value at the given coordinate indices.
        /// </summary>
        /// <param name="indices">Coordinate indices of the value to access.</param>
        /// <returns>Value at the given coordinate indices.</returns>
        public double this[params int[] indices]
        {
            get => Data[LinearIndex(indices)];
            set => Data[LinearIndex(indices)] = value;
        }

        /// <summary>
        /// Gets the value at the given linear index.
        /// </summary>
        /// <param name="index">Linear index of the value to access.</param>
        /// <returns>Value at the given linear index.</returns>
        public double this[int index]
        {
            get => Data[index];
            set => Data[index] = value;
        }

        // Convert linear index to multidimensional indices
        /// <summary>
        /// Converts a linear index to the corresponding coordinate indices.
        /// </summary>
        /// <param name="index">Linear index to convert.</param>
        /// <returns>Corresponding coordinate indices of the linear index.</returns>
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

        /// <summary>
        /// Converts a linear index to the corresponding indices.
        /// </summary>
        /// <param name="index">Linear index to convert.</param>
        /// <param name="indices">Span to fill with the corresponding coordinate indices.</param>
        void GetFullIndices(int index, Span<int> indices)
        {
            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = index % Dimensions[i];
                index /= Dimensions[i];
            }
        }

        /// <summary>
        /// Converts coordinate indices to the corresponding linear index based on strides.
        /// </summary>
        /// <param name="indices">Coordinate indices to convert.</param>
        /// <returns>Corresponding linear index.</returns>
        public int LinearIndex(params int[] indices)
        {
            int offset = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                offset += indices[i] * Strides[i];
            }

            return offset;
        }

        /// <summary>
        /// Converts a span of coordinate indices to the corresponding linear index based on strides.
        /// </summary>
        /// <param name="indices">Span of coordinate indices to convert.</param>
        /// <returns>Corresponding linear index.</returns>
        int LinearIndex(Span<int> indices)
        {
            int offset = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                offset += indices[i] * Strides[i];
            }

            return offset;
        }

        // Autograd graph functions

        /// <summary>
        /// Creates a new gradient array which matches the size of the linear value array of the tensor.
        /// </summary>
        public void RestoreGrad()
        {
            if (RequiresGrad) Grad = new double[ElementCount];
        }

        /// <summary>
        /// Clears the parent and gradient function data of the tensor.
        /// </summary>
        public void ClearGraph()
        {
            _parents.Clear();
            _backward = delegate { };
        }

        /// <summary>
        /// Increments the current autgrad graph generation.
        /// </summary>
        public static void BeginForward()
        {
            if (!Inference) _forwardGen++;
        }

        /// <summary>
        /// Prepares the autograd graph node for another forward pass.
        /// </summary>
        void PrepareForward()
        {
            if (_lastGen == _forwardGen) return;

            _opIndex = 0;

            foreach (var r in _results)
            {
                r.ClearGraph();
            }

            _lastGen = _forwardGen;
        }

        /// <summary>
        /// Finalizes the autograd graph node's internal data after a forward pass.
        /// </summary>
        void FinalizeForward()
        {
            // Trim any excess result tensor references not used in the last autograd graph
            if (_opIndex < _results.Count)
            {
                _results.RemoveRange(_opIndex, _results.Count - _opIndex);
            }
        }

        /// <summary>
        /// Calculates the gradients of all tensor nodes in the current autograd graph.
        /// </summary>
        public void Backward()
        {
            // Initialize topography and visited node buffers if not yet initialized
            _topo ??= [];
            _visited ??= [];

            // Clear previous topography and visited nodes
            _topo.Clear();
            _visited.Clear();

            BuildTopo(this, _topo, _visited); // build topography of current graph

            // Zero out the gradients of all nodes
            foreach (var t in _topo)
            {
                if (t.RequiresGrad) Array.Clear(t.Grad, 0, t.GradCount);
            }

            Array.Fill(Grad, 1.0); // initialize current node's gradient to 1 (assume Backward() was called on the final node)

            // Iterate backwards from last node in the graph
            for (int i = _topo.Count - 1; i >= 0; i--)
            {
                _topo[i]._backward();
            }

            // Finalize forward pass for each node
            foreach (var t in _topo)
            {
                t.FinalizeForward();
            }
        }

        /// <summary>
        /// Builds the topography of the current autograd graph.
        /// </summary>
        /// <param name="t">Node which is being added to the topography.</param>
        /// <param name="topo">Buffer to store topography in.</param>
        /// <param name="visited">HashSet to track visited nodes in.</param>
        static void BuildTopo(Tensor t, List<Tensor> topo, HashSet<Tensor> visited)
        {
            if (visited.Contains(t)) return; // skip if node appears in graph multiple times
            visited.Add(t);

            // Recursively add all parent nodes to the topography
            foreach (var p in t._parents)
            {
                BuildTopo(p, topo, visited);
            }

            topo.Add(t); // append current node at the end of the topography
        }

        /// <summary>
        /// Creates a tensor instance filled with a single scalar value.
        /// </summary>
        /// <param name="value">Scalar value to fill the new tensor with.</param>
        /// <param name="dims">Dimensions of the new tensor.</param>
        /// <param name="requiresGrad">Whether it is necessary to calculate the gradient of the new tensor.</param>
        /// <returns>New tensor instance filled with the given scalar value.</returns>
        public static Tensor Scalar(double value, int[] dims, bool requiresGrad = false)
        {
            Tensor t = new(dims, requiresGrad);
            Array.Fill(t.Data, value);
            return t;
        }

        // Element-wise algebraic operations

        /// <summary>
        /// Adds two tensors using element-wise addition.
        /// </summary>
        /// <param name="a">First tensor to add.</param>
        /// <param name="b">Second tensor to add.</param>
        /// <returns>Tensor containing the element-wise sum of the two input tensors.</returns>
        public static Tensor operator +(Tensor a, Tensor b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized sum
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] + bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] + b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function for r = a + b -> dr/da = 1, dr/db = 1
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    // Vectorize gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients of parents
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] += rgVecs[i];
                    }

                    // Clean up unvectorized tails
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Adds a scalar to every element in a tensor.
        /// </summary>
        /// <param name="a">Tensor to add.</param>
        /// <param name="b">Scalar to add.</param>
        /// <returns>Tensor containing the result of adding the scalar to every element in the input tensor.</returns>
        public static Tensor operator +(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b); // splat scalar into a vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized sum
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] + vb;
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] + b;
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(a);

                // Gradient calculation function for r = a + b -> dr/da = 1, dr/db = 1
                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    // Vectorize gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Adds a scalar to every element in a tensor.
        /// </summary>
        /// <param name="a">Scalar to add.</param>
        /// <param name="b">Tensor to add.</param>
        /// <returns>Tensor containing the result of adding the scalar to every value in the input tensor.</returns>
        public static Tensor operator +(double a, Tensor b) => b + a; // commutative operation -> a + b = b + a

        /// <summary>
        /// Subtracts a tensor from another tensor using element-wise subtraction.
        /// </summary>
        /// <param name="a">Tensor to subtract from.</param>
        /// <param name="b">Tensor to subtract.</param>
        /// <returns>Tensor containing the element-wise difference of the two input tensors.</returns>
        public static Tensor operator -(Tensor a, Tensor b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized difference
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] - bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] - b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    // Vectorize gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parents
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] -= rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] -= result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Subtracts a scalar from every element in a tensor.
        /// </summary>
        /// <param name="a">Tensor to subtract from.</param>
        /// <param name="b">Scalar to subtract.</param>
        /// <returns>Tensor containing the element-wise difference of the input tensor and input scalar.</returns>
        public static Tensor operator -(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b); // splat scalar into a vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized difference
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] - vb;
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] - b;
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(a);

                // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    // Vectorize gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Subtracts every element in a tensor from a scalar.
        /// </summary>
        /// <param name="a">Scalar to subtract from.</param>
        /// <param name="b">Tensor to subtract.</param>
        /// <returns>Tensor containing the difference of the input scalar and every element in the input tensor.</returns>
        public static Tensor operator -(double a, Tensor b)
        {
            Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);

            // Vectorize inputs and results
            var va = new Vector<double>(a); // splat scalar into a vector
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized difference
            for (int i = 0; i < bVecs.Length; i++)
            {
                rVecs[i] = va - bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a - b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(b);

                // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
                result._backward = () =>
                {
                    if (!b.RequiresGrad) return;

                    // Vectorize gradients
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < bgVecs.Length; i++)
                    {
                        bgVecs[i] -= rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                    {
                        b.Grad[i] -= result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Multiplies two tensors using element-wise multiplication.
        /// </summary>
        /// <param name="a">First tensor to multiply.</param>
        /// <param name="b">Second tensor to multiply.</param>
        /// <returns>Tensor containing the element-wise product of the two input tensors.</returns>
        public static Tensor operator *(Tensor a, Tensor b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized product
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] * bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] * b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function for r = a * b -> dr/da = b; dr/db = a
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parents
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += bvVecs[i] * rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] += avVecs[i] * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += b[i] * result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] += a[i] * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Mutliplies every element in a tensor by a scalar.
        /// </summary>
        /// <param name="a">Tensor to multiply.</param>
        /// <param name="b">Scalar to multiply by.</param>
        /// <returns>Tensor containing the product of every element in the input tensor and the input scalar.</returns>
        public static Tensor operator *(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b); // splat constant into vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized product
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] * vb;
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] * b;
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(a);

                // Gradient calculation function for r = a * b -> dr/da = b; dr/db = a
                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var vb = new Vector<double>(b); // splat scalar into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += vb * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += b * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Multiplies every element in a tensor by a scalar.
        /// </summary>
        /// <param name="a">Scalar to multiply by.</param>
        /// <param name="b">Tensor to multiply.</param>
        /// <returns>Tensor containing the product of every element in the input tensor and the input scalar.</returns>
        public static Tensor operator *(double a, Tensor b) => b * a; // commutative operation -> a * b = b * a

        /// <summary>
        /// Divides a tensor by another tensor using element-wise division.
        /// </summary>
        /// <param name="a">Tensor to divide.</param>
        /// <param name="b">Tensor to divide by.</param>
        /// <returns>Tensor containing the element-wise quotient of the two input tensors.</returns>
        public static Tensor operator /(Tensor a, Tensor b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized quotient
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] / bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] / b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());
                    
                    // Calculate vectorized gradients for parents
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i] / bvVecs[i];
                        if (b.RequiresGrad) bgVecs[i] -= (avVecs[i] / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i] / b[i];
                        if (b.RequiresGrad) b.Grad[i] -= (a[i] / (b[i] * b[i])) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Divides every element in a tensor by a scalar.
        /// </summary>
        /// <param name="a">Tensor to divide.</param>
        /// <param name="b">Scalar to divide by.</param>
        /// <returns>Tensor containing the quotient of every element in the input tensor and the input scalar.</returns>
        public static Tensor operator /(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b); // splat scalar into vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized quotient
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] / vb;
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] / b;
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(a);

                // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var vb = new Vector<double>(b); // splat constant into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i] / vb;
                    }

                    // Clean up unvectorized tail
                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += (1.0 / b) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Divides a scalar by every element in a tensor.
        /// </summary>
        /// <param name="a">Scalar to divide.</param>
        /// <param name="b">Tensor to divide by.</param>
        /// <returns>Tensor containing the quotient of the input scalar and every element in the input tensor.</returns>
        public static Tensor operator /(double a, Tensor b)
        {
            Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);
            
            // Vectorize inputs and results
            var va = new Vector<double>(a); // splat scalar into vector
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized quotient
            for (int i = 0; i < bVecs.Length; i++)
            {
                rVecs[i] = va / bVecs[i];
            }

            // Clean up unvectorized tail
            for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a / b[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(b);

                // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
                result._backward = () =>
                {
                    if (!b.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var va = new Vector<double>(a); // splat scalar into vector
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorize gradients for parent
                    for (int i = 0; i < bgVecs.Length; i++)
                    {
                        bgVecs[i] -= (va / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                    {
                        b.Grad[i] -= (a / (b[i] * b[i])) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Raises a tensor to the power of another tensor using element-wise exponentiation.
        /// </summary>
        /// <param name="a">Tensor to exponentiate.</param>
        /// <param name="exp">Tensor to raise to the power of.</param>
        /// <returns>Tensor containing the element-wise power of the two input tensors.</returns>
        public static Tensor Pow(Tensor a, Tensor exp)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || exp.RequiresGrad);

            // Calculate power sequentially - no general exponentiation vectorization available
            for (int i = 0; i < result.ElementCount; i++)
            {
                result[i] = Math.Pow(a[i], exp[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, exp]);

                // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !exp.RequiresGrad) return;

                    // Calculate gradients for parents sequentially - no general exponentiation vectorization available
                    for (int i = 0; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += exp[i] * Math.Pow(a[i], exp[i] - 1.0) * result.Grad[i];
                        if (exp.RequiresGrad) exp.Grad[i] += result[i] * Math.Log(a[i]) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Raises every element in a tensor to the power of a scalar.
        /// </summary>
        /// <param name="a">Tensor to exponentiate.</param>
        /// <param name="exp">Scalar to raise to the power of.</param>
        /// <returns>Tensor containing the power of every element in the input tensor and the input scalar.</returns>
        public static Tensor Pow(Tensor a, double exp)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            // Special vectorization cases - power of 2 and square root
            if (exp == 2.0 || exp == 0.5)
            {
                // Vectorize inputs and results
                var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

                // Calculate vectorized power
                for (int i = 0; i < aVecs.Length; i++)
                {
                    rVecs[i] = exp == 2.0 ? aVecs[i] * aVecs[i] : Vector.SquareRoot(aVecs[i]);
                }

                // Clean up unvectorized tail
                for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    result[i] = exp == 2.0 ? a[i] * a[i] : Math.Sqrt(a[i]);
                }
            }
            else // Calculate power sequentially - no general exponentiation vectorization available
            {
                for (int i = 0; i < result.ElementCount; i++)
                {
                    result[i] = Math.Pow(a[i], exp);
                }
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(a);

                // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    // Special vectorization case - power of 2 and square root
                    if (exp == 2.0 || exp == 0.5)
                    {
                        // Vectorize inputs and gradients
                        var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                        var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                        var vexp = new Vector<double>(exp); // splat scalar into vector
                        var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                        // Calculate vectorized gradients of parent
                        for (int i = 0; i < agVecs.Length; i++)
                        {
                            agVecs[i] += exp == 2.0 ? vexp * avVecs[i] * rgVecs[i] : (vexp / Vector.SquareRoot(avVecs[i])) * rgVecs[i];
                        }

                        // Clean up unvectorized tail
                        for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                        {
                            a.Grad[i] += exp * Math.Pow(a[i], exp - 1.0) * result.Grad[i];
                        }
                    }
                    else // calculate gradients of parent sequentially - no general exponentiation vectorization available
                    {
                        for (int i = 0; i < a.ElementCount; i++)
                        {
                            a.Grad[i] += exp * Math.Pow(a[i], exp - 1.0) * result.Grad[i];
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Raises a scalar to the power of every element in a tensor.
        /// </summary>
        /// <param name="a">Scalar to exponentiate.</param>
        /// <param name="exp">Tensor to raise to the power of.</param>
        /// <returns>Tensor containing the power of the input scalar and every element in the input tensor.</returns>
        public static Tensor Pow(double a, Tensor exp)
        {
            Tensor result = GetResultTensor(exp, exp.Dimensions, exp.RequiresGrad);

            // Calculate power sequentially - no general exponentiation vectorization available
            for (int i = 0; i < result.ElementCount; i++)
            {
                result[i] = Math.Pow(a, exp[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(exp);

                // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
                result._backward = () =>
                {
                    if (!exp.RequiresGrad) return;

                    // Vectorize inputs, results, and gradients - possible due to derivative only involving a^b (already calculated) and ln(a) (scalar)
                    var expgVecs = MemoryMarshal.Cast<double, Vector<double>>(exp.Grad.AsSpan());
                    double lna = Math.Log(a); // precalculate ln(a)
                    var vlna = new Vector<double>(lna); // splat ln(a) into vector
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < expgVecs.Length; i++)
                    {
                        expgVecs[i] += rvVecs[i] * vlna * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = expgVecs.Length * VectorSize; i < exp.ElementCount; i++)
                    {
                        exp.Grad[i] += result[i] * lna * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Raises e to the power of every element in the tensor.
        /// </summary>
        /// <param name="t">Tensor to raise e to the power of.</param>
        /// <returns>Tensor containing the power of e and every element in the input tensor.</returns>
        public static Tensor Exp(Tensor t)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized powers of e
            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Exp(tVecs[i]);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Exp(t[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = e^t -> dr/dt = e^t
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize results and gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        tgVecs[i] += rvVecs[i] * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += result[i] * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Takes the element-wise logarithm of two tensors.
        /// </summary>
        /// <param name="logBase">Tensor to use as logarithm base.</param>
        /// <param name="arg">Tensor to take logarithm of.</param>
        /// <returns>Tensor containing the element-wise logarithm of the two input tensors.</returns>
        public static Tensor Log(Tensor logBase, Tensor arg)
        {
            Tensor result = GetResultTensor(logBase, logBase.Dimensions, logBase.RequiresGrad || arg.RequiresGrad);

            // Vectorize inputs and results
            var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
            var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized logarithms - using log base change formula -> log_lb(arg) = ln(arg) / ln(lb)
            for (int i = 0; i < lbVecs.Length; i++)
            {
                rVecs[i] = Vector.Log(argVecs[i]) / Vector.Log(lbVecs[i]);
            }

            // Clean up unvectorized tail
            for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg[i], logBase[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([logBase, arg]);

                // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
                result._backward = () =>
                {
                    if (!logBase.RequiresGrad && !arg.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                    var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                    var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                    var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parents
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var lnb = Vector.Log(lbvVecs[i]);
                        if (logBase.RequiresGrad)
                        {
                            lbgVecs[i] -= (Vector.Log(argvVecs[i]) / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                        }
                        if (arg.RequiresGrad) arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnb);
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        double lnb = Math.Log(logBase[i]);
                        if (logBase.RequiresGrad) logBase.Grad[i] -= (Math.Log(arg[i]) / (logBase[i] * (lnb * lnb))) * result.Grad[i];
                        if (arg.RequiresGrad) arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Takes the logarithm of a scalar with the base of every element in a tensor.
        /// </summary>
        /// <param name="logBase">Tensor to use as logarithm base.</param>
        /// <param name="arg">Scalar to take logarithm of.</param>
        /// <returns>Tensor containing the logarithm of the input scalar with the base of every element in the input tensor.</returns>
        public static Tensor Log(Tensor logBase, double arg)
        {
            Tensor result = GetResultTensor(logBase, logBase.Dimensions, logBase.RequiresGrad);

            // Vectorize inputs and results
            var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
            var lnvarg = Vector.Log(new Vector<double>(arg));
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized logarithm
            for (int i = 0; i < lbVecs.Length; i++)
            {
                rVecs[i] = lnvarg / Vector.Log(lbVecs[i]);
            }

            // Clean up unvectorized tail
            for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg, logBase[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(logBase);

                // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
                result._backward = () =>
                {
                    if (!logBase.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                    var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                    double lnarg = Math.Log(arg); // precalculate ln(arg)
                    var lnvarg = new Vector<double>(lnarg); // splat scalar into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var lnb = Vector.Log(lbvVecs[i]);
                        lbgVecs[i] -= (lnvarg / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        double lnb = Math.Log(logBase[i]);
                        logBase.Grad[i] -= (lnarg / (logBase[i] * (lnb * lnb))) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Takes the element-wise logarithm with a scalar base of a tensor.
        /// </summary>
        /// <param name="logBase">Scalar to use as logarithm base.</param>
        /// <param name="arg">Tensor to take logarithm of.</param>
        /// <returns>Tensor containing the element-wise logarithm with the input scalar base of the input tensor.</returns>
        public static Tensor Log(double logBase, Tensor arg)
        {
            Tensor result = GetResultTensor(arg, arg.Dimensions, arg.RequiresGrad);

            // Vectorize inputs and results
            var lnvlb = Vector.Log(new Vector<double>(logBase)); // splat scalar into vector
            var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized logarithms
            for (int i = 0; i < argVecs.Length; i++)
            {
                rVecs[i] = Vector.Log(argVecs[i]) / lnvlb;
            }

            // Clean up unvectorized tail
            for (int i = argVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg[i], logBase);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(arg);

                // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
                result._backward = () =>
                {
                    if (!arg.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    double lnb = Math.Log(logBase); // precalculate ln(lb)
                    var lnvlb = new Vector<double>(lnb); // splat scalar into vector
                    var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                    var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnvlb);
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Matrix multiplication

        /// <summary>
        /// Multiplies two tensors using 2D matrix multiplication.
        /// </summary>
        /// <param name="a">First tensor to multiply.</param>
        /// <param name="b">Second tensor to multiply.</param>
        /// <returns>Tensor containing the matrix multiplication product of the two input tensors.</returns>
        public static Tensor operator ^(Tensor a, Tensor b)
        {
            // Extract "outer" and "inner" 2D matrix dimensions - preceding dimensions represent "batch" dimensions
            int rank = a.Rank;
            int m = a.Dimensions[^2]; // "outer" dimension of A
            int n = a.Dimensions[^1]; // shared "inner" dimension
            int p = b.Dimensions[^1]; // "outer" dimension of B

            bool bBatched = b.Rank == a.Rank; // whether B contains a separate 2D matrix per batch of A

            // Compute number of batches and linear sizes of inputs and outputs
            int batchSize = 1;
            for (int i = 0; i < rank - 2; i++) batchSize *= a.Dimensions[i];
            int aMatSize = m * n;
            int bMatSize = n * p;
            int rMatSize = m * p;

            int totalRows = batchSize * m;

            // Build result dimensions -> batch dimensions + outer 2D matrix dimensions
            var resultDims = (int[])a.Dimensions.Clone();
            resultDims[^1] = p;

            Tensor result = GetResultTensor(a, resultDims, a.RequiresGrad || b.RequiresGrad);

            // Calculate matrix multiplication product
            bool useParallel = (long)totalRows * n * p > ParallelThreshold; // whether inputs are large enough to warrant parallelizing (multithreading)
            double[] bT = ArrayPool<double>.Shared.Rent(bMatSize * batchSize); // Rent buffer for transposition of B - avoid garbage collector
            try
            {
                // Transpose B - columns -> rows - dot product calculated along contiguous rows of A and contiguous rows of B - reduces cache misses
                for (int batch = 0; batch < batchSize; batch++)
                {
                    int bSrcOff = bBatched ? batch * bMatSize : 0; // duplicates B's data if same 2D matrix should be used for each batch
                    TransposeMatrix(b.Data, bT, bSrcOff, batch * bMatSize, n, p);
                }

                // Parallelize if inputs large enough
                if (useParallel)
                {
                    Parallel.For(0, batchSize * m, row =>
                    {
                        int batch = row / m;
                        int i = row % m;
                        ComputeRow(i, n, p, a.Data, bT, result.Data, batch * aMatSize, batch * bMatSize, batch * rMatSize);
                    });
                }
                else // calculate product sequentially
                {
                    for (int batch = 0; batch < batchSize; batch++)
                    {
                        int aOff = batch * aMatSize;
                        int bTOff = batch * bMatSize;
                        int rOff = batch * rMatSize;

                        for (int i = 0; i < m; i++)
                        {
                            ComputeRow(i, n, p, a.Data, bT, result.Data, aOff, bTOff, rOff);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(bT); // release B transposition buffer
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function for R = A ^ B (2D matmul) -> grad_A = grad_C ^ B^T; grad_B = A^T ^ grad_C
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    for (int batch = 0; batch < batchSize; batch++)
                    {
                        int aOff = batch * aMatSize;
                        int bOff = bBatched ? batch * bMatSize : 0; // ensure B gradients accumulate correctly if B was broadcasted during forward pass
                        int rOff = batch * rMatSize;
                        bool par = (long)m * n * p > ParallelThreshold;

                        if (a.RequiresGrad && b.RequiresGrad) // branch if both parent gradients needed
                        {
                            // Rent buffers for transpositions - avoid garbage collector
                            double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                            double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                            try
                            {
                                TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                                TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p); // transpose grad_C for contiguous memory access

                                // Parallelize if inputs large enough
                                if (par)
                                {
                                    // Calculate grad_A in parallel
                                    Parallel.For(0, m, i =>
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            // Use untransposed B for contiguous memory access - reduce cache misses
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                        }
                                    });

                                    // Calculate grad_B in parallel
                                    Parallel.For(0, n, k =>
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    });
                                }
                                else // calculate gradients sequentially
                                {
                                    // Calculate grad_A sequentially
                                    for (int i = 0; i < m; i++)
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            // Use untransposed B for contiguous memory access - reduce cache misses
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                        }
                                    }

                                    // Calculate grad_B sequentially
                                    for (int k = 0; k < n; k++)
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                // Release transposition buffers
                                ArrayPool<double>.Shared.Return(aT);
                                ArrayPool<double>.Shared.Return(dOutT);
                            }
                        }
                        else if (a.RequiresGrad) // branch if only grad_A is needed
                        {
                            // Parallelize if inputs large enough
                            if (par)
                            {
                                Parallel.For(0, m, i =>
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        // Use untransposed B for contiguous memory access - reduce cache misses
                                        a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                    }
                                });
                            }
                            else // calculate gradients sequentially
                            {
                                for (int i = 0; i < m; i++)
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        // Use untransposed B for contiguous memory access - reduce cache misses
                                        a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                    }
                                }
                            }
                        }
                        else if (b.RequiresGrad) // branch if only grad_B is needed
                        {
                            // Rent buffers for transpositions - avoid garbage collector
                            double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                            double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                            try
                            {
                                TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                                TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p); // transpose grad_C for contiguous memory access

                                // Parallelize if inputs large enough
                                if (par)
                                {
                                    Parallel.For(0, n, k =>
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    });
                                }
                                else // calculate gradients sequentially
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                // Release transposition buffers
                                ArrayPool<double>.Shared.Return(aT);
                                ArrayPool<double>.Shared.Return(dOutT);
                            }
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Transposes a 2D matrix, switching its rows and columns.
        /// </summary>
        /// <param name="src">Linear array buffer of matrix to transpose.</param>
        /// <param name="dst">Linear array buffer to write transposed matrix to.</param>
        /// <param name="srcOff">Offset of the first element to transpose in the source buffer.</param>
        /// <param name="dstOff">Offset of the first element to write to in the destination buffer.</param>
        /// <param name="rows">Number of rows of the source matrix.</param>
        /// <param name="cols">Number of columns of the source matrix.</param>
        static void TransposeMatrix(double[] src, double[] dst, int srcOff, int dstOff, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    dst[dstOff + c * rows + r] = src[srcOff + r * cols + c]; // switch row and column indices of data
                }
            }
        }

        /// <summary>
        /// Computes a single row of the result matrix of a 2D matrix multiplication.
        /// </summary>
        /// <param name="i">Index of the row to compute.</param>
        /// <param name="n">Inner dimension of the input matrices.</param>
        /// <param name="p">Outer dimension of the second input matrix.</param>
        /// <param name="a">Linear array buffer of the data of the first input matrix</param>
        /// <param name="bT">Linear array buffer of the data of the transposition of the second input matrix.</param>
        /// <param name="r">Linear array buffer of the result matrix to write to.</param>
        /// <param name="aOff">Linear offset of the current row in the first input matrix.</param>
        /// <param name="bTOff">Linear offset of the current row in the second input matrix.</param>
        /// <param name="rOff">Linear offset of the current row in the result matrix.</param>
        static void ComputeRow(int i, int n, int p, double[] a, double[] bT, double[] r, int aOff, int bTOff, int rOff)
        {
            for (int j = 0; j < p; j++) // iterate across row
            {
                // Calculate dot product at current position
                r[rOff + i * p + j] = DotProduct(a, bT, aOff + i * n, bTOff + j * n, n);
            }
        }

        public static Tensor Convolve(Tensor input, Tensor kernels, Tensor biases)
        {
            int batch = input.Dimensions[0];
            int spatialRank = kernels.Rank - 2;
            int filterCount = kernels.Dimensions[0];
            int inputChannels = kernels.Dimensions[^1];

            var outSpatialDims = new int[spatialRank];
            for (int i = 0; i < spatialRank; i++)
            {
                outSpatialDims[i] = input.Dimensions[i + 1] - kernels.Dimensions[i + 1] + 1;
            }

            int inSpatialSize = 1;
            for (int i = 1; i <= spatialRank; i++)
            {
                inSpatialSize *= input.Dimensions[i];
            }

            int outSpatialSize = 1;
            foreach (var dim in outSpatialDims)
            {
                outSpatialSize *= dim;
            }

            int kernelSpatialSize = 1;
            for (int i = 1; i <= spatialRank; i++)
            {
                kernelSpatialSize *= kernels.Dimensions[i];
            }

            int kernelVolumeSize = kernelSpatialSize * inputChannels;

            var inSpatialStrides = new int[spatialRank];
            inSpatialStrides[^1] = 1;
            for (int i = spatialRank - 2; i >= 0; i--)
            {
                inSpatialStrides[i] = inSpatialStrides[i + 1] * input.Dimensions[i + 2];
            }

            var outSpatialStrides = new int[spatialRank];
            outSpatialStrides[^1] = 1;
            for (int i = spatialRank - 2; i >= 0; i--)
            {
                outSpatialStrides[i] = outSpatialStrides[i + 1] * outSpatialDims[i + 1];
            }

            var kernelSpatialStrides = new int[spatialRank];
            kernelSpatialStrides[^1] = 1;
            for (int i = spatialRank - 2; i >= 0; i--)
            {
                kernelSpatialStrides[i] = kernelSpatialStrides[i + 1] * kernels.Dimensions[i + 2];
            }

            var resultDims = new int[input.Rank];
            resultDims[0] = batch;
            Array.Copy(outSpatialDims, 0, resultDims, 1, outSpatialDims.Length);
            resultDims[^1] = filterCount;

            var result = GetResultTensor(input, resultDims, input.RequiresGrad || kernels.RequiresGrad || biases.RequiresGrad);

            bool useParallel = (long)batch * outSpatialSize * filterCount * kernelVolumeSize > ParallelThreshold;

            if (useParallel)
            {
                Parallel.For(0, batch * outSpatialSize, i => ComputeOutputPosition(i, spatialRank, outSpatialSize, filterCount,
                    kernelSpatialSize, inputChannels, outSpatialStrides, kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides,
                    input.Data, kernels.Data, biases.Data, result.Data));
            }
            else
            {
                for (int i = 0; i < batch * outSpatialSize; i++)
                {
                    ComputeOutputPosition(i, spatialRank, outSpatialSize, filterCount, kernelSpatialSize, inputChannels, outSpatialStrides,
                        kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Data, kernels.Data, biases.Data, 
                        result.Data);
                }
            }

            if (!Inference)
            {
                result._parents.AddRange([input, kernels, biases]);

                result._backward = () =>
                {
                    bool par = (long)batch * outSpatialSize * filterCount * kernelVolumeSize > ParallelThreshold;

                    if (biases.RequiresGrad)
                    {
                        for (int f = 0; f < filterCount; f++)
                        {
                            ComputeBiasGrad(f, filterCount, result.Grad, biases.Grad);
                        }
                    }

                    if (kernels.RequiresGrad)
                    {
                        if (par)
                        {
                            Parallel.For(0, filterCount * kernelSpatialSize, fkp => ComputeKernelGrad(fkp, spatialRank,
                                batch, outSpatialSize, filterCount, kernelSpatialSize, inputChannels, outSpatialStrides,
                                kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Data, kernels.Grad,
                                result.Grad));
                        }
                        else
                        {
                            for (int fkp = 0; fkp < filterCount * kernelSpatialSize; fkp++)
                            {
                                ComputeKernelGrad(fkp, spatialRank, batch, outSpatialSize, filterCount, kernelSpatialSize,
                                    inputChannels, outSpatialStrides, kernelSpatialStrides, input.Strides, kernels.Strides,
                                    result.Strides, input.Data, kernels.Grad, result.Grad);
                            }
                        }
                    }

                    if (input.RequiresGrad)
                    {
                        if (par)
                        {
                            Parallel.For(0, batch * inSpatialSize, batchInPos => ComputeInputGrad(batchInPos, spatialRank,
                                inSpatialSize, filterCount, kernelSpatialSize, inputChannels, inSpatialStrides, kernelSpatialStrides,
                                outSpatialDims, outSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Grad, kernels.Data,
                                result.Grad));
                        }
                        else
                        {
                            for (int batchInPos = 0; batchInPos < batch * inSpatialSize; batchInPos++)
                            {
                                ComputeInputGrad(batchInPos, spatialRank, inSpatialSize, filterCount, kernelSpatialSize, inputChannels,
                                    inSpatialStrides, kernelSpatialStrides, outSpatialDims, outSpatialStrides, input.Strides, kernels.Strides,
                                    result.Strides, input.Grad, kernels.Data, result.Grad);
                            }
                        }
                    }
                };
            }

            return result;
        }

        static void ComputeOutputPosition(int batchOutPos, int spatialRank, int outSpatialSize, int filterCount, int kernelSpatialSize,
            int inputChannels, int[] outSpatialStrides, int[] kernelSpatialStrides, int[] inputStrides, int[] kernelStrides, int[] resultStrides,
            double[] inputData, double[] kernelData, double[] biasData, double[] resultData)
        {
            int b = batchOutPos / outSpatialSize;

            Span<int> outCoords = stackalloc int[spatialRank];
            int rem = batchOutPos % outSpatialSize;
            for (int i = 0; i < spatialRank; i++)
            {
                outCoords[i] = rem / outSpatialStrides[i];
                rem %= outSpatialStrides[i];
            }

            Span<int> kernelCoords = stackalloc int[spatialRank];
            Span<double> sums = stackalloc double[filterCount];
            biasData.AsSpan(0, filterCount).CopyTo(sums);
            int inputOffsetBase = b * inputStrides[0];
            int kernelOffsetBase = kernelSpatialSize * inputChannels;
            for (int kp = 0; kp < kernelSpatialSize; kp++)
            {
                rem = kp;
                for (int i = 0; i < spatialRank; i++)
                {
                    kernelCoords[i] = rem / kernelSpatialStrides[i];
                    rem %= kernelSpatialStrides[i];
                }

                int inputOffset = inputOffsetBase;
                for (int i = 0; i < spatialRank; i++)
                {
                    inputOffset += (outCoords[i] + kernelCoords[i]) * inputStrides[i + 1];
                }

                for (int f = 0; f < filterCount; f++)
                {
                    int kernelOffset = kernelOffsetBase * f;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        kernelOffset += kernelCoords[i] * kernelStrides[i + 1];
                    }

                    sums[f] += DotProduct(inputData, kernelData, inputOffset, kernelOffset, inputChannels);
                }
            }

            int resultOffset = b * resultStrides[0];
            for (int i = 0; i < spatialRank; i++)
            {
                resultOffset += outCoords[i] * resultStrides[i + 1];
            }
            sums.CopyTo(resultData.AsSpan(resultOffset, filterCount));
        }

        static void ComputeBiasGrad(int f, int filterCount, double[] resultGrad, double[] biasGrad)
        {
            double sum = 0.0;
            for (int i = f; i < resultGrad.Length; i += filterCount)
            {
                sum += resultGrad[i];
            }
            biasGrad[f] += sum;
        }

        static void ComputeKernelGrad(int fkp, int spatialRank, int batch, int outSpatialSize, int filterCount,
            int kernelSpatialSize, int inputChannels, int[] outSpatialStrides, int[] kernelSpatialStrides,
            int[] inputStrides, int[] kernelStrides, int[] resultStrides, double[] inputData, double[] kernelGrad,
            double[] resultGrad)
        {
            int f = fkp / kernelSpatialSize;

            Span<int> kernelCoords = stackalloc int[spatialRank];
            int rem = fkp % kernelSpatialSize;
            for (int i = 0; i < spatialRank; i++)
            {
                kernelCoords[i] = rem / kernelSpatialStrides[i];
                rem %= kernelSpatialStrides[i];
            }

            int kernelOffset = f * kernelSpatialSize * inputChannels;
            for (int i = 0; i < spatialRank; i++)
            {
                kernelOffset += kernelCoords[i] * kernelStrides[i + 1];
            }

            Span<int> outCoords = stackalloc int[spatialRank];
            for (int b = 0; b < batch; b++)
            {
                int inputOffsetBase = b * inputStrides[0];
                int resultOffsetBase = b * resultStrides[0] + f;

                for (int op = 0; op < outSpatialSize; op++)
                {
                    rem = op;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        outCoords[i] = rem / outSpatialStrides[i];
                        rem %= outSpatialStrides[i];
                    }

                    int inputOffset = inputOffsetBase;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        inputOffset += (outCoords[i] + kernelCoords[i]) * inputStrides[i + 1];
                    }

                    int resultOffset = resultOffsetBase;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        resultOffset += outCoords[i] * resultStrides[i + 1];
                    }
                    double dOut = resultGrad[resultOffset];

                    var kgVecs = MemoryMarshal.Cast<double, Vector<double>>(kernelGrad.AsSpan(kernelOffset, inputChannels));
                    var inVecs = MemoryMarshal.Cast<double, Vector<double>>(inputData.AsSpan(inputOffset, inputChannels));
                    var vdOut = new Vector<double>(dOut);

                    for (int i = 0; i < kgVecs.Length; i++)
                    {
                        kgVecs[i] += inVecs[i] * vdOut;
                    }

                    for (int i = kgVecs.Length * VectorSize; i < inputChannels; i++)
                    {
                        kernelGrad[kernelOffset + i] += inputData[inputOffset + i] * dOut;
                    }
                }
            }
        }

        static void ComputeInputGrad(int batchInPos, int spatialRank, int inSpatialSize, int filterCount, int kernelSpatialSize,
            int inputChannels, int[] inSpatialStrides, int[] kernelSpatialStrides, int[] outSpatialDims, int[] outSpatialStrides,
            int[] inputStrides, int[] kernelStrides, int[] resultStrides, double[] inputGrad, double[] kernelData, double[] resultGrad)
        {
            int b = batchInPos / inSpatialSize;

            Span<int> inCoords = stackalloc int[spatialRank];
            int rem = batchInPos % inSpatialSize;
            for (int i = 0; i < spatialRank; i++)
            {
                inCoords[i] = rem / inSpatialStrides[i];
                rem %= inSpatialStrides[i];
            }

            int inputOffset = b * inputStrides[0];
            for (int i = 0; i < spatialRank; i++)
            {
                inputOffset += inCoords[i] * inputStrides[i + 1];
            }

            Span<int> kernelCoords = stackalloc int[spatialRank];
            int resultOffsetBase = b * resultStrides[0];
            for (int f = 0; f < filterCount; f++)
            {
                int kernelOffsetBase = f * kernelSpatialSize * inputChannels;
                for (int kp = 0; kp < kernelSpatialSize; kp++)
                {
                    rem = kp;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        kernelCoords[i] = rem / kernelSpatialStrides[i];
                        rem %= kernelSpatialStrides[i];
                    }

                    bool valid = true;
                    int resultOffset = resultOffsetBase + f;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        int outCoord = inCoords[i] - kernelCoords[i];
                        if (outCoord < 0 || outCoord >= outSpatialDims[i])
                        {
                            valid = false;
                            break;
                        }
                        resultOffset += outCoord * resultStrides[i + 1];
                    }
                    if (!valid) continue;

                    double dOut = resultGrad[resultOffset];

                    int kernelOffset = kernelOffsetBase;
                    for (int i = 0; i < spatialRank; i++)
                    {
                        kernelOffset += kernelCoords[i] * kernelStrides[i + 1];
                    }

                    var igVecs = MemoryMarshal.Cast<double, Vector<double>>(inputGrad.AsSpan(inputOffset, inputChannels));
                    var kVecs = MemoryMarshal.Cast<double, Vector<double>>(kernelData.AsSpan(kernelOffset, inputChannels));
                    var vdOut = new Vector<double>(dOut);

                    for (int i = 0; i < igVecs.Length; i++)
                    {
                        igVecs[i] += kVecs[i] * vdOut;
                    }

                    for (int i = igVecs.Length * VectorSize; i < inputChannels; i++)
                    {
                        inputGrad[inputOffset + i] += kernelData[kernelOffset + i] * dOut;
                    }
                }
            }
        }

        /// <summary>
        /// Calculates the dot product of a subrange of two vectors.
        /// </summary>
        /// <param name="x">First input vector.</param>
        /// <param name="y">Second input vector.</param>
        /// <param name="xOff">Offset of the subrange in the first input vector.</param>
        /// <param name="yOff">Offset of the subrange in the second input vector.</param>
        /// <param name="len">Length of the subrange.</param>
        /// <returns>Dot product of the given subrange of the two input vectors.</returns>
        static double DotProduct(double[] x, double[] y, int xOff, int yOff, int len)
        {
            // Vectorize inputs
            var xVecs = MemoryMarshal.Cast<double, Vector<double>>(x.AsSpan(xOff, len));
            var yVecs = MemoryMarshal.Cast<double, Vector<double>>(y.AsSpan(yOff, len));

            var acc = Vector<double>.Zero; // product/sum vector

            // Calculate vectorized product/sum
            for (int i = 0; i < xVecs.Length; i++)
            {
                acc += xVecs[i] * yVecs[i];
            }
            double sum = Vector.Sum(acc);

            // Clean up unvectorized tail
            for (int i = xVecs.Length * VectorSize; i < len; i++)
            {
                sum += x[xOff + i] * y[yOff + i];
            }

            return sum;
        }

        // Activation functions

        /// <summary>
        /// Applies the Rectified Linear Unit function to the input tensor.
        /// </summary>
        /// <param name="t">Tensor to apply ReLU function to.</param>
        /// <returns>Result tensor of applying ReLU function to the input tensor.</returns>
        public static Tensor ReLU(Tensor t) // ReLU(t) = t (t > 0), 0 (t <= 0)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vzero = Vector<double>.Zero; // preallocate 0's vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized ReLU result
            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Max(tVecs[i], vzero);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Max(t[i], 0.0);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = ReLU(t) -> dr/dt = 1 (t > 0), 0 (t <= 0)
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vzero = Vector<double>.Zero; // preallocate 0's vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients of parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var mask = Vector.GreaterThan(tvVecs[i], vzero);
                        tgVecs[i] += Vector.ConditionalSelect(mask, rgVecs[i], vzero);
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += (t[i] > 0.0 ? 1.0 : 0.0) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Applies the Leaky Rectified Linear Unit function to the input tensor.
        /// </summary>
        /// <param name="t">Tensor to apply Leaky ReLU function to.</param>
        /// <param name="tau">Tau coefficient to use.</param>
        /// <returns>Result tensor of applying Leaky ReLU function to the input tensor.</returns>
        public static Tensor LeakyReLU(Tensor t, double tau) // Leaky ReLU(t) = t (t > 0), Tau * t (t <= 0)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vzero = Vector<double>.Zero; // preallocate 0's vector
            var vtau = new Vector<double>(tau); // splat scalar into vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
            
            // Calculate vectorized Leaky ReLU result
            for (int i = 0; i < tVecs.Length; i++)
            {
                var mask = Vector.GreaterThan(tVecs[i], vzero);
                rVecs[i] = Vector.ConditionalSelect(mask, tVecs[i], vtau * tVecs[i]);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = t[i] > 0.0 ? t[i] : tau * t[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = Leaky ReLU(t) -> dr/dt = 1 (t > 0), Tau (t <= 0)
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vzero = Vector<double>.Zero; // preallocate 0's vector
                    var vone = Vector<double>.One; // preallocate 1's vector
                    var vtau = new Vector<double>(tau); // splat scalar into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients of parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var mask = Vector.GreaterThan(tvVecs[i], vzero);
                        tgVecs[i] += Vector.ConditionalSelect(mask, vone, vtau) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += (t[i] > 0.0 ? 1.0 : tau) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Applies the sigmoid activation function to the input tensor.
        /// </summary>
        /// <param name="t">Tensor to apply sigmoid function to.</param>
        /// <returns>Result tensor of applying sigmoid function to the input tensor.</returns>
        public static Tensor Sigmoid(Tensor t) // Sigmoid(t) = 1 / (1 + e^(-t))
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vone = Vector<double>.One; // preallocate 1's vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized sigmoid result
            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = vone / (vone + Vector.Exp(-tVecs[i]));
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = MathUtils.Sigmoid(t[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = Sigmoid(t) -> dr/dt = Sigmoid(t) * (1 - Sigmoid(t))
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize inputs, results, and gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vone = Vector<double>.One; // preallocate 1's vector
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        // Reuse forward results as Sigmoid(t)
                        tgVecs[i] += rvVecs[i] * (vone - rvVecs[i]) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        // Reuse forward results as Sigmoid(t)
                        t.Grad[i] += result[i] * (1.0 - result[i]) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Applies the hyperbolic tangent activation function to the input tensor.
        /// </summary>
        /// <param name="t">Tensor to apply Tanh function to.</param>
        /// <returns>Result tensor of applying Tanh function to the input tensor.</returns>
        public static Tensor Tanh(Tensor t) // Tanh(t) = (e^2t - 1) / (e^2t + 1)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vone = Vector<double>.One; // preallocate 1's vector
            var vtwo = new Vector<double>(2.0); // preallocate 2's vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectoried Tanh result
            for (int i = 0; i < tVecs.Length; i++)
            {
                var e2t = Vector.Exp(vtwo * tVecs[i]);
                rVecs[i] = (e2t - vone) / (e2t + vone);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = MathUtils.Tanh(t[i]);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = Tanh(t) -> dr/dt = 1 - Tanh^2(t)
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize inputs, results, and gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vone = Vector<double>.One; // preallocate 1's tensor
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients of parent
                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        // Reuse forward results as Tanh(t)
                        tgVecs[i] += (vone - (rvVecs[i] * rvVecs[i])) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        // Reuse forward results as Tanh(t)
                        t.Grad[i] += (1.0 - (result[i] * result[i])) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Applies the softmax activation function to the input tensor.
        /// </summary>
        /// <param name="t">Tensor to apply softmax function to.</param>
        /// <returns>Result tensor of applying softmax function to the input tensor.</returns>
        public static Tensor Softmax(Tensor t) // Softmax(z_i) = e^z_i / sum(j = 1 to n) [e^z_j]
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            int classes = t.Dimensions[^1]; // number of classification probabilities per output
            int batchSize = t.ElementCount / classes; // number of batches of separate outputs

            // Apply softmax function to each batch
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * classes; // linear offset of current batch

                // Vectorize inputs and results
                var tSlice = t.Data.AsSpan(offset, classes);
                var tVecs = MemoryMarshal.Cast<double, Vector<double>>(tSlice);
                var rSlice = result.Data.AsSpan(offset, classes);
                var rVecs = MemoryMarshal.Cast<double, Vector<double>>(rSlice);

                // Find vectorized maximum value in batch
                var vmax = new Vector<double>(double.MinValue);
                for (int i = 0; i < tVecs.Length; i++)
                {
                    vmax = Vector.Max(vmax, tVecs[i]);
                }

                // Find single maximum value in vectorized maximum
                double max = double.MinValue;
                for (int lane = 0; lane < VectorSize; lane++)
                {
                    max = Math.Max(max, vmax[lane]);
                }

                // Clean up unvectorized tail
                for (int i = tVecs.Length * VectorSize; i < classes; i++)
                {
                    max = Math.Max(max, tSlice[i]);
                }

                // Calculate vectorized exponentiated values and exponentiated sum
                var vmaxSplat = new Vector<double>(max); // splat scalar into vector
                var acc = Vector<double>.Zero; // sum vector
                for (int i = 0; i < tVecs.Length; i++)
                {
                    rVecs[i] = Vector.Exp(tVecs[i] - vmaxSplat);
                    acc += rVecs[i];
                }
                double sum = Vector.Sum(acc);

                // Clean up unvectorized tail
                for (int i = tVecs.Length * VectorSize; i < classes; i++)
                {
                    rSlice[i] = Math.Exp(tSlice[i] - max);
                    sum += rSlice[i];
                }

                // Calculate vectorized normalized values
                var vsumSplat = new Vector<double>(sum); // splat scalar into vector
                for (int i = 0; i < rVecs.Length; i++)
                {
                    rVecs[i] /= vsumSplat;
                }

                // Clean up unvectorized tail
                for (int i = rVecs.Length * VectorSize; i < classes; i++)
                {
                    rSlice[i] /= sum;
                }
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = softmax(t) -> grad_t_i = r_i * (grad_r_i - sum_j(grad_r_j * r_j)) -> Vector-Jacobian product
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Calculate gradient per batch
                    for (int b = 0; b < batchSize; b++)
                    {
                        int offset = b * classes; // linear offset of current batch

                        // Vectorize inputs, results, and gradients
                        var tgSlice = t.Grad.AsSpan(offset, classes);
                        var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(tgSlice);
                        var rvSlice = result.Data.AsSpan(offset, classes);
                        var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(rvSlice);
                        var rgSlice = result.Grad.AsSpan(offset, classes);
                        var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);

                        // Calculate vectorized dot product of result gradient and forward softmax result -> sum_j(grad_r_j * r_j)
                        var vdot = Vector<double>.Zero; // product/sum vector
                        for (int i = 0; i < rvVecs.Length; i++)
                        {
                            vdot += rgVecs[i] * rvVecs[i];
                        }
                        double dot = Vector.Sum(vdot);

                        // Clean up unvectorized tail
                        for (int i = rvVecs.Length * VectorSize; i < classes; i++)
                        {
                            dot += rgSlice[i] * rvSlice[i];
                        }

                        // Calculate vectorized gradients for parent
                        var vdotSplat = new Vector<double>(dot); // splat scalar into vector
                        for (int i = 0; i < tgVecs.Length; i++)
                        {
                            tgVecs[i] += rvVecs[i] * (rgVecs[i] - vdotSplat);
                        }

                        // Clean up unvectorized tail
                        for (int i = tgVecs.Length * VectorSize; i < classes; i++)
                        {
                            tgSlice[i] += rvSlice[i] * (rgSlice[i] - dot);
                        }
                    }
                };
            }

            return result;
        }

        // Other neural network utility functions

        /// <summary>
        /// Calculates the flat sum of all elements in the input tensor.
        /// </summary>
        /// <param name="t">Tensor to calculate sum of.</param>
        /// <returns>Tensor containing the flat sum of the input tensor.</returns>
        public static Tensor Sum(Tensor t)
        {
            Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var acc = Vector<double>.Zero; // sum vector

            // Calculate vectorized sum
            for (int i = 0; i < tVecs.Length; i++)
            {
                acc += tVecs[i];
            }
            double sum = Vector.Sum(acc);

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < t.ElementCount; i++)
            {
                sum += t[i];
            }

            result[0] = sum;

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = sum(t) -> grad_t_i = grad_r
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vrg = new Vector<double>(result.Grad[0]); // splat scalar into vector

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += vrg;
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < t.ElementCount; i++)
                    {
                        t.Grad[i] += result.Grad[0];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Calculates the flat mean average of all elements in the input tensor.
        /// </summary>
        /// <param name="t">Tensor to calculate mean of.</param>
        /// <returns>Tensor containing the flat mean average of the input tensor.</returns>
        public static Tensor Mean(Tensor t)
        {
            Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var acc = Vector<double>.Zero; // sum vector

            // Calculate vectorized sum
            for (int i = 0; i < tVecs.Length; i++)
            {
                acc += tVecs[i];
            }
            double sum = Vector.Sum(acc);

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < t.ElementCount; i++)
            {
                sum += t[i];
            }

            result[0] = sum / t.ElementCount; // calculate mean

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = mean(t) -> dr/dt = 1 / n
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    double rg = result.Grad[0] / t.ElementCount; // precalculate 1 / n * grad_r
                    var vrg = new Vector<double>(rg); // splat scalar into vector

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += vrg;
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += rg;
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Masks predicted Q-Values based on action selected per experience.
        /// </summary>
        /// <param name="qValues">Q-Values to mask.</param>
        /// <param name="batch">Batch of corresponding experiences.</param>
        /// <returns>Tensor containing the masked predicted Q-Values per experience.</returns>
        public static Tensor MaskActions(Tensor qValues, List<Experience> batch)
        {
            int batchSize = batch.Count;
            int actionCount = qValues.Dimensions[^1];

            Tensor result = GetResultTensor(qValues, [batchSize, 1], qValues.RequiresGrad);

            // Find corresponding Q-Values of experience actions
            for (int i = 0; i < batchSize; i++)
            {
                result[i] = qValues[i * actionCount + batch[i].Action];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(qValues);

                // Gradient calculation function for r = Mask(q) -> dr_i/dq_i = 1
                result._backward = () =>
                {
                    if (!qValues.RequiresGrad) return;

                    for (int i = 0; i < batchSize; i++)
                    {
                        qValues.Grad[i * actionCount + batch[i].Action] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Finds the index of the largest value in the given tensor.
        /// </summary>
        /// <param name="t">Tensor to find maximum of.</param>
        /// <returns>Index of the maximum value in the input tensor.</returns>
        public static int ArgMax(Tensor t)
        {
            int index = 0;
            double max = t[0];

            for (int i = 1; i < t.ElementCount; i++)
            {
                if (t[i] > max)
                {
                    max = t[i];
                    index = i;
                }
            }

            return index;
        }

        // Transpose the Tensor along specific axes
        /// <summary>
        /// Transposes the given tensor based on the given axes permutation order.
        /// </summary>
        /// <param name="t">Tensor to transpose.</param>
        /// <param name="axes">Axes permutation order to transpose by.</param>
        /// <returns>Transpose of the input tensor based on the given axes permutation order.</returns>
        public static Tensor Transpose(Tensor t, int[]? axes = null)
        {
            // Default permutation order: reverse all axes
            if (axes is null)
            {
                axes = new int[t.Rank];
                for (int i = 0; i < axes.Length; i++)
                {
                    axes[i] = axes.Length - 1 - i;
                }
            }

            // Build result dimensions
            var resultDims = new int[t.Rank];
            for (int i = 0; i < axes.Length; i++)
            {
                resultDims[i] = t.Dimensions[axes[i]];
            }

            Tensor result = GetResultTensor(t, resultDims, t.RequiresGrad);

            // Preallocate index mappings on stack - avoid garbage collector
            Span<int> srcIndices = stackalloc int[t.Rank];
            Span<int> dstIndices = stackalloc int[axes.Length];

            // Remap each element based on permutation order
            for (int i = 0; i < t.ElementCount; i++)
            {
                t.GetFullIndices(i, srcIndices);
                for (int j = 0; j < axes.Length; j++)
                {
                    dstIndices[j] = srcIndices[axes[j]];
                }
                result[result.LinearIndex(dstIndices)] = t[i];
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Inverse permutation order
                var invAxes = new int[axes.Length];
                for (int i = 0; i < axes.Length; i++)
                {
                    invAxes[axes[i]] = i;
                }

                // Gradient calculation function for r = Transpose(t) -> grad_t = Transpose(grad_r)
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Preallocate index mappings on stack - avoid garbage collector
                    Span<int> srcIndices = stackalloc int[result.Rank];
                    Span<int> dstIndices = stackalloc int[invAxes.Length];

                    // Remap each gradient based on inverse permutation order
                    for (int i = 0; i < result.ElementCount; i++)
                    {
                        result.GetFullIndices(i, srcIndices);
                        for (int j = 0; j < invAxes.Length; j++)
                        {
                            dstIndices[j] = srcIndices[invAxes[j]];
                        }
                        t.Grad[t.LinearIndex(dstIndices)] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Broadcasts a tensor into the given new dimensions.
        /// </summary>
        /// <param name="t">Tensor to broadcast.</param>
        /// <param name="targetDims">Dimensions to broadcast into.</param>
        /// <returns>Tensor with target dimensions and data from the input tensor.</returns>
        public static Tensor Broadcast(Tensor t, int[] targetDims)
        {
            // T.Dimensions must be a suffix of targetDims (t -> [16], targetDims = [32, 16])

            Tensor result = GetResultTensor(t, targetDims, t.RequiresGrad);

            // Shortcut for 1D input
            if (t.Rank == 1)
            {
                int stride = t.ElementCount;
                int rows = result.ElementCount / stride;
                for (int r = 0; r < rows; r++)
                {
                    t.Data.AsSpan(0, stride).CopyTo(result.Data.AsSpan(r * stride, stride));
                }
            }
            else
            {
                // Preallocate index mappings on stack - avoid garbage collector
                Span<int> indices = stackalloc int[result.Rank];
                Span<int> srcIndices = stackalloc int[t.Rank];

                // Copy input tensor data into result tensor
                for (int i = 0; i < result.ElementCount; i++)
                {
                    result.GetFullIndices(i);
                    indices[^t.Rank..].CopyTo(srcIndices);
                    result[i] = t[t.LinearIndex(srcIndices)];
                }
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = Broadcast(t) -> grad_t = sum(broadcasted axes) [grad_r]
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Shortcut for 1D input
                    if (t.Rank == 1)
                    {
                        int stride = t.ElementCount;
                        int rows = result.ElementCount / stride;

                        // Vectorize gradients
                        var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());

                        // Calculate gradients for parent per broadcasted dimension
                        for (int r = 0; r < rows; r++)
                        {
                            // Vectorize gradients
                            var rgSlice = result.Grad.AsSpan(r * stride, stride);
                            var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);

                            // Calculate vectorized gradients for parent
                            for (int i = 0; i < tgVecs.Length; i++)
                            {
                                tgVecs[i] += rgVecs[i];
                            }

                            // Clean up unvectorized tail
                            for (int i = tgVecs.Length * VectorSize; i < stride; i++)
                            {
                                t.Grad[i] += rgSlice[i];
                            }
                        }
                    }
                    else
                    {
                        // Preallocate index mappings on stack - avoid garbage collector
                        Span<int> indices = stackalloc int[result.Rank];
                        Span<int> srcIndices = stackalloc int[t.Rank];

                        // Calculate gradients for parent sequentially
                        for (int i = 0; i < result.ElementCount; i++)
                        {
                            result.GetFullIndices(i, indices);
                            indices[^t.Rank..].CopyTo(srcIndices);
                            t.Grad[t.LinearIndex(srcIndices)] += result.Grad[i];
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Flattens a tensor beyond the given axis.
        /// </summary>
        /// <param name="t">Tensor to flatten.</param>
        /// <param name="startAxis">First axis to flatten.</param>
        /// <returns>Result tensor of flattening the input tensor beyond the given axis.</returns>
        public static Tensor Flatten(Tensor t, int startAxis = 0)
        {
            // Calculate size of flat dimension
            int flatSize = 1;
            for (int i = startAxis; i < t.Rank; i++) flatSize *= t.Dimensions[i];

            // Build new dimensions
            var newDims = new int[startAxis + 1];
            for (int i = 0; i < startAxis; i++) newDims[i] = t.Dimensions[i];
            newDims[^1] = flatSize;

            return Reshape(t, newDims); // use general reshape function
        }

        /// <summary>
        /// Reshapes a tensor into the given dimensions without rearranging linear data.
        /// </summary>
        /// <param name="t">Tensor to reshape.</param>
        /// <param name="newDims">Dimensions to reshape to.</param>
        /// <returns>Result tensor of reshaping the input tensor into the given dimensions.</returns>
        public static Tensor Reshape(Tensor t, int[] newDims)
        {
            Tensor result = GetResultTensor(t, newDims, t.RequiresGrad);

            Array.Copy(t.Data, result.Data, t.ElementCount); // copy linear data

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function for r = Reshape(t) -> dr_i/dt_i = 1
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Wraps a tensor into a new batch with 1 experience entry.
        /// </summary>
        /// <param name="t">Tensor to wrap into batch.</param>
        /// <returns>Batch tensor containing the input tensor.</returns>
        public static Tensor WrapBatch(Tensor t)
        {
            // Add batch dimension
            var batchDims = new int[t.Rank + 1];
            batchDims[0] = 1;
            Array.Copy(t.Dimensions, 0, batchDims, 1, t.Rank);

            Tensor batch = GetResultTensor(t, batchDims, t.RequiresGrad);

            Array.Copy(t.Data, batch.Data, t.ElementCount); // copy linear data

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                batch._parents.Add(t);

                // Gradient calculation function for b = WrapBatch(t) -> db_i/dt_i = 1
                batch._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(batch.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += bgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += batch.Grad[i];
                    }
                };
            }

            return batch;
        }

        // Clip values
        /// <summary>
        /// Clips a tensor's values to be within a specified range.
        /// </summary>
        /// <param name="t">Tensor to clip.</param>
        /// <param name="min">Minimum value to clip to.</param>
        /// <param name="max">Maximum value to clip to.</param>
        /// <returns>Result tensor containing the input tensor's values clipped to between the given min and max.</returns>
        public static Tensor Clip(Tensor t, double min, double max)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            // Vectorize inputs and results
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vmin = new Vector<double>(min); // splat scalar into vector
            var vmax = new Vector<double>(max); // splat scalar into vector
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized clamp result
            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Min(Vector.Max(tVecs[i], vmin), vmax);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Clamp(t[i], min, max);
            }

            // Connect result tensor to autograd graph if needed
            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function r = Clip(t) -> dr/dt = 1 (min < t < max), 0 (t <= min or t >= max)
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Vectorize inputs and gradients
                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vmin = new Vector<double>(min); // splat scalar into vector
                    var vmax = new Vector<double>(max); // splat scalar into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients for parent
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        var inRange = Vector.BitwiseAnd(
                            Vector.GreaterThanOrEqual(tvVecs[i], vmin),
                            Vector.LessThanOrEqual(tvVecs[i], vmax));
                        tgVecs[i] += Vector.ConditionalSelect(inRange, rgVecs[i], Vector<double>.Zero);
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += (t[i] >= min && t[i] <= max) ? result.Grad[i] : 0;
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// Gets the tensor instance to write the result of the current function/operation to.
        /// </summary>
        /// <param name="owner">Tensor being used as an input in the function/operation.</param>
        /// <param name="dims">Dimensions of the result tensor.</param>
        /// <param name="requiresGrad">Whether it is necessary to calculate the result tensor's gradient.</param>
        /// <returns>Tensor instance into which to write the result of the function/operation.</returns>
        static Tensor GetResultTensor(Tensor owner, int[] dims, bool requiresGrad)
        {
            // Prepare input tensor for forward pass of the operation
            owner.PrepareForward();

            // Whether the operation should already be represented in the autograd graph
            bool newOp = owner._results.Count <= owner._opIndex;

            Tensor result;
            if (newOp) // create new tensor instance to write result into
            {
                result = new(dims, requiresGrad);
                owner._results.Add(result);
            }
            else
            {
                // Find tensor instance used to store result of operation with the same instance during last forward pass
                result = owner._results[owner._opIndex];

                // Check whether the stored instance has the required dimensions
                bool shapeMismatch = result.Rank != dims.Length;
                if (!shapeMismatch)
                {
                    for (int i = 0; i < dims.Length; i++)
                    {
                        if (result.Dimensions[i] != dims[i])
                        {
                            shapeMismatch = true;
                            break;
                        }
                    }
                }

                // Create new tensor instance if stored instance does not match required dimensions
                if (shapeMismatch)
                {
                    result = new(dims, requiresGrad);
                    owner._results[owner._opIndex] = result;
                }
                else
                {
                    // Clear out previous autograd graph data from stored instance
                    result._parents.Clear();
                    result._backward = delegate { };
                }
            }

            owner._opIndex++;

            return result;
        }
    }

    /// <summary>
    /// Static class containing various math utility functions.
    /// </summary>
    public static class MathUtils
    {
        /// <summary>
        /// Current Random instance.
        /// </summary>
        static readonly Random Random = new();

        /// <summary>
        /// Generates a random number from a probability distribution with a mean of 0 and standard deviation of 1.
        /// </summary>
        /// <returns>Random number generated from a probability distribution with a mean of 0 and standard deviation of 1.</returns>
        public static double NextGaussian()
        {
            // Generate 2 random values from uniform distribution
            double u1 = 1.0 - Random.NextDouble();
            double u2 = 1.0 - Random.NextDouble();

            // Apply Box-Muller Transform -> Z_1 = sqrt(-2 * ln(u_1)) * sin(2π * u_2)
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return randStdNormal;
        }

        /// <summary>
        /// Generates a random number from a probability distribution with the given mean and standard deviation.
        /// </summary>
        /// <param name="mean">Mean of the probability distribution to generate from.</param>
        /// <param name="stdDev">Standard deviation of the probability distribution to generate from.</param>
        /// <returns>Random number generated from a probability distribution with the given mean and standard deviation.</returns>
        public static double NextGaussian(double mean, double stdDev)
        {
            double randStdNormal = NextGaussian(); // generate random number from standard distribution

            // Convert to target distribution -> X = (Z * σ) + μ
            double randNormal = randStdNormal * stdDev + mean;
            return randNormal;
        }

        /// <summary>
        /// Rounds a value to the given interval digit position.
        /// </summary>
        /// <param name="value">Value to round.</param>
        /// <param name="interval">Interval to round to.</param>
        /// <returns>Input value rounded to the given interval.</returns>
        public static int RoundToInterval(double value, int interval)
        {
            return (int)Math.Round(value / interval, MidpointRounding.AwayFromZero) * interval;
        }

        /// <summary>
        /// Samples the sigmoid function at the given input value.
        /// </summary>
        /// <param name="value">Input value to sample at.</param>
        /// <returns>Value of the sigmoid function at the given input value.</returns>
        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        /// <summary>
        /// Samples the hyperbolic tangent function at the given input value.
        /// </summary>
        /// <param name="value">Input value to sample at.</param>
        /// <returns>Value of the hyperbolic tangent function at the given input value.</returns>
        public static double Tanh(double value)
        {
            return Math.Tanh(value);
        }

        /// <summary>
        /// Rounds a TimeSpan to the nearest whole millisecond.
        /// </summary>
        /// <param name="input">TimeSpan to round.</param>
        /// <returns>Given TimeSpan rounded to the nearest whole millisecond.</returns>
        public static TimeSpan RoundToMS(TimeSpan input)
        {
            return TimeSpan.FromMilliseconds(Math.Round(input.TotalMilliseconds));
        }
    }

    /// <summary>
    /// Static class for handling saving and loading of trained neural networks.
    /// </summary>
    public static class Saver
    {
        /// <summary>
        /// Directory to save to and load from.
        /// </summary>
        const string DirectoryName = "Models";
        /// <summary>
        /// Full directory path to save to and load from.
        /// </summary>
        static string DirectoryPath = string.Empty;
        /// <summary>
        /// File extension for neural network save files.
        /// </summary>
        const string Extension = ".nnn";

        #pragma warning disable CS8604 // Possible null reference argument.
        /// <summary>
        /// Saves the given model to a file with the given name.
        /// </summary>
        /// <param name="model">Model to save.</param>
        /// <param name="fileName">Name of file to save to.</param>
        public static void SaveModel(Model model, string fileName)
        {
            InitDirectory(); // ensure directory exists

            string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

            // Generate save data for the model
            var layers = new LayerData[model.Layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                var layer = model.Layers[i];

                switch (layer)
                {
                    case Dense dense:
                        layers[i] = new(layerName: dense.GetType().AssemblyQualifiedName,
                            activation: dense.Activation.GetType().AssemblyQualifiedName, biases: dense.Biases,
                            neuronCount: dense.NeuronCount, weights: dense.Weights);

                        break;
                    case Conv conv:
                        layers[i] = new(layerName: conv.GetType().AssemblyQualifiedName,
                            activation: conv.Activation.GetType().AssemblyQualifiedName, biases: conv.Biases,
                            filterCount: conv.FilterCount, kernelDims: conv.KernelDims, kernels: conv.Kernels);

                        break;
                }
            }
            ModelData modelData = new(layers);

            // Serialize the model as Json and save to file
            string json = JsonSerializer.Serialize(modelData);

            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads the model stored in the given file.
        /// </summary>
        /// <param name="fileName">Name of file to load from.</param>
        /// <returns>Model loaded from the given file.</returns>
        public static Model LoadModel(string fileName)
        {
            InitDirectory(); // ensure directory exists

            string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

            string json = File.ReadAllText(filePath); // load model Json

            // Reconstruct model from save data Json
            var modelData = JsonSerializer.Deserialize<ModelData>(json);
            Model model = new(modelData);

            return model;
        }
        #pragma warning restore CS8604 // Possible null reference argument.

        /// <summary>
        /// Checks whether a neural network file with the given name exists.
        /// </summary>
        /// <param name="fileName">Name of file to search for.</param>
        /// <returns>Whether a neural network file with the given name exists.</returns>
        public static bool FileExists(string fileName)
        {
            InitDirectory(); // ensure directory exists

            string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

            if (File.Exists(filePath)) return true;
            else return false;
        }

        /// <summary>
        /// Initializes the directory and directory path if not yet initialized.
        /// </summary>
        static void InitDirectory()
        {
            if (string.IsNullOrEmpty(DirectoryPath))
            {
                string? exePath = System.Environment.ProcessPath;
                string? exeDirPath = Path.GetDirectoryName(exePath);
                if (!string.IsNullOrEmpty(exeDirPath))
                {
                    DirectoryPath = Path.Combine(exeDirPath, DirectoryName);
                    if (!Directory.Exists(DirectoryPath)) Directory.CreateDirectory(DirectoryPath);
                }
            }
        }

        /// <summary>
        /// Class for storing save data for a trained model.
        /// </summary>
        /// <param name="layers">Array of save data for layers constituting the model.</param>
        [Serializable]
        public class ModelData(LayerData[] layers)
        {
            /// <summary>
            /// Array of save data for layers constituting the model.
            /// </summary>
            public LayerData[] Layers { get; set; } = layers;
        }

        /// <summary>
        /// Class for storing save data for a single layer of a trained model.
        /// </summary>
        /// <param name="neuronCount">Number of neurons in the layer.</param>
        /// <param name="layerName">Name of the specific layer subclass.</param>
        /// <param name="weights">Weights parameter of the layer.</param>
        /// <param name="biases">Bias parameter of the layer.</param>
        /// <param name="activation">Name of the activation function of the layer.</param>
        [Serializable]
        public class LayerData(string layerName, string activation, Tensor biases, int? neuronCount = null, Tensor? weights = null,
            int? filterCount = null, int[]? kernelDims = null, Tensor? kernels = null)
        {
            /// <summary>
            /// Name of the specific layer subclass.
            /// </summary>
            public string LayerName { get; set; } = layerName;
            /// <summary>
            /// Name of the activation function of the layer.
            /// </summary>
            public string Activation { get; set; } = activation;
            /// <summary>
            /// Bias parameter of the layer.
            /// </summary>
            public Tensor Biases { get; set; } = biases;
            /// <summary>
            /// Number of neurons in the layer.
            /// </summary>
            public int? NeuronCount { get; set; } = neuronCount;
            /// <summary>
            /// Weights parameter of the layer.
            /// </summary>
            public Tensor? Weights { get; set; } = weights;
            /// <summary>
            /// Number of filters used by the layer.
            /// </summary>
            public int? FilterCount { get; set; } = filterCount;
            /// <summary>
            /// Dimensions of the layer's kernels.
            /// </summary>
            public int[]? KernelDims { get; set; } = kernelDims;
            /// <summary>
            /// Kernels parameter of the layer.
            /// </summary>
            public Tensor? Kernels { get; set; } = kernels;
        }
    }

    /// <summary>
    /// Static class containing various UI utility functions.
    /// </summary>
    public static class UIUtils
    {
        /// <summary>
        /// Standard user input mappings.
        /// </summary>
        public static readonly Dictionary<UserInput, string> userInputs = new() { { UserInput.Yes, "y" },
            { UserInput.No, "n" }, { UserInput.Quit, "q" } };

        /// <summary>
        /// Standard user inputs.
        /// </summary>
        public enum UserInput { Yes, No, Quit }

        /// <summary>
        /// Saves the given model to the file entered by the user.
        /// </summary>
        /// <param name="model">Model to be saved.</param>
        public static void SaveLoop(Model model)
        {
            string fileName;

            while (true)
            {
                fileName = GetInput("Enter file name");
                if (Saver.FileExists(fileName))
                {
                    if (GetInput($"File with name \"{fileName}\" already exists. Overwrite existing file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]])
                        == userInputs[UserInput.Yes])
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

        /// <summary>
        /// Prompts the user to enter a file name.
        /// </summary>
        /// <returns>Name of an existing file entered by the user.</returns>
        public static string GetFileName()
        {
            string input;
            while (true)
            {
                input = GetInput("Enter file name");
                if (Saver.FileExists(input)) return input;
                else Console.WriteLine("\nFile not found");
            }
        }

        /// <summary>
        /// Prompts the user to enter an integer input.
        /// </summary>
        /// <param name="prompt">Prompt to display to the user.</param>
        /// <returns>Valid integer entered by the user.</returns>
        public static int GetInteger(string prompt)
        {
            while (true)
            {
                if (int.TryParse(GetInput(prompt), out int integer)) return integer;
                else Console.WriteLine("\nNot a valid number");
            }
        }

        /// <summary>
        /// Gets a valid episode index from the user.
        /// </summary>
        /// <param name="episodeBuffer">Episode buffer which is being accessed.</param>
        /// <returns>Valid episode index entered by the user.</returns>
        public static int GetEpisodeSelection(FIFOBuffer<Episode> episodeBuffer)
        {
            while (true)
            {
                int index = GetInteger($"Enter episode number ({episodeBuffer.Count} episodes cached)");
                if (index > 0 && index <= episodeBuffer.Count) return index - 1;
                else Console.WriteLine("Invalid episode number");
            }
        }

        /// <summary>
        /// Prompts the user for an input.
        /// </summary>
        /// <param name="prompt">Prompt to display to the user.</param>
        /// <param name="options">Optional list of valid user inputs.</param>
        /// <returns>Valid input entered by the user.</returns>
        public static string GetInput(string prompt, List<string>? options = null)
        {
            options ??= [];
            for (int i = 0; i < options.Count; i++)
            {
                options[i] = options[i].ToLowerInvariant();
            }

            // Prompt the user until a valid input is given or the quit input is given
            string input;
            while (true)
            {
                Console.WriteLine($"\n{prompt}");
                input = Console.ReadLine()?.ToLowerInvariant() ?? "";

                if (input == userInputs[UserInput.Quit]) System.Environment.Exit(0); // terminate the program if the user chooses to quit
                else if (options.Count == 0 || options.Contains(input)) return input; // return input if valid or no valid inputs given
            }
        }
    }

    /// <summary>
    /// Static class containing demonstration functionality.
    /// </summary>
    public static class DemoHandler
    {
        /// <summary>
        /// Array of all environments with trained and implemented demonstrations.
        /// </summary>
        static readonly Environment[] DemoEnvs = [new TicTacToe(), new Snake()];

        /// <summary>
        /// Runs the demonstration user interaction loop.
        /// </summary>
        public static void RunDemo()
        {
            Console.WriteLine("Welcome to the Neural Network Nonsense library demonstration.");
            Console.WriteLine("Enter Q at any time to close the demonstration.");

            // Main interaction loop
            bool done = false;
            while (!done)
            {
                // Get demo index from the user
                int envIndex = -1;
                bool validIndex = false;
                while (!validIndex)
                {
                    string prompt = "Please select which environment you would like to see a demo of:";
                    for (int i = 0; i < DemoEnvs.Length; i++)
                    {
                        prompt += $"\n{i + 1} - {DemoEnvs[i].GetType().Name}";
                    }

                    envIndex = GetInteger(prompt);

                    if (envIndex > 0 && envIndex <= DemoEnvs.Length) validIndex = true;
                    else Console.WriteLine("Invalid selection...");
                }

                Console.WriteLine("\n");
                DemoEnvs[envIndex - 1].PlayDemo();

                Console.Clear();
                if (GetInput("Continue viewing demos? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) done = true;
            }

            Console.WriteLine("Press any key to close...");
            Console.ReadKey();
        }
    }
}