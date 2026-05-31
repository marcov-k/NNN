using NNN;

Model model;
NNN.Environment env = new TicTacToe();
double exploration = 1.0;
double explorationDecay = 0.999;
double minExploration = 0.1;
double discount = 0.99;
Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 256;
int agentBufferSize = 10;
int opponentCopyRate = 100;
int minRandomOpponentEpisodes = 100;
double tau = 0.005;
double maxGradNorm = 1.0;
int minExperiences = 512;
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
        exploration = minExploration;
    }
    else
    {
        model = new([
            new Dense(64, new LeakyReLU()),
            new Dense(64, new LeakyReLU()),
            new Dense(env.ActionCount, new Linear())
        ], env.StateFormat);
    }

    dqnTrainer = new(
        agent: model,
        environment: env,
        exploration: exploration,
        explorationDecay: explorationDecay,
        minExploration: minExploration,
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
    Tensor trueState;
    bool done = false;
    double totalReward = 0;
    int steps = 0;

    List<Experience> episodeExperiences = [];
    while (!done && steps < 50)
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
    using System.Linq;
    using System.Text.Json;
    using System.Numerics;
    using System.Runtime.InteropServices;
    using System.Buffers;

    public abstract class Environment
    {
        public int StateSize => StateFormat.ElementCount;
        public virtual Tensor StateFormat => throw new NotImplementedException();
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

        public virtual int PickAgentAction(Tensor qValues, Tensor? state = null)
        {
            throw new NotImplementedException();
        }

        public virtual int PickRandomAction()
        {
            throw new NotImplementedException();
        }
    }

    public class MovementGrid2D : Environment
    {
        public override Tensor StateFormat => new([1, 4]);
        public override int ActionCount => 4;
        readonly Random random = new();
        readonly Tensor State = new([4]); // current x, current y, target x, target y
        readonly int[] Bounds; // xMin, xMax, yMin, yMax
        readonly int MaxSteps;
        readonly double XRange;
        readonly double YRange;

        public MovementGrid2D(int xMin, int xMax, int yMin, int yMax, int maxSteps = 50)
        {
            State[0] = random.Next(xMin, xMax + 1);
            State[1] = random.Next(yMin, yMax + 1);
            State[2] = random.Next(xMin, xMax + 1);
            State[3] = random.Next(yMin, yMax + 1);
            Bounds = [xMin, xMax, yMin, yMax];
            XRange = xMax - xMin;
            YRange = yMax - yMin;
            MaxSteps = maxSteps;
        }

        public override Tensor GetNormalizedState()
        {
            Tensor normalized = new([4]);

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

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            double xDiff = State[2] - State[0];
            double yDiff = State[3] - State[1];
            double prevDist = Math.Sqrt(Math.Pow(xDiff, 2.0) + Math.Pow(yDiff, 2.0));

            switch (action)
            {
                case (int)Action.Left: // left
                    State[0]--;
                    break;
                case (int)Action.Right: // right
                    State[0]++;
                    break;
                case (int)Action.Up: // up
                    State[1]++;
                    break;
                case (int)Action.Down: // down
                    State[1]--;
                    break;
            }

            xDiff = State[2] - State[0];
            yDiff = State[3] - State[1];
            double newDist = Math.Sqrt(Math.Pow(xDiff, 2.0) + Math.Pow(yDiff, 2.0));
            double deltaDist = prevDist - newDist;

            bool reachedTarget = (State[0] == State[2] && State[1] == State[3]);
            bool outOfBounds = (State[0] < Bounds[0]) || (State[0] > Bounds[1]) ||
                               (State[1] < Bounds[2]) || (State[1] > Bounds[3]);
            bool outOfSteps = steps >= MaxSteps && !reachedTarget;

            bool done = reachedTarget || outOfBounds || outOfSteps;

            double reward = 5.0 * deltaDist;
            reward += reachedTarget ? 100.0 : 0.0;
            reward -= outOfBounds ? 100.0 : 0.0;
            reward -= outOfSteps ? 5.0 : 0.0;

            return (reward, GetNormalizedState(), done);
        }

        public override void Reset()
        {
            State[0] = random.Next(Bounds[0], Bounds[1] + 1);
            State[1] = random.Next(Bounds[2], Bounds[3] + 1);
            State[2] = random.Next(Bounds[0], Bounds[1] + 1);
            State[3] = random.Next(Bounds[2], Bounds[3] + 1);
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

                    if (x == state[0] && y == state[1])
                    {
                        Console.Write("A");
                    }
                    else if (x == state[2] && y == state[3])
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

        public override int PickAgentAction(Tensor qValues, Tensor? state = null) => Tensor.ArgMax(qValues);

        public override int PickRandomAction() => random.Next(ActionCount);

        enum Action { Left, Right, Up, Down }
    }

    public class TicTacToe : Environment, ISelfPlay
    {
        public override Tensor StateFormat => new([1, 10]);
        public override int ActionCount => 9;
        public bool AgentTurn { get; set; } = true;
        public int OpponentCount { get; set; }
        public int OpponentIndex { get; set; }
        public Tensor State { get; init; } = new([10]);
        readonly int MaxSteps = 9;
        public Random Random { get; init; } = new();
        static readonly int[][] WinOrients = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
        const double WinRewardBase = 1.0;
        const double BlockRewardBase = 0.1;
        const double DrawRewardBase = 0.2;
        const double Penalty = 0.0;

        public TicTacToe() { }

        public override Tensor GetNormalizedState()
        {
            return GetState();
        }

        public override Tensor GetState()
        {
            return State.Copy();
        }

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            if (!ValidAction(action)) throw new ArgumentException("Invalid Action");

            State[action] = State[9] == 1.0 ? 1.0 : -1.0;
            var (reward, done) = EvaluateAction(action);
            AgentTurn = !AgentTurn;
            State[9] *= -1.0;

            var nextState = GetNormalizedState();

            done = done || BoardFilled() || steps >= MaxSteps;

            return (reward, nextState, done);
        }

        public override void Reset()
        {
            AgentTurn = Random.Next(2) == 1;
            OpponentIndex = Random.Next(OpponentCount + 1);

            for (int i = 0; i < State.ElementCount - 1; i++)
            {
                State[i] = 0.0;
            }
            State[9] = 1.0;
        }

        public override void Render(Episode episode, int step)
        {
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
            var state = step == episode.Experiences.Count ? exp.NextState : exp.State;
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);

            DrawState(state);

            Console.WriteLine($"\n\nPosition Taken: {action}, Reward: {reward}");
        }

        bool ValidAction(int action, Tensor? state = null)
        {
            state ??= State;
            return (action != state.ElementCount - 1) && (state[action] == 0.0);
        }

        (double reward, bool won) EvaluateAction(int action)
        {
            var relevantOrients = WinOrients.Where(o => o.Contains(action)).ToArray();

            double[][] orientValues = new double[relevantOrients.Length][];
            for (int orient = 0; orient < relevantOrients.Length; orient++)
            {
                orientValues[orient] = new double[relevantOrients[orient].Length];
                for (int pos = 0; pos < relevantOrients[orient].Length; pos++)
                {
                    orientValues[orient][pos] = State[relevantOrients[orient][pos]];
                }
            }

            double ownValue = State[9] == 1.0 ? 1.0 : -1.0;
            double oppValue = -ownValue;

            var advantOrients = orientValues.Where(o => !o.Contains(oppValue));
            var blockOrients = orientValues.Where(o => o.Contains(oppValue) && !o.Contains(ownValue));
            var falseOrients = orientValues.Where(o => o.Contains(ownValue) && o.Contains(oppValue));

            bool boardFilled = BoardFilled();
            double reward = boardFilled ? 0.0 : Penalty * falseOrients.Count();
            bool won = false;

            foreach (var orient in advantOrients)
            {
                int ownPositions = orient.Count(p => p == ownValue);
                reward += ownPositions switch
                {
                    2 => 0.05 * WinRewardBase,
                    3 => WinRewardBase,
                    _ => 0.0
                };
                won = won || ownPositions == 3;
            }

            foreach (var orient in blockOrients)
            {
                int oppPositions = orient.Count(p => p == oppValue);
                reward += oppPositions == 2 ? BlockRewardBase : 0.0;
            }

            reward += (boardFilled && !won) ? DrawRewardBase : 0.0;

            return (reward, won);
        }

        bool BoardFilled(Tensor? state = null)
        {
            state ??= State;
            return !state.Data.Any(p => p == 0.0);
        }

        public void Play(Model agent)
        {
            Reset();
            bool playerTurn = Random.Next(2) == 0;
            string winner = "Draw";
            bool done = false;
            while (!done)
            {
                Console.Clear();
                DrawState(State);

                int action = playerTurn ? GetPlayerAction() : GetAgentAction(agent);
                if (action == -1) break;

                State[action] = State[9] == 1.0 ? 1.0 : -1.0;

                if (CheckWin())
                {
                    winner = State[9] == 1.0 ? "X" : "O";
                    break;
                }

                done = BoardFilled();

                State[9] *= -1.0;
                playerTurn = !playerTurn;
            }

            Console.Clear();
            DrawState(State);
            Console.WriteLine($"\n\nWinner: {winner}");
        }

#pragma warning disable CS8602 // Dereference of a possibly null reference.
        int GetPlayerAction()
        {
            string input = string.Empty;
            Console.WriteLine();
            while (input != "q")
            {
                Console.WriteLine("\nEnter desired position:");
                input = Console.ReadLine().ToLowerInvariant();

                if (int.TryParse(input, out int action) && ValidAction(action)) return action;
                else if (input == "q") break;

                Console.WriteLine("Invalid position...");
            }
            return -1;
        }
#pragma warning restore CS8602 // Dereference of a possibly null reference.

        public int GetAgentAction(Model agent, Tensor? state = null)
        {
            state ??= State;
            return PickAgentAction(agent.Predict(Tensor.WrapBatch(state)), state);
        }

        bool CheckWin(Tensor? state = null)
        {
            state ??= State;
            return WinOrients.Any(o => o.All(p => state[p] == (state[9] == 1.0 ? 1.0 : -1.0)));
        }

        static void DrawState(Tensor state)
        {
            for (int i = 0; i < state.ElementCount - 1; i++)
            {
                if (i % 3 == 0) Console.WriteLine();

                string fill = state[i] switch
                {
                    1.0 => " X ",
                    -1.0 => " O ",
                    _ => "   "
                };
                Console.Write(fill);
            }
        }

        public override int PickAgentAction(Tensor agentQValues, Tensor? state = null)
        {
            state ??= State;
            var qValues = agentQValues.Copy();
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
            List<int> validActions = [];
            for (int i = 0; i < State.ElementCount - 1; i++)
            {
                if (State[i] == 0.0) validActions.Add(i);
            }
            return validActions[Random.Next(validActions.Count)];
        }
    }

    public class Snake : Environment
    {
        public override Tensor StateFormat => new([1, 7]); // x apple dist, y apple dist, dir, obst dist f, obst dist l, obst dist r, reachable pos
        Int2 GridDims;
        public override int ActionCount => 3;
        readonly Random Random = new();
        SnakeNode SnakeHead = new();
        Int2 ApplePosition = new();
        int StepsWithoutApple = 0;
        const int MaxStepsWithoutApple = 50;
        const double AppleReward = 2.0;
        const double DistRewardMult = 0.2;
        const double TimeoutPenalty = -1.0;
        const double CollisionPenalty = -1.0;
        const double StepPenalty = -0.01;
        const int FrameTime = 100;
        const int MaxSteps = 10000;

        public Snake(int width = 20, int height = 20)
        {
            GridDims = new(width, height);
            Reset();
        }

        public override Tensor GetNormalizedState()
        {
            var state = GetState();

            double maxDim = Math.Max(GridDims.X, GridDims.Y);

            state[0] /= GridDims.X;
            state[1] /= GridDims.Y;
            state[3] /= maxDim;
            state[4] /= maxDim;
            state[5] /= maxDim;
            state[6] /= GridDims.X * GridDims.Y;

            return state;
        }

        public override Tensor GetState()
        {
            Tensor state = new([StateFormat.Dimensions[1]]);

            state[0] = ApplePosition.X - SnakeHead.Position.X;
            state[1] = ApplePosition.Y - SnakeHead.Position.Y;

            state[2] = SnakeHead.Direction;

            state[3] = NearestObstacle(SnakeHead.Direction);
            state[4] = NearestObstacle((SnakeHead.Direction + 3) % 4);
            state[5] = NearestObstacle((SnakeHead.Direction + 1) % 4);
            state[6] = ReachablePositions(SnakeHead.Position, BlockedCells());

            return state;
        }

        int NearestObstacle(int dir)
        {
            List<Int2> filledPositions = [];
            SnakeNode? node = SnakeHead;
            while (node is not null)
            {
                filledPositions.Add(node.Position);
                node = node.Child;
            }

            int steps = 0;
            bool xDir = dir % 2 == 0;
            bool posDir = dir >= 2;
            int pos = xDir ? SnakeHead.Position.X : SnakeHead.Position.Y;
            bool hit = false;
            while (!hit)
            {
                steps++;
                pos += posDir ? 1 : -1;
                hit = pos < 0 || pos >= (xDir ? GridDims.X : GridDims.Y) ||
                    filledPositions.Any(p => xDir ? (p.X == pos && p.Y == SnakeHead.Position.Y) : (p.X == SnakeHead.Position.X && p.Y == pos));
            }

            return steps;
        }

        int ReachablePositions(Int2 from, HashSet<Int2> blocked)
        {
            HashSet<Int2> visited = [];
            Queue<Int2> queue = [];

            queue.Enqueue(from);
            while (queue.Count > 0)
            {
                var pos = queue.Dequeue();
                if (!visited.Add(pos)) continue;
                foreach (var neighbor in GetNeighbors(pos))
                {
                    if (!blocked.Contains(neighbor)) queue.Enqueue(neighbor);
                }
            }

            return visited.Count;
        }

        HashSet<Int2> BlockedCells()
        {
            HashSet<Int2> blocked = [];

            SnakeNode? node = SnakeHead.Child;
            while (node is not null)
            {
                if (node.Child is not null) blocked.Add(node.Position);
                else break;

                node = node.Child;
            }

            return blocked;
        }

        List<Int2> GetNeighbors(Int2 pos)
        {
            List<Int2> neighbors = [ new(pos.X - 1, pos.Y), new(pos.X, pos.Y - 1), new(pos.X + 1, pos.Y), new(pos.X, pos.Y + 1) ];

            foreach (var neighbor in neighbors.ToList())
            {
                if (!ValidPosition(neighbor)) neighbors.Remove(neighbor);
            }

            return neighbors;
        }

        bool ValidPosition(Int2 pos) => pos.X >= 0 && pos.X < GridDims.X && pos.Y >= 0 && pos.Y < GridDims.Y;

        public override void Reset()
        {
            StepsWithoutApple = 0;
            int startX = Random.Next(0, GridDims.X);
            int startY = Random.Next(0, GridDims.Y);
            int startDir = Random.Next(0, 4);

            SnakeHead = new(x: startX, y: startY) { Direction = startDir };

            GenerateApple();
        }

        void GenerateApple()
        {
            var state = GetBoardState();

            List<int> validPositions = [];
            for (int i = 0; i < state.ElementCount; i++)
            {
                if (state[i] == BoardEncoding.Empty) validPositions.Add(i);
            }

            int linearPos = validPositions[Random.Next(validPositions.Count)];
            var arrayPos = state.GetFullIndices(linearPos);

            ApplePosition = new(arrayPos[1], arrayPos[0]);
        }

        Tensor GetBoardState()
        {
            Tensor state = new([GridDims.Y, GridDims.X]);

            SnakeNode node = SnakeHead;
            state[[node.Position.Y, node.Position.X]] = BoardEncoding.Head;
            while (node.Child is not null)
            {
                node = node.Child;
                state[[node.Position.Y, node.Position.X]] = (node.Child is not null) ? BoardEncoding.Body : BoardEncoding.Tail;
            }
            state[[ApplePosition.Y, ApplePosition.X]] = BoardEncoding.Apple;

            return state;
        }

        public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
        {
            if (steps >= MaxSteps) return (0.0, GetNormalizedState(), true);

            double xDiff = ApplePosition.X - SnakeHead.Position.X;
            double yDiff = ApplePosition.Y - SnakeHead.Position.Y;
            double prevDist = Math.Sqrt(Math.Pow(xDiff, 2) + Math.Pow(yDiff, 2));

            var prevState = GetNormalizedState();

            int dir = MapAction(action);
            SnakeHead.Move(dir);

            double reward = StepPenalty;
            if (AteApple())
            {
                reward += AppleReward;
                StepsWithoutApple = 0;
                return (reward, GetNormalizedState(), false);
            }
            else if (Collided())
            {
                reward += CollisionPenalty;
                return (reward, prevState, true);
            }

            StepsWithoutApple++;
            if (StepsWithoutApple >= MaxStepsWithoutApple) return (TimeoutPenalty, GetNormalizedState(), true);

            xDiff = ApplePosition.X - SnakeHead.Position.X;
            yDiff = ApplePosition.Y - SnakeHead.Position.Y;
            double newDist = Math.Sqrt(Math.Pow(xDiff, 2) + Math.Pow(yDiff, 2));

            reward += DistRewardMult * (prevDist - newDist);

            return (reward, GetNormalizedState(), false);
        }

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

        bool AteApple()
        {
            if (SnakeHead.Position == ApplePosition)
            {
                SnakeHead.Grow();
                GenerateApple();
                return true;
            }
            else
            {
                return false;
            }
        }

        bool Collided()
        {
            return HitWall() || HitBody();
        }

        bool HitWall()
        {
            return SnakeHead.Position.X < 0 || SnakeHead.Position.X >= GridDims.X || SnakeHead.Position.Y < 0 || SnakeHead.Position.Y >= GridDims.Y;
        }

        bool HitBody()
        {
            SnakeNode? node = SnakeHead.Child;
            while (node is not null)
            {
                if (SnakeHead.Position == node.Position) return true;
                node = node.Child;
            }
            return false;
        }

        public override int PickAgentAction(Tensor qValues, Tensor? state = null) => Tensor.ArgMax(qValues);

        public override int PickRandomAction() => Random.Next(ActionCount);

        public override void Render(Episode episode, int step)
        {
            step = Math.Clamp(step, 0, episode.Experiences.Count);
            (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);
            string dirMoved = action switch
            {
                (int)Actions.Forward => "Forward",
                (int)Actions.Left => "Left",
                (int)Actions.Right => "Right",
                _ => "Invalid Action Made"
            };

            Console.WriteLine($"\nDirection Moved: {dirMoved}, Step: {step}, Reward: {reward}");
        }

        void DrawSnake()
        {
            var state = GetBoardState();
            for (int row = -1; row <= GridDims.Y; row++)
            {
                for (int col = -1; col <= GridDims.X; col++)
                {
                    bool rowEdge = row == -1 || row == GridDims.Y;
                    bool colEdge = col == -1 || col == GridDims.X;
                    if (rowEdge && colEdge) Console.Write("+");
                    else if (rowEdge) Console.Write("-");
                    else if (colEdge) Console.Write("|");
                    else
                    {
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

        public void Play(Model agent)
        {
            Reset();

            int stepsWithoutApple = 0;
            while (!Collided())
            {
                Console.Clear();
                int action = PickAgentAction(agent.Predict(Tensor.WrapBatch(GetNormalizedState())));
                SnakeHead.Move(MapAction(action));

                if (Collided()) break;

                if (AteApple()) stepsWithoutApple = 0;
                else stepsWithoutApple++;

                DrawSnake();

                if (stepsWithoutApple >= MaxStepsWithoutApple) break;

                Thread.Sleep(FrameTime);
            }

            Console.WriteLine("\nAgent collided or timed out!");
        }

        class SnakeNode(SnakeNode? parent = null, int x = 0, int y = 0)
        {
            public SnakeNode? Parent { get; } = parent;
            public int Direction { get; set; }
            public Int2 Position { get; set; } = new(x, y);
            Int2 PrevPosition { get; set; } = new();
            public SnakeNode? Child { get; private set; } = null;

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

                Child?.Move(Direction);

                Direction = dir;
            }

            public void Grow()
            {
                if (Child is not null)
                {
                    Child.Grow();
                }
                else
                {
                    Child = new(this, PrevPosition.X, PrevPosition.Y);
                }
            }
        }

        struct Int2(int x = 0, int y = 0)
        {
            public int X { get; set; } = x;
            public int Y { get; set; } = y;

            public static bool operator ==(Int2 a, Int2 b)
            {
                return a.X == b.X && a.Y == b.Y;
            }

            public static bool operator !=(Int2 a, Int2 b)
            {
                return !(a == b);
            }

            public override readonly bool Equals(object? obj) => obj is Int2 other && Equals(other);

            public readonly bool Equals(Int2 other) => this == other;

            public override readonly int GetHashCode() => HashCode.Combine(X, Y);
        }

        enum Actions { Left, Forward, Right }

        enum Movements { Left, Up, Right, Down }

        struct BoardEncoding
        {
            public const double Empty = 0.0;
            public const double Head = 1.0;
            public const double Body = 2.0;
            public const double Tail = 3.0;
            public const double Apple = 4.0;
        }
    }

    public interface ISelfPlay
    {
        public bool AgentTurn { get; set; }
        public int OpponentCount { get; set; }
        public int OpponentIndex { get; set; }
        public Random Random { get; init; }
        public Tensor State { get; init; }

        public int PickOpponentAction(FIFOBuffer<Model> agents)
        {
            if (agents.Count != OpponentCount) UpdateOpponentIndex(agents.Count);

            if (OpponentIndex >= OpponentCount)
            {
                return PickRandomAction();
            }
            else
            {
                return GetAgentAction(agents[OpponentIndex]);
            }
        }

        public int GetAgentAction(Model agent, Tensor? state = null);

        public int PickAgentAction(Tensor qValues, Tensor? state = null);

        public int PickRandomAction();

        void UpdateOpponentIndex(int newOpponentCount)
        {
            OpponentCount = newOpponentCount;
            OpponentIndex = Random.Next(OpponentCount + 1);
        }
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
        public int MaxSize { get; init; } = maxSize;
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

    public class DQNTrainer(Model agent, Environment environment, Optimizer optimizer, Cost cost, double discount = 0.995,
        double exploration = 1.0, double explorationDecay = 0.99, double minExploration = 0.01, int replayBufferSize = 10000, int batchSize = 64,
        int agentBufferSize = 5, int opponentCopyRate = 100, int minRandomOpponentEpisodes = 200, double tau = 0.005, double maxGradNorm = 1.0,
        int minExperiences = 1000)
    {
        readonly Random random = new();
        readonly Model Agent = agent;
        readonly Model TargetModel = agent.Copy();
        readonly Environment Environment = environment;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;
        readonly ReplayBuffer ReplayBuffer = new(replayBufferSize);
        readonly FIFOBuffer<Model> AgentBuffer = new(agentBufferSize);
        readonly int BatchSize = batchSize;
        readonly int OpponentCopyRate = opponentCopyRate;
        readonly int MinRandomOppEpisodes = minRandomOpponentEpisodes;
        readonly double Discount = discount;
        double Exploration = exploration;
        readonly double ExplorationDecay = explorationDecay;
        readonly double MinExploration = minExploration;
        int optimizerSteps = 0;
        readonly double Tau = tau;
        readonly double MaxNorm = maxGradNorm;
        readonly int MinExperiences = minExperiences;
        readonly bool SelfPlay = environment is ISelfPlay;
        double totalLoss = 0.0;

        Tensor? _currentBatch;
        Tensor? _nextBatch;
        Tensor? _predictedQs;
        Tensor? _targetQs;

        public void Train(ref FIFOBuffer<Episode>? episodeBuffer, int episodes = 1000)
        {
            List<Experience> episodeExperiences = [];
            Tensor state;
            Tensor trueState;
            bool done;
            bool learnerTurn;
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
                if (SelfPlay && ((e + 1) >= MinRandomOppEpisodes) && ((e + 1) % OpponentCopyRate == 0)) AgentBuffer.Add(Agent.Copy());

                totalLoss = 0.0;
                episodeExperiences.Clear();
                Environment.Reset();
                state = Environment.GetNormalizedState();

                done = false;
                step = 0;
                totalReward = 0;
                while (!done)
                {
                    step++;
                    trueState = Environment.GetState();
                    learnerTurn = Environment is not ISelfPlay sp || sp.AgentTurn;
                    action = PickNextAction(state);

                    (reward, nextState, done) = Environment.Step(action, step);
                    totalReward += reward;
                    if (learnerTurn) ReplayBuffer.Add(new(state, action, reward, nextState, done));
                    episodeExperiences.Add(new(trueState, action, reward, Environment.GetState(), done));

                    TrainNetwork();

                    state = nextState;
                }

                episodeBuffer?.Add(new(episodeExperiences));

                Exploration = Math.Max(Exploration * ExplorationDecay, MinExploration);

                var elapsed = stopwatch.Elapsed;
                avgElapsed += (elapsed - avgElapsed) / (e + 1);
                var eta = avgElapsed * (episodes - e - 1);

                Console.WriteLine($"\nEpisode {e + 1}/{episodes} finished...");
                Console.WriteLine($"Total Reward: {totalReward:F2},");
                Console.WriteLine($"Average Loss: {(totalLoss / step):F3}");
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

        int PickAgentAction(Tensor state)
        {
            if (random.NextDouble() < Exploration)
            {
                return Environment.PickRandomAction();
            }
            else
            {
                return Environment.PickAgentAction(Agent.Predict(Tensor.WrapBatch(state)));
            }
        }

        int PickOpponentAction()
        {
            if (Environment is ISelfPlay selfPlayEnv)
            {
                return selfPlayEnv.PickOpponentAction(AgentBuffer);
            }
            else throw new Exception("Environment not self-play");
        }

        void TrainNetwork()
        {
            if (ReplayBuffer.Count < MinExperiences) return;

            var (batch, weights) = ReplayBuffer.GetBatch(BatchSize);


            if (_currentBatch is null || _nextBatch is null)
            {
                var stateDims = batch[0].State.Dimensions;
                var batchDims = new int[stateDims.Length + 1];
                batchDims[0] = BatchSize;
                stateDims.CopyTo(batchDims, 1);
                _currentBatch = new(batchDims);
                _nextBatch = new(batchDims);
            }

            for (int b = 0; b < BatchSize; b++)
            {
                for (int i = 0; i < Environment.StateSize; i++)
                {
                    _currentBatch[b * Environment.StateSize + i] = batch[b].State[i];
                    _nextBatch[b * Environment.StateSize + i] = batch[b].NextState[i];
                }
            }

            var predictions = Agent.Forward(_currentBatch);
            var nextAgentQs = Agent.Predict(_nextBatch).Copy();
            var nextTargetQs = TargetModel.Predict(_nextBatch).Copy();

            var predictedQs = MaskQValues(predictions, batch);
            var targetQs = MaskQValuesDouble(nextAgentQs, nextTargetQs, batch);

            Agent.ZeroGrad();

            var lossResult = Cost.CalculateCostWithPriority(predictedQs, targetQs, weights);
            for (int i = 0; i < BatchSize; i++)
            {
                batch[i].Priority = lossResult.Priorities[i];
            }
            var loss = Tensor.Mean(lossResult.Losses);
            totalLoss += loss[0];

            loss.Backward();
            Agent.ClipGradients(MaxNorm);

            for (int i = 0; i < Agent.ParameterCount; i++)
            {
                Optimizer.Step(Agent.Parameters[i], optimizerSteps);
            }

            for (int i = 0; i < TargetModel.ParameterCount; i++)
            {
                for (int j = 0; j < TargetModel.Parameters[i].ElementCount; j++)
                {
                    TargetModel.Parameters[i][j] = Tau * Agent.Parameters[i][j]
                        + (1.0 - Tau) * TargetModel.Parameters[i][j];
                }
            }

            optimizerSteps++;
        }

        Tensor MaskQValues(Tensor qValues, List<Experience> batch)
        {
            _predictedQs ??= new([BatchSize, 1]);
            for (int i = 0; i < BatchSize; i++)
            {
                _predictedQs[i] = qValues[[i, batch[i].Action]];
            }
            return _predictedQs;
        }

        Tensor MaskQValuesDouble(Tensor agentQValues, Tensor targetQValues, List<Experience> batch)
        {
            _targetQs ??= new([BatchSize, 1]);
            int actionCount = agentQValues.Dimensions[^1];

            for (int i = 0; i < BatchSize; i++)
            {
                double qTarget = batch[i].Reward;

                if (!batch[i].Done)
                {
                    int bestAction = 0;
                    double bestQ = agentQValues[i * actionCount];
                    for (int a = 1; a < actionCount; a++)
                    {
                        double q = agentQValues[i * actionCount + a];
                        if (q > bestQ) { bestQ = q; bestAction = a; }
                    }

                    double evalQ = targetQValues[i * actionCount + bestAction];
                    qTarget += Discount * evalQ * (SelfPlay ? -1.0 : 1.0);
                }

                _targetQs[i] = qTarget;
            }

            return _targetQs;
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
            Tensor loss;

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
                    Console.WriteLine($"Epoch {e} : Loss = {loss[0]} : Time elapsed = {timer.ElapsedMilliseconds}ms : Time per epoch = {((float)timer.ElapsedMilliseconds / logEvery):F2}ms");
                    timer.Restart();
                }
            }
        }
    }

    public abstract class Optimizer(double learningRate)
    {
        protected readonly double LR = learningRate;

        public abstract void Step(Tensor parameter, int iterations);
    }

    public class SGD(double learningRate) : Optimizer(learningRate)
    {
        public override void Step(Tensor parameter, int iterations)
        {
            for (int i = 0; i < parameter.ElementCount; i++)
            {
                parameter[i] -= parameter.Grad[i] * LR;
            }
        }
    }

    public class Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : Optimizer(learningRate)
    {
        readonly double Beta1 = beta1;
        readonly double Beta2 = beta2;
        readonly double Epsilon = epsilon;

        // Per-Tensor moment state
        readonly Dictionary<Tensor, (double[] m, double[] v)> _state = [];

        public override void Step(Tensor parameter, int iteration)
        {
            iteration++;
            if (!_state.TryGetValue(parameter, out var moments))
            {
                moments = (new double[parameter.ElementCount], new double[parameter.ElementCount]);
                _state[parameter] = moments;
            }

            var (m, v) = moments;
            for (int i = 0; i < parameter.ElementCount; i++)
            {
                m[i] = Beta1 * m[i] + (1.0 - Beta1) * parameter.Grad[i];
                v[i] = Beta2 * v[i] + (1.0 - Beta2) * Math.Pow(parameter.Grad[i], 2.0);

                double mHat = m[i] / (1.0 - Math.Pow(Beta1, iteration));
                double vHat = v[i] / (1.0 - Math.Pow(Beta2, iteration));

                parameter.Data[i] -= (LR * mHat) / (Math.Sqrt(vHat) + Epsilon);
            }
        }
    }

    public class Model
    {
        public Layer[] Layers { get; private set; }
        public List<Tensor> Parameters
        {
            get
            {
                parameters ??= [.. GetParameters()];
                return parameters;
            }
        }
        List<Tensor>? parameters;
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
            int inputs = 1;
            for (int i = 1; i < inputFormat.Rank; i++)
            {
                inputs *= inputFormat.Dimensions[i];
            }

            foreach (var layer in Layers)
            {
                layer.SetUpLayer(inputs);
                inputs = layer.NeuronCount;
            }
        }

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

        public Tensor Predict(Tensor input)
        {
            Tensor.Inference = true;
            var output = Forward(input);
            Tensor.Inference = false;
            return output;
        }

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

        public void ClipGradients(double maxNorm)
        {
            double totalNorm = 0.0;
            foreach (var param in Parameters)
            {
                for (int i = 0; i < param.GradCount; i++)
                {
                    totalNorm += Math.Pow(param.Grad[i], 2.0);
                }
            }
            totalNorm = Math.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                double scale = maxNorm / (totalNorm + 1e-8);
                foreach (var param in Parameters)
                {
                    for (int i = 0; i < param.GradCount; i++)
                    {
                        param.Grad[i] *= scale;
                    }
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var param in Parameters)
            {
                param.ZeroGrad();
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

        public abstract void SetUpLayer(int inputCount);

        public abstract Tensor Forward(Tensor input);

        public abstract IEnumerable<Tensor> GetParameters();

        public abstract Layer Copy();

        public abstract void BuildFromData(Saver.LayerData data);
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
            Biases = Tensor.InitBiases(NeuronCount);
        }

        public override Tensor Forward(Tensor input)
        {
            var flatInput = input.Rank > 2 ? Tensor.Flatten(input, 1) : input;
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
            NeuronCount = data.NeuronCount;

            Weights = data.Weights;
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

    public abstract class Activation
    {
        public abstract Tensor Forward(Tensor input);

        public abstract Activation Copy();
    }

    public class ReLU : Activation
    {
        public override Tensor Forward(Tensor input) => Tensor.ReLU(input);

        public override Activation Copy() => new ReLU();
    }

    public class LeakyReLU : Activation
    {
        readonly double Tau;

        public LeakyReLU(double tau = 0.01)
        {
            Tau = tau;
        }

        public LeakyReLU() { }

        public override Tensor Forward(Tensor input) => Tensor.LeakyReLU(input, Tau);

        public override Activation Copy() => new LeakyReLU(Tau);
    }

    public class Tanh : Activation
    {
        public override Tensor Forward(Tensor input) => Tensor.Tanh(input);

        public override Activation Copy() => new Tanh();
    }

    public class Sigmoid : Activation
    {
        public override Tensor Forward(Tensor input) => Tensor.Sigmoid(input);

        public override Activation Copy() => new Sigmoid();
    }

    public class Linear : Activation
    {
        public override Tensor Forward(Tensor input) => input;

        public override Activation Copy() => new Linear();
    }

    public record CostResult(Tensor Losses, double[] Priorities);

    public abstract class Cost
    {
        public abstract Tensor CalculateCost(Tensor input, Tensor target);

        public abstract Tensor CalculatePerSampleCost(Tensor input, Tensor target);

        public virtual CostResult CalculateCostWithPriority(Tensor input, Tensor target, double[]? weights = null)
        {
            var losses = CalculatePerSampleCost(input, target);
            var priorities = new double[losses.ElementCount];

            for (int i = 0; i < losses.ElementCount; i++)
            {
                priorities[i] = Math.Abs(losses[i]) + 1e-8;

                if (weights is not null) losses[i] *= weights[i];
            }

            return new(losses, priorities);
        }
    }

    public class MSE : Cost
    {
        public override Tensor CalculateCost(Tensor input, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(input, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor input, Tensor target)
        {
            return Tensor.Pow(input - target, 2.0);
        }
    }

    public class Huber(double delta = 1.0) : Cost
    {
        readonly double Delta = delta;

        public override Tensor CalculateCost(Tensor input, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(input, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor input, Tensor target)
        {
            var diff = input - target;

            // Huber: delta^2 * (sqrt(1 + (diff/delta)^2) - 1)
            var scaled = diff / Delta;
            var inner = Tensor.Pow(scaled, 2.0) + 1.0;
            return (Tensor.Pow(inner, 0.5) - 1.0) * Math.Pow(Delta, 2.0);
        }
    }

    [Serializable]
    public class Tensor
    {
        // Linear value storage
        public double[] Data { get; init; } = [];
        public double[] Grad { get; private set; } = [];

        // Shape properties
        public int Rank => Dimensions.Length;
        public int ElementCount => Data.Length;
        public int GradCount => Grad.Length;

        // Index mapping
        public int[] Dimensions { get; init; } = [];
        public int[] Strides { get; init; } = [];

        // AutoGrad graph
        public static bool Inference { get; set; } = false;
        public bool RequiresGrad { get; set; }
        readonly List<Tensor> _parents = [];
        readonly List<Tensor> _results = [];
        int _opIndex = 0;
        Action _backward = delegate { };
        static int _forwardGen = 0;
        int _lastGen = -1;
        List<Tensor>? _topo = null;
        HashSet<Tensor>? _visited = null;

        // Optimizations
        static readonly int VectorSize = Vector<double>.Count;
        const long ParallelThreshold = 500_000;

        // Parameterless constructor for JsonSerializer
        public Tensor() { }

        // Primary constructor
        public Tensor(int[] dims, bool requiresGrad = false)
        {
            Dimensions = (int[])dims.Clone();
            Strides = ComputeStrides(dims);

            int size = 1;
            foreach (var dim in dims) size *= dim;

            Data = new double[size];

            RequiresGrad = requiresGrad;
            if (RequiresGrad) Grad = new double[size];
        }

        // Calculate the stride represented by each index when accessing by multidimensional indices
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

        // Convert multidimensional indices to linear index
        public int LinearIndex(params int[] indices)
        {
            int offset = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                offset += indices[i] * Strides[i];
            }

            return offset;
        }

        int LinearIndex(Span<int> indices)
        {
            int offset = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                offset += indices[i] * Strides[i];
            }

            return offset;
        }

        // Zero out the current accumulated gradient
        public void ZeroGrad()
        {
            if (RequiresGrad) Array.Clear(Grad, 0, GradCount);

            foreach (var r in _results)
            {
                r.ZeroGrad();
            }
        }

        // Restores the gradient array to match the data array
        public void RestoreGrad() => Grad = new double[ElementCount];

        public static void BeginForward()
        {
            if (!Inference) _forwardGen++;
        }

        void PrepareForward()
        {
            if (_lastGen == _forwardGen) return;

            _opIndex = 0;

            foreach (var r in _results)
            {
                r._parents.Clear();
                r._backward = delegate { };
            }

            _lastGen = _forwardGen;
        }

        void FinalizeForward()
        {
            if (_opIndex < _results.Count)
            {
                _results.RemoveRange(_opIndex, _results.Count - _opIndex);
            }
        }

        // Calculate the gradients of all Tensors in the current graph
        public void Backward()
        {
            Array.Fill(Grad, 1.0);

            _topo ??= [];
            _visited ??= [];

            _topo.Clear();
            _visited.Clear();

            BuildTopo(this, _topo, _visited);

            for (int i = _topo.Count - 1; i >= 0; i--)
            {
                _topo[i]._backward();
            }

            foreach (var t in _topo)
            {
                t.FinalizeForward();
            }
        }

        // Build the topography of the current function graph
        static void BuildTopo(Tensor t, List<Tensor> topo, HashSet<Tensor> visited)
        {
            if (visited.Contains(t)) return;
            visited.Add(t);

            foreach (var p in t._parents)
            {
                BuildTopo(p, topo, visited);
            }

            topo.Add(t);
        }

        // Constructor for a Tensor filled with only one value
        public static Tensor Scalar(double value, int[] dims, bool requiresGrad = false)
        {
            Tensor t = new(dims, requiresGrad);
            Array.Fill(t.Data, value);
            return t;
        }

        // Primary accessor using multidimensional indices
        public double this[int[] indices]
        {
            get => Data[LinearIndex(indices)];
            set => Data[LinearIndex(indices)] = value;
        }

        public double this[int index]
        {
            get => Data[index];
            set => Data[index] = value;
        }

        // Element-wise addition operator
        public static Tensor operator +(Tensor a, Tensor b)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] + bVecs[i];
            }
            
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] + b[i];
            }

            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] += rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator +(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] + vb;
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] + b;
            }

            if (!Inference)
            {
                result._parents.Add(a);

                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i];
                    }

                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator +(double a, Tensor b) => b + a;
        

        // Element-wise subtraction operator
        public static Tensor operator -(Tensor a, Tensor b)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] - bVecs[i];
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] - b[i];
            }

            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] -= rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] -= result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator -(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] - vb;
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] - b;
            }

            if (!Inference)
            {
                result._parents.Add(a);

                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i];
                    }

                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator -(double a, Tensor b)
        {
            Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);

            var va = new Vector<double>(a);
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
            
            for (int i = 0; i < bVecs.Length; i++)
            {
                rVecs[i] = va - bVecs[i];
            }

            for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a - b[i];
            }

            if (!Inference)
            {
                result._parents.Add(b);

                result._backward = () =>
                {
                    if (!b.RequiresGrad) return;

                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < bgVecs.Length; i++)
                    {
                        bgVecs[i] -= rgVecs[i];
                    }

                    for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                    {
                        b.Grad[i] -= result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Element-wise multiplication operator
        public static Tensor operator *(Tensor a, Tensor b)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] * bVecs[i];
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] * b[i];
            }

            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += bvVecs[i] * rgVecs[i];
                        if (b.RequiresGrad) bgVecs[i] += avVecs[i] * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += b[i] * result.Grad[i];
                        if (b.RequiresGrad) b.Grad[i] += a[i] * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator *(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] * vb;
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] * b;
            }

            if (!Inference)
            {
                result._parents.Add(a);

                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var vb = new Vector<double>(b);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += vb * rgVecs[i];
                    }

                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += b * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator *(double a, Tensor b) => b * a;

        // Element-wise division operator
        public static Tensor operator /(Tensor a, Tensor b)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || b.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] / bVecs[i];
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] / b[i];
            }

            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !b.RequiresGrad) return;

                    var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        if (a.RequiresGrad) agVecs[i] += rgVecs[i] / bvVecs[i];
                        if (b.RequiresGrad) bgVecs[i] -= (avVecs[i] / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += result.Grad[i] / b[i];
                        if (b.RequiresGrad) b.Grad[i] -= (a[i] / Math.Pow(b[i], 2.0)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator /(Tensor a, double b)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var vb = new Vector<double>(b);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = aVecs[i] / vb;
            }

            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a[i] / b;
            }

            if (!Inference)
            {
                result._parents.Add(a);

                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var vb = new Vector<double>(b);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += rgVecs[i] / vb;
                    }

                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += (1.0 / b) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor operator /(double a, Tensor b)
        {
            Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);

            var va = new Vector<double>(a);
            var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < bVecs.Length; i++)
            {
                rVecs[i] = va / bVecs[i];
            }

            for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = a / b[i];
            }

            if (!Inference)
            {
                result._parents.Add(b);

                result._backward = () =>
                {
                    if (!b.RequiresGrad) return;

                    var va = new Vector<double>(a);
                    var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < bgVecs.Length; i++)
                    {
                        bgVecs[i] -= (va / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                    }

                    for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                    {
                        b.Grad[i] -= (a / Math.Pow(b[i], 2.0)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Element-wise exponentiation
        public static Tensor Pow(Tensor a, Tensor exp)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad || exp.RequiresGrad);

            for (int i = 0; i < result.ElementCount; i++)
            {
                result[i] = Math.Pow(a[i], exp[i]);
            }

            if (!Inference)
            {
                result._parents.AddRange([a, exp]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!a.RequiresGrad && !exp.RequiresGrad) return;

                    for (int i = 0; i < result.ElementCount; i++)
                    {
                        if (a.RequiresGrad) a.Grad[i] += exp[i] * Math.Pow(a[i], exp[i] - 1.0) * result.Grad[i];
                        if (exp.RequiresGrad) exp.Grad[i] += result[i] * Math.Log(a[i]) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor Pow(Tensor a, double exp)
        {
            Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

            if (exp == 2.0 || exp == 0.5)
            {
                var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

                for (int i = 0; i < aVecs.Length; i++)
                {
                    rVecs[i] = exp == 2.0 ? aVecs[i] * aVecs[i] : Vector.SquareRoot(aVecs[i]);
                }

                for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    result[i] = exp == 2.0 ? a[i] * a[i] : Math.Sqrt(a[i]);
                }
            }
            else
            {
                for (int i = 0; i < result.ElementCount; i++)
                {
                    result[i] = Math.Pow(a[i], exp);
                }
            }

            if (!Inference)
            {
                result._parents.Add(a);

                result._backward = () =>
                {
                    if (!a.RequiresGrad) return;

                    if (exp == 2.0 || exp == 0.5)
                    {
                        var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                        var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                        var vexp = new Vector<double>(exp);
                        var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                        for (int i = 0; i < agVecs.Length; i++)
                        {
                            agVecs[i] += exp == 2.0 ? vexp * avVecs[i] * rgVecs[i] : (vexp / Vector.SquareRoot(avVecs[i])) * rgVecs[i];
                        }

                        for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                        {
                            a.Grad[i] += exp * Math.Pow(a[i], exp - 1.0) * result.Grad[i];
                        }
                    }
                    else
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

        public static Tensor Pow(double a, Tensor exp)
        {
            Tensor result = GetResultTensor(exp, exp.Dimensions, exp.RequiresGrad);

            for (int i = 0; i < result.ElementCount; i++)
            {
                result[i] = Math.Pow(a, exp[i]);
            }

            if (!Inference)
            {
                result._parents.Add(exp);

                result._backward = () =>
                {
                    if (!exp.RequiresGrad) return;

                    var expgVecs = MemoryMarshal.Cast<double, Vector<double>>(exp.Grad.AsSpan());
                    double lna = Math.Log(a);
                    var vlna = new Vector<double>(lna);
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < expgVecs.Length; i++)
                    {
                        expgVecs[i] += rvVecs[i] * vlna * rgVecs[i];
                    }

                    for (int i = expgVecs.Length * VectorSize; i < exp.ElementCount; i++)
                    {
                        exp.Grad[i] += result[i] * lna * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Exponentiation with base e
        public static Tensor Exp(Tensor t)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Exp(tVecs[i]);
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Exp(t[i]);
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        tgVecs[i] += rvVecs[i] * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += result[i] * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Element-wise logarithm
        public static Tensor Log(Tensor logBase, Tensor arg)
        {
            // Calculate each element of the resulting Tensor
            Tensor result = GetResultTensor(logBase, logBase.Dimensions, logBase.RequiresGrad || arg.RequiresGrad);

            var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
            var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < lbVecs.Length; i++)
            {
                rVecs[i] = Vector.Log(argVecs[i]) / Vector.Log(lbVecs[i]);
            }

            for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg[i], logBase[i]);
            }

            if (!Inference)
            {
                result._parents.AddRange([logBase, arg]);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!logBase.RequiresGrad && !arg.RequiresGrad) return;

                    var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                    var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                    var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                    var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var lnb = Vector.Log(lbvVecs[i]);
                        if (logBase.RequiresGrad)
                        {
                            lbgVecs[i] -= (Vector.Log(argvVecs[i]) / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                        }
                        if (arg.RequiresGrad) arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnb);
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        double lnb = Math.Log(logBase[i]);
                        if (logBase.RequiresGrad) logBase.Grad[i] -= (Math.Log(arg[i]) / (logBase[i] * Math.Pow(lnb, 2.0))) * result.Grad[i];
                        if (arg.RequiresGrad) arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor Log(Tensor logBase, double arg)
        {
            Tensor result = GetResultTensor(logBase, logBase.Dimensions, logBase.RequiresGrad);

            var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
            var lnvarg = Vector.Log(new Vector<double>(arg));
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < lbVecs.Length; i++)
            {
                rVecs[i] = lnvarg / Vector.Log(lbVecs[i]);
            }

            for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg, logBase[i]);
            }

            if (!Inference)
            {
                result._parents.Add(logBase);

                result._backward = () =>
                {
                    if (!logBase.RequiresGrad) return;

                    var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                    var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                    double lnarg = Math.Log(arg);
                    var lnvarg = new Vector<double>(lnarg);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var lnb = Vector.Log(lbvVecs[i]);
                        lbgVecs[i] -= (lnvarg / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        logBase.Grad[i] -= (lnarg / (logBase[i] * Math.Pow(Math.Log(logBase[i]), 2.0))) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        public static Tensor Log(double logBase, Tensor arg)
        {
            Tensor result = GetResultTensor(arg, arg.Dimensions, arg.RequiresGrad);

            var lnvlb = Vector.Log(new Vector<double>(logBase));
            var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < argVecs.Length; i++)
            {
                rVecs[i] = Vector.Log(argVecs[i]) / lnvlb;
            }

            for (int i = argVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Log(arg[i], logBase);
            }

            if (!Inference)
            {
                result._parents.Add(arg);

                result._backward = () =>
                {
                    if (!arg.RequiresGrad) return;

                    double lnb = Math.Log(logBase);
                    var lnvlb = new Vector<double>(lnb);
                    var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                    var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnvlb);
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // 2D Matrix multiplication operator
        public static Tensor operator ^(Tensor a, Tensor b)
        {
            // Last two dimensions are matrix multiplication dimensions, everything ahead is batch
            int rank = a.Rank;
            int m = a.Dimensions[^2];
            int n = a.Dimensions[^1];
            int p = b.Dimensions[^1];

            int batchSize = 1;
            for (int i = 0; i < rank - 2; i++) batchSize *= a.Dimensions[i];
            int aMatSize = m * n;
            int bMatSize = n * p;
            int rMatSize = m * p;

            int totalRows = batchSize * m;

            // Build result dims -> batch dims + [m, p]
            var resultDims = (int[])a.Dimensions.Clone();
            resultDims[^1] = p;

            Tensor result = GetResultTensor(a, resultDims, a.RequiresGrad || b.RequiresGrad);

            bool useParallel = (long)totalRows * n * p > ParallelThreshold;
            double[] bT = ArrayPool<double>.Shared.Rent(bMatSize * batchSize);
            try
            {
                for (int batch = 0; batch < batchSize; batch++)
                {
                    TransposeMatrix(b.Data, bT, batch * bMatSize, batch * bMatSize, n, p);
                }

                if (useParallel)
                {
                    Parallel.For(0, batchSize * m, row =>
                    {
                        int batch = row / m;
                        int i = row % m;
                        ComputeRow(i, n, p, a.Data, bT, result.Data, batch * aMatSize, batch * bMatSize, batch * rMatSize);
                    });
                }
                else
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
                ArrayPool<double>.Shared.Return(bT);
            }

            if (!Inference)
            {
                result._parents.AddRange([a, b]);

                // Gradient calculation function
                result._backward = () =>
                {
                    for (int batch = 0; batch < batchSize; batch++)
                    {
                        int aOff = batch * aMatSize;
                        int bOff = batch * bMatSize;
                        int rOff = batch * rMatSize;
                        bool par = (long)m * n * p > ParallelThreshold;

                        if (a.RequiresGrad && b.RequiresGrad)
                        {
                            double[] bT = ArrayPool<double>.Shared.Rent(bMatSize);
                            double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                            double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                            try
                            {
                                TransposeMatrix(b.Data, bT, bOff, 0, n, p);
                                TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                                TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p);

                                if (par)
                                {
                                    Parallel.For(0, m, i =>
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, bT, rOff + i * p, k * n, p);
                                        }
                                    });

                                    Parallel.For(0, n, k =>
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    });
                                }
                                else
                                {
                                    for (int i = 0; i < m; i++)
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, bT, rOff + i * p, k * n, p); 
                                        }
                                    }

                                    for (int k = 0; k < n; k++)
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                ArrayPool<double>.Shared.Return(bT);
                                ArrayPool<double>.Shared.Return(aT);
                                ArrayPool<double>.Shared.Return(dOutT);
                            }
                        }
                        else if (a.RequiresGrad)
                        {
                            double[] bT = ArrayPool<double>.Shared.Rent(bMatSize);
                            try
                            {
                                TransposeMatrix(b.Data, bT, bOff, 0, n, p);

                                if (par)
                                {
                                    Parallel.For(0, m, i =>
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, bT, rOff + i * p, k * n, p);
                                        }
                                    });
                                }
                                else
                                {
                                    for (int i = 0; i < m; i++)
                                    {
                                        for (int k = 0; k < n; k++)
                                        {
                                            a.Grad[aOff + i * n + k] += DotProduct(result.Grad, bT, rOff + i * p, k * n, p);
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                ArrayPool<double>.Shared.Return(bT);
                            }
                        }
                        else if (b.RequiresGrad)
                        {
                            double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                            double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                            try
                            {
                                TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                                TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p);

                                if (par)
                                {
                                    Parallel.For(0, n, k =>
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    });
                                }
                                else
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        for (int j = 0; j < p; j++)
                                        {
                                            b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                ArrayPool<double>.Shared.Return(aT);
                                ArrayPool<double>.Shared.Return(dOutT);
                            }
                        }
                    }
                };
            }

            return result;
        }

        static void TransposeMatrix(double[] src, double[] dst, int srcOff, int dstOff, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    dst[dstOff + c * rows + r] = src[srcOff + r * cols + c];
                }
            }
        }

        static void ComputeRow(int i, int n, int p, double[] a, double[] bT, double[] r, int aOff, int bTOff, int rOff)
        {
            for (int j = 0; j < p; j++)
            {
                r[rOff + i * p + j] = DotProduct(a, bT, aOff + i * n, bTOff + j * n, n);
            }
        }

        static double DotProduct(double[] x, double[] y, int xOff, int yOff, int len)
        {
            var xVecs = MemoryMarshal.Cast<double, Vector<double>>(x.AsSpan(xOff, len));
            var yVecs = MemoryMarshal.Cast<double, Vector<double>>(y.AsSpan(yOff, len));

            var acc = Vector<double>.Zero;
            for (int i = 0; i < xVecs.Length; i++)
            {
                acc += xVecs[i] * yVecs[i];
            }
            double sum = Vector.Sum(acc);
            for (int i = xVecs.Length * VectorSize; i < len; i++)
            {
                sum += x[xOff + i] * y[yOff + i];
            }

            return sum;
        }

        // Rectified Linear Unit activation function
        public static Tensor ReLU(Tensor t)
        {
            // Apply ReLU function to each element
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Max(tVecs[i], Vector<double>.Zero);
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Max(t[i], 0.0);
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var mask = Vector.GreaterThan(tvVecs[i], Vector<double>.Zero);
                        tgVecs[i] += Vector.ConditionalSelect(mask, rgVecs[i], Vector<double>.Zero);
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += (t[i] > 0.0 ? 1.0 : 0.0) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Leaky Rectified Linear Unit activation function
        public static Tensor LeakyReLU(Tensor t, double tau)
        {
            // Apply LeakyReLU function to each element
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vtau = new Vector<double>(tau);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                var mask = Vector.GreaterThan(tVecs[i], Vector<double>.Zero);
                rVecs[i] = Vector.ConditionalSelect(mask, tVecs[i], vtau * tVecs[i]);
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = t[i] > 0.0 ? t[i] : tau * t[i];
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vtau = new Vector<double>(tau);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        var mask = Vector.GreaterThan(tvVecs[i], Vector<double>.Zero);
                        tgVecs[i] += Vector.ConditionalSelect(mask, Vector<double>.One, vtau) * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += (t[i] > 0.0 ? 1.0 : tau) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Sigmoid activation function
        public static Tensor Sigmoid(Tensor t)
        {
            // Apply Sigmoid function to each element
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vone = Vector<double>.One;
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = vone / (vone + Vector.Exp(-tVecs[i]));
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = MathUtils.Sigmoid(t[i]);
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        tgVecs[i] += rvVecs[i] * (Vector<double>.One - rvVecs[i]) * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += result[i] * (1.0 - result[i]) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Hyperbolic tangent activation function
        public static Tensor Tanh(Tensor t)
        {
            // Apply Tanh function to each element
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vone = Vector<double>.One;
            var vtwo = new Vector<double>(2.0);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                var e2x = Vector.Exp(vtwo * tVecs[i]);
                rVecs[i] = (e2x - vone) / (e2x + vone);
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = MathUtils.Tanh(t[i]);
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < rgVecs.Length; i++)
                    {
                        tgVecs[i] += (Vector<double>.One - (rvVecs[i] * rvVecs[i])) * rgVecs[i];
                    }

                    for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                    {
                        t.Grad[i] += (1.0 - Math.Pow(result[i], 2.0)) * result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Softmax function
        public static Tensor Softmax(Tensor t)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            int classes = t.Dimensions[^1];
            int batchSize = t.ElementCount / classes;

            // Apply Softmax function to each element
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * classes;

                var tSlice = t.Data.AsSpan(offset, classes);
                var tVecs = MemoryMarshal.Cast<double, Vector<double>>(tSlice);
                var rSlice = result.Data.AsSpan(offset, classes);
                var rVecs = MemoryMarshal.Cast<double, Vector<double>>(rSlice);

                var vmax = new Vector<double>(double.MinValue);
                for (int i = 0; i < tVecs.Length; i++)
                {
                    vmax = Vector.Max(vmax, tVecs[i]);
                }
                double max = double.MinValue;
                for (int lane = 0; lane < VectorSize; lane++)
                {
                    max = Math.Max(max, vmax[lane]);
                }
                for (int i = tVecs.Length * VectorSize; i < classes; i++)
                {
                    max = Math.Max(max, tSlice[i]);
                }

                var vmaxSplat = new Vector<double>(max);
                var acc = Vector<double>.Zero;
                for (int i = 0; i < tVecs.Length; i++)
                {
                    rVecs[i] = Vector.Exp(tVecs[i] - vmaxSplat);
                    acc += rVecs[i];
                }
                double sum = Vector.Sum(acc);

                for (int i = tVecs.Length * VectorSize; i < classes; i++)
                {
                    rSlice[i] = Math.Exp(tSlice[i] - max);
                    sum += rSlice[i];
                }

                var vsumSplat = new Vector<double>(sum);
                for (int i = 0; i < rVecs.Length; i++)
                {
                    rVecs[i] /= vsumSplat;
                }
                
                for (int i = rVecs.Length * VectorSize; i < classes; i++)
                {
                    rSlice[i] /= sum;
                }
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    // Jacobian-vector product: dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                    for (int b = 0; b < batchSize; b++)
                    {
                        int offset = b * classes;

                        var tgSlice = t.Grad.AsSpan(offset, classes);
                        var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(tgSlice);
                        var rvSlice = result.Data.AsSpan(offset, classes);
                        var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(rvSlice);
                        var rgSlice = result.Grad.AsSpan(offset, classes);
                        var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);

                        var vdot = Vector<double>.Zero;
                        for (int i = 0; i < rvVecs.Length; i++)
                        {
                            vdot += rgVecs[i] * rvVecs[i];
                        }
                        double dot = Vector.Sum(vdot);
                        for (int i = rvVecs.Length * VectorSize; i < classes; i++)
                        {
                            dot += rgSlice[i] * rvSlice[i];
                        }

                        var vdotSplat = new Vector<double>(dot);
                        for (int i = 0; i < tgVecs.Length; i++)
                        {
                            tgVecs[i] += rvVecs[i] * (rgVecs[i] - vdotSplat);
                        }

                        for (int i = tgVecs.Length * VectorSize; i < classes; i++)
                        {
                            tgSlice[i] += rvSlice[i] * (rgSlice[i] - dot);
                        }
                    }
                };
            }

            return result;
        }

        // Flattened Sum function
        public static Tensor Sum(Tensor t)
        {
            // Calculate sum of all elements
            Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var acc = Vector<double>.Zero;

            for (int i = 0; i < tVecs.Length; i++)
            {
                acc += tVecs[i];
            }
            double sum = Vector.Sum(acc);

            for (int i = tVecs.Length * VectorSize; i < t.ElementCount ; i++)
            {
                sum += t[i];
            }

            result[0] = sum;

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vrg = new Vector<double>(result.Grad[0]);

                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += vrg;
                    }

                    for (int i = tgVecs.Length * VectorSize; i < t.ElementCount; i++)
                    {
                        t.Grad[i] += result.Grad[0];
                    }
                };
            }

            return result;
        }

        // Mean function
        public static Tensor Mean(Tensor t)
        {
            // Calculate mean of all elements
            Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var acc = Vector<double>.Zero;

            for (int i = 0; i < tVecs.Length; i++)
            {
                acc += tVecs[i];
            }
            double sum = Vector.Sum(acc);

            for (int i = tVecs.Length * VectorSize; i < t.ElementCount; i++)
            {
                sum += t[i];
            }

            result[0] = sum / t.ElementCount;

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    double rg = result.Grad[0] / t.ElementCount;
                    var vrg = new Vector<double>(rg);

                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += vrg;
                    }

                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += rg;
                    }
                };
            }

            return result;
        }

        // Find index of largest number
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
        public static Tensor Transpose(Tensor t, int[]? axes = null)
        {
            // Default: reverse all axes
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

            Span<int> srcIndices = stackalloc int[t.Rank];
            Span<int> dstIndices = stackalloc int[axes.Length];

            // Remap each element
            for (int i = 0; i < t.ElementCount; i++)
            {
                t.GetFullIndices(i, srcIndices);
                for (int j = 0; j < axes.Length; j++)
                {
                    dstIndices[j] = srcIndices[axes[j]];
                }
                result[result.LinearIndex(dstIndices)] = t[i];
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Inverse permutation
                var invAxes = new int[axes.Length];
                for (int i = 0; i < axes.Length; i++)
                {
                    invAxes[axes[i]] = i;
                }

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    Span<int> srcIndices = stackalloc int[result.Rank];
                    Span<int> dstIndices = stackalloc int[invAxes.Length];

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

        // Broadcast a Tensor along a new dimension
        public static Tensor Broadcast(Tensor t, int[] targetDims)
        {
            // T.Dimensions must be a suffix of targetDims (t -> [16], targetDims = [32, 16])

            Tensor result = GetResultTensor(t, targetDims, t.RequiresGrad);

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
                Span<int> indices = stackalloc int[result.Rank];
                Span<int> srcIndices = stackalloc int[t.Rank];

                for (int i = 0; i < result.ElementCount; i++)
                {
                    result.GetFullIndices(i);
                    indices[^t.Rank..].CopyTo(srcIndices);
                    result[i] = t[t.LinearIndex(srcIndices)];
                }
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    if (t.Rank == 1)
                    {
                        int stride = t.ElementCount;
                        int rows = result.ElementCount / stride;

                        var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());

                        for (int r = 0; r < rows; r++)
                        {
                            var rgSlice = result.Grad.AsSpan(r * stride, stride);
                            var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);
                            for (int i = 0; i < tgVecs.Length; i++)
                            {
                                tgVecs[i] += rgVecs[i];
                            }
                            for (int i = tgVecs.Length * VectorSize; i < stride; i++)
                            {
                                t.Grad[i] += rgSlice[i];
                            }
                        }
                    }
                    else
                    {
                        Span<int> indices = stackalloc int[result.Rank];
                        Span<int> srcIndices = stackalloc int[t.Rank];

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

        // Flatten Tensor dimensions starting from startAxis onward
        public static Tensor Flatten(Tensor t, int startAxis = 0)
        {
            int flatSize = 1;
            for (int i = startAxis; i < t.Rank; i++) flatSize *= t.Dimensions[i];

            var newDims = new int[startAxis + 1];
            for (int i = 0; i < startAxis; i++) newDims[i] = t.Dimensions[i];
            newDims[^1] = flatSize;

            return Reshape(t, newDims);
        }

        // Reinterpret dimensions without moving data
        public static Tensor Reshape(Tensor t, int[] newDims)
        {
            Tensor result = GetResultTensor(t, newDims, t.RequiresGrad);

            Array.Copy(t.Data, result.Data, t.ElementCount);

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += rgVecs[i];
                    }

                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += result.Grad[i];
                    }
                };
            }

            return result;
        }

        // Add a leading dimension with length 1 to represent as batch
        public static Tensor WrapBatch(Tensor t)
        {
            var batchDims = new int[t.Rank + 1];
            batchDims[0] = 1;
            Array.Copy(t.Dimensions, 0, batchDims, 1, t.Rank);

            Tensor batch = GetResultTensor(t, batchDims, t.RequiresGrad);

            Array.Copy(t.Data, batch.Data, t.ElementCount);

            if (!Inference)
            {
                batch._parents.Add(t);

                // Gradient calculation function
                batch._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(batch.Grad.AsSpan());

                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += bgVecs[i];
                    }

                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += batch.Grad[i];
                    }
                };
            }

            return batch;
        }

        static Tensor GetResultTensor(Tensor owner, int[] dims, bool requiresGrad)
        {
            owner.PrepareForward();
            bool newOp = owner._results.Count <= owner._opIndex;

            Tensor result;
            if (newOp)
            {
                result = new(dims, requiresGrad);
                owner._results.Add(result);
            }
            else
            {
                result = owner._results[owner._opIndex];

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

                if (shapeMismatch)
                {
                    result = new(dims, requiresGrad);
                    owner._results[owner._opIndex] = result;
                }
                else
                {
                    result._parents.Clear();
                    result._backward = delegate { };
                }
            }

            owner._opIndex++;

            return result;
        }

        // Convert linear index to multidimensional indices
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

        void GetFullIndices(int index, Span<int> indices)
        {
            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = index % Dimensions[i];
                index /= Dimensions[i];
            }
        }

        // Initialize neural network weights using He Initialization
        public static Tensor InitWeights(int inputCount, int neuronCount)
        {
            Tensor weights = new([inputCount, neuronCount], true);

            for (int i = 0; i < weights.ElementCount; i++)
            {
                weights[i] = MathUtils.NextGaussian(0, Math.Sqrt(2.0 / inputCount));
            }

            return weights;
        }

        // Initialize neural network biases
        public static Tensor InitBiases(int neuronCount) => Scalar(0.01, [neuronCount], true);

        // Clip gradients
        public static Tensor Clip(Tensor t, double min, double max)
        {
            Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
            var vmin = new Vector<double>(min);
            var vmax = new Vector<double>(max);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Min(Vector.Max(tVecs[i], vmin), vmax);
            }

            for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = Math.Clamp(t[i], min, max);
            }

            if (!Inference)
            {
                result._parents.Add(t);

                // Gradient calculation function
                result._backward = () =>
                {
                    if (!t.RequiresGrad) return;

                    var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                    var vmin = new Vector<double>(min);
                    var vmax = new Vector<double>(max);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        var inRange = Vector.BitwiseAnd(
                            Vector.GreaterThanOrEqual(tvVecs[i], vmin),
                            Vector.LessThanOrEqual(tvVecs[i], vmax));
                        tgVecs[i] += Vector.ConditionalSelect(inRange, rgVecs[i], Vector<double>.Zero);
                    }

                    for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                    {
                        t.Grad[i] += (t[i] >= min && t[i] <= max) ? result.Grad[i] : 0;
                    }
                };
            }

            return result;
        }

        // Create an identical Tensor detached from the existing graph
        public Tensor Copy()
        {
            Tensor copy = new(Dimensions, false);
            Array.Copy(Data, copy.Data, ElementCount);
            return copy;
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

        public static TimeSpan RoundToMS(TimeSpan input)
        {
            return TimeSpan.FromMilliseconds(Math.Round(input.TotalMilliseconds));
        }
    }

    public static class Saver
    {
        const string DirectoryName = "Models";
        static string DirectoryPath = string.Empty;
        const string Extension = ".nnn";

#pragma warning disable CS8604 // Possible null reference argument.
        public static void SaveModel(Model model, string fileName)
        {
            InitDirectory();

            string filePath = Path.Combine(DirectoryPath, fileName + Extension);

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

            File.WriteAllText(filePath, json);
        }

        public static Model LoadModel(string fileName)
        {
            InitDirectory();

            string filePath = Path.Combine(DirectoryPath, fileName + Extension);

            string json = File.ReadAllText(filePath);

            var modelData = JsonSerializer.Deserialize<ModelData>(json);

            Model model = new(modelData);

            return model;
        }
#pragma warning restore CS8604 // Possible null reference argument.

        public static bool FileExists(string fileName)
        {
            InitDirectory();

            string filePath = Path.Combine(DirectoryPath, fileName + Extension);

            if (File.Exists(filePath)) return true;
            else return false;
        }

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