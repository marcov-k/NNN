using NNNCSharp.Components.Episodes;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Utilities.SaveSystem;
using static NNNCSharp.Components.Utilities.UIUtils;
using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Environments;

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
    /// <summary>
    /// Whether to draw individual frames while agent is playing.
    /// </summary>
    static bool DrawPlaying = true;

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
        Tensor state = new(StateFormat.Dimensions[1..].ToArray());

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

    public override double TestTrainingProgress(Model agent, int testEpisodes)
    {
        int totalLength = 0;
        DrawPlaying = false;
        for (int i = 0; i < testEpisodes; i++)
        {
            Play(agent);
            totalLength += SnakeLength;
        }
        DrawPlaying = true;

        double averageLength = (double)totalLength / testEpisodes;
        Console.WriteLine($"Agent reached an average length of {averageLength}");
        return averageLength;
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
        using var state = GetBoardState(); // get the current game board

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
                int action;
                using (var normState = GetNormalizedState())
                using (var wrapped = Tensor.WrapBatch(normState))
                using (var predicted = agent.Predict(wrapped))
                {
                    action = PickAgentAction(predicted);
                }
                SnakeHead.Move(MapAction(action));

                if (Collided()) break;

                if (AteApple()) stepsWithoutApple = 0;
                else stepsWithoutApple++;

                if (DrawPlaying)
                {
                    Console.Clear();
                    DrawSnake();
                }

                if (stepsWithoutApple >= MaxStepsWithoutApple) break;

                if (DrawPlaying) Thread.Sleep(FrameTime);
            }

            if (!DrawPlaying) break;

            Console.WriteLine("\nAgent collided or timed out!");

            playing = GetInput("Watch agent play again? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes];
        }
    }

    /// <summary>
    /// Draws the current game board to the console.
    /// </summary>
    void DrawSnake()
    {
        using var state = GetBoardState(); // get the current state of the board
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
