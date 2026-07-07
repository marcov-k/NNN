using NNNCSharp.Components.Episodes;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities.SaveSystem;
using static NNNCSharp.Components.Utilities.UIUtils;

namespace NNNCSharp.Components.Environments;

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
    /// Time per frame when showing the agent navigating the grid.
    /// </summary>
    const int FrameTime = 150;
    /// <summary>
    /// Whether to draw each movement step when agent is navigating.
    /// </summary>
    static bool DrawRunning = false;
    /// <summary>
    /// Action index mapping.
    /// </summary>
    enum Action { Left, Right, Up, Down }

    // Demo
    /// <summary>
    /// Name of the file containing the demonstration agent for the environment.
    /// </summary>
    const string DemoFileName = "2dmovedemo";

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

    public override int PickAgentAction(Tensor qValues, Tensor? state = null)
    {
        int action = Tensor.ArgMax(qValues); // no invalid actions - return highest Q-Value
        GC.KeepAlive(qValues);
        return action;
    }

    public override int PickRandomAction() => Random.Next(ActionCount); // no invalid actions - return random action index

    public override bool ValidAction(int action, Tensor? state) => true; // no invalid actions

    public override (double reward, Tensor nextState, bool done) Step(int action, int steps)
    {
        // Calculate distance to target before moving
        double xDiff = State[2] - State[0];
        double yDiff = State[3] - State[1];
        double prevDist = Math.Sqrt(xDiff * xDiff + yDiff * yDiff);

        // Move the agent's position based on action mapping
        Move(action);

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
        double reward = 1.0 * deltaDist; // shaped reward based on change in distance to target
        reward += reachedTarget ? 10.0 : 0.0; // reward for reaching the target
        reward -= outOfBounds ? 10.0 : 0.0; // penalty for going out of bounds
        reward -= outOfSteps ? 2.0 : 0.0; // penalty for exceeding step limit

        return (reward, GetNormalizedState(), done);
    }

    public override double TestTrainingProgress(Model agent, int testEpisodes)
    {
        int successes = 0;
        for (int i = 0; i < testEpisodes; i++)
        {
            if (Run(agent)) successes++;
        }

        double successPercent = ((double)successes / testEpisodes) * 100.0;
        Console.WriteLine($"Agent successfully reached target in {successPercent:F2}% of test episodes");
        return successPercent;
    }

    public override void Render(Episode episode, int step)
    {
        // Extract the state at the given step from the episode
        step = Math.Clamp(step, 0, episode.Experiences.Count);
        var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
        var state = exp.NextState;
        (int action, double reward) = step > 0 ? (episode.Experiences[step - 1].Action, episode.Experiences[step - 1].Reward) : (-1, 0);

        DrawState(state);

        Console.Write($"Step: {step}, Action: {(Enum.IsDefined(typeof(Action), action) ? ((Action)action).ToString() : "None")}, Reward: {reward:F3}");
    }

    public override void PlayDemo()
    {
        ShowDemoInstructions();
        var agent = Saver.LoadModel(DemoFileName);
        DrawRunning = true;
        while (true)
        {
            Run(agent);
            if (GetInput("Watch agent navigate again? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) break;
        }
        DrawRunning = false;
    }

    /// <summary>
    /// Draws the given state to the console.
    /// </summary>
    /// <param name="state">State to draw.</param>
    void DrawState(Tensor state)
    {
        Console.WriteLine("Key: A - Agent, T - Target\n");

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
    }

    /// <summary>
    /// Runs a single grid navigation episode.
    /// </summary>
    /// <param name="agent">Agent to use for episode.</param>
    /// <returns>Whether the agent successfully reached the target.</returns>
    public bool Run(Model agent)
    {
        Reset();

        int steps = 0;
        while (steps < MaxSteps)
        {
            int action = GetAgentAction(agent);
            Move(action);

            if (DrawRunning)
            {
                Console.Clear();
                DrawState(State);
                Thread.Sleep(FrameTime);
            }

            if (State[0] == State[2] && State[1] == State[3]) return true;
            steps++;
        }
        return false;
    }

    /// <summary>
    /// Selects the next action to take using the given agent.
    /// </summary>
    /// <param name="agent">Agent to select action with.</param>
    /// <returns>Selected action index.</returns>
    int GetAgentAction(Model agent)
    {
        using var normState = GetNormalizedState();
        using var wrapped = Tensor.WrapBatch(normState);
        using var predicted = agent.Predict(wrapped);
        return PickAgentAction(predicted);
    }

    /// <summary>
    /// Moves the agent's position based on the given action.
    /// </summary>
    /// <param name="action">Action taken by the agent.</param>
    /// <exception cref="ArgumentException">Invalid action index.</exception>
    void Move(int action)
    {
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
    }

    static void ShowDemoInstructions()
    {
        Console.WriteLine("Welcome to the 2D movement agent demonstration.");
        Console.WriteLine("The agent contains a total of 132 neurons.");
        Console.WriteLine("These are arranged in two layers of 64 neurons each, and an output layer of 4 neurons - one for each cardinal direction.");
        Console.WriteLine("This agent was trained over the course of 15,000 episodes using randomly generated 21x21 grids.");
        Console.WriteLine("By the end of this training, it was achieving a 100% success rate in test sessions, each of which used 10,000 episodes.");
        Console.WriteLine("For this demonstration, a 21x21 grid is used, with the agent's starting and target positions being generated randomly.");
        Console.WriteLine("Press any key to continue...");
        Console.ReadKey();
    }
}
