using NNN.Components.Autodiff;
using NNN.Components.Episodes;
using NNN.Components.Models;

namespace NNN.Components.Environments;

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

    // TODO: Impelement training progress test
    public override void TestTrainingProgress(Model agent, int testEpisodes)
    {
        throw new NotImplementedException();
    }

    public override void Render(Episode episode, int step)
    {
        // Extract the state at the given step from the episode
        step = Math.Clamp(step, 0, episode.Experiences.Count);
        var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
        var state = exp.NextState;
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
