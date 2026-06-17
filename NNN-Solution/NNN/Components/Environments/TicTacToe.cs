using NNN.Components.Autodiff;
using NNN.Components.Episodes;
using NNN.Components.Models;
using NNN.Components.Utilities.SaveSystem;
using static NNN.Components.Utilities.UIUtils;

namespace NNN.Components.Environments;

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
    const double WinRewardBase = 2.0;
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

    public override void Render(Episode episode, int step)
    {
        // Extract the state at the given step from the episode
        step = Math.Clamp(step, 0, episode.Experiences.Count);
        var exp = step == episode.Experiences.Count ? episode.Experiences[step - 1] : episode.Experiences[step];
        var state = exp.NextState;
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
