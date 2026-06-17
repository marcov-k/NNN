using NNN.Components.Autodiff;
using NNN.Components.Buffers;
using NNN.Components.Episodes;
using NNN.Components.Models;

namespace NNN.Components.Environments;

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
    /// Tests an agent's performance in the environment.
    /// </summary>
    /// <param name="agent">Agen to test.</param>
    /// <param name="testEpisodes">Number of episodes to test for.</param>
    public abstract void TestTrainingProgress(Model agent, int testEpisodes);

    /// <summary>
    /// Plays a demonstration of the model trained on the environment.
    /// </summary>
    public abstract void PlayDemo();
}

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