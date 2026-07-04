using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Episodes;

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
