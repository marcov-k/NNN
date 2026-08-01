using NNNCSharp.Components.Autodiff;
using System;

namespace NNNCSharp.Components.Episodes
{
    /// <summary>
    /// Record of a single DQN training experience.
    /// </summary>
    public record Experience : IDisposable
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
        public float Reward { get; init; }
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
        public float Priority { get; set; }

        /// <summary>
        /// Creates a new experience instance.
        /// </summary>
        /// <param name="state">Initial environment state.</param>
        /// <param name="action">Action selected by the agent.</param>
        /// <param name="reward">Reward of the selected action.</param>
        /// <param name="nextState">Environment state following the selected action.</param>
        /// <param name="done">Whether the episode terminated.</param>
        /// <param name="priority">Replay priority of the experience - temporal difference error.</param>
        public Experience(Tensor state, int action, float reward, Tensor nextState, bool done, float priority = 1.0f)
        {
            State = state.Copy();
            Action = action;
            Reward = reward;
            NextState = nextState.Copy();
            Done = done;
            Priority = Math.Max(priority, 1e-8f); // ensure non-zero priority
        }

        /// <summary>
        /// Releases all native C++ memory used by the instance.
        /// </summary>
        public void Dispose()
        {
            State.Dispose();
            NextState.Dispose();
        }
    }
}
