using NNN.Components.Autodiff;
using NNN.Components.Buffers;
using NNN.Components.Costs;
using NNN.Components.Environments;
using NNN.Components.Episodes;
using NNN.Components.Models;
using NNN.Components.Optimizers;
using NNN.Components.Utilities;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNN.Components.Trainers;

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
public class DQNTrainer(Model agent, Environments.Environment environment, Optimizer optimizer, Cost cost, int trainEvery = 4,
    double discount = 0.995, double exploration = 1.0, double explorationDecay = 0.99, double minExploration = 0.01, int replayBufferSize = 10000,
    int batchSize = 64, int agentBufferSize = 5, int opponentCopyRate = 100, int minRandomOpponentEpisodes = 200, double tau = 0.005,
    double maxGradNorm = 1.0, int minExperiences = 1000)
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
    readonly Environments.Environment Environment = environment;
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
    /// <summary>
    /// Precalculated 1 - tau value.
    /// </summary>
    readonly double OneMinusTau = 1.0 - tau;
    /// <summary>
    /// Preallocated vectorization of tau value.
    /// </summary>
    readonly Vector<double> TauVec = new(tau);
    /// <summary>
    /// Preallocated vectorization of 1 - tau value.
    /// </summary>
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
    /// <summary>
    /// Size of vectors in the current CPU architecture.
    /// </summary>
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
    public void Train(ref FIFOBuffer<Episode>? episodeBuffer, int episodes = 1000, int testEvery = 100, int testEpisodes = 5000)
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
        Stopwatch totalStopwatch = new();
        Stopwatch stopwatch = new();
        totalStopwatch.Start();
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
                episodeExperiences.Add(new(trueState, action, reward, Environment.GetState(), done));

                if ((step - 1) % TrainEvery == 0)
                {
                    TrainNetwork();
                    trainSteps++;
                }

                state = nextState;
            }

            episodeBuffer?.Add(new(episodeExperiences));

            if (ReplayBuffer.Count >= MinExperiences) Exploration = Math.Max(Exploration * ExplorationDecay, MinExploration); // exponentially decay exploration rate

            // Calculate diagnostic data
            var elapsed = stopwatch.Elapsed;
            avgElapsed += (elapsed - avgElapsed) / (e + 1);

            // Log episode diagnostics in the console
            if ((e + 1) % testEvery == 0 || (e + 1) == episodes)
            {
                var eta = avgElapsed * (episodes - e - 1);
                Console.WriteLine($"\n\nEpisodes completed: {e + 1}/{episodes}");
                Console.WriteLine($"Total reward for last episode: {totalReward:F2},");
                Console.WriteLine($"Average loss for last episode: {(totalLoss / trainSteps):F3}");
                Console.WriteLine($"Exploration rate: {Exploration:F2}");
                Console.WriteLine($"Experience count: {ReplayBuffer.Count}");
                if (Environment is ISelfPlay selfPlayEnv)
                {
                    Console.WriteLine($"Opponent agent for last episode: {(selfPlayEnv.OpponentIndex < selfPlayEnv.OpponentCount ?
                        $"{selfPlayEnv.OpponentIndex + 1}/{selfPlayEnv.OpponentCount}" : "Random")}");
                }
                Console.WriteLine($"Final state of last episode:");
                if (episodeBuffer is not null) Environment.Render(episodeBuffer[^1], step + 1);
                Console.WriteLine($"Ended on step: {step}");
                Console.WriteLine($"Episode duration: {MathUtils.RoundToMS(elapsed):g}");
                Console.WriteLine($"\nAverage time per episode: {MathUtils.RoundToMS(avgElapsed)}");
                Console.WriteLine($"Estimated time remaining: {MathUtils.RoundToMS(eta)}");
                Console.WriteLine($"\nEvaluating agent performance...");
                Environment.TestTrainingProgress(Agent, testEpisodes);
            }
            stopwatch.Restart();
        }

        totalStopwatch.Stop();
        Console.WriteLine($"Total Training Duration: {MathUtils.RoundToMS(totalStopwatch.Elapsed):g}");
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
