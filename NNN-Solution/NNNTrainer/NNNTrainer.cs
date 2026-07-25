using NNNCSharp.Components.Activations;
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Buffers;
using NNNCSharp.Components.Costs;
using NNNCSharp.Components.DQNEnvironments;
using NNNCSharp.Components.Episodes;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Models.Layers;
using NNNCSharp.Components.Optimizers;
using NNNCSharp.Components.Trainers;
using NNNCSharp.Components.Utilities;
using NNNCSharp.Components.Utilities.DataLoaders;
using NNNCSharp.Components.Utilities.SaveSystem;
using static NNNCSharp.Components.Utilities.UIUtils;

namespace NNNTrainer;

/// <summary>
/// Neural Network Notions model training program.
/// </summary>
public class NNNTrainer
{
    public static void Main(string[] args)
    {
        NNNLog.Output = Console.Write;
        if (GetIntegerInRange("Select training mode:\n1 - Standard\n2 - DQN", 1, 2) == 1)
        {
            StandardTraining();
        }
        else
        {
            DQNTraining();
        }

        NNNLog.WriteLine("\nPress any key to quit...");
        Console.ReadKey();
        System.Environment.Exit(0);
    }

    /// <summary>
    /// Sets up training hyperparameters and prompts the user through DQN training.
    /// </summary>
    static void DQNTraining()
    {
        Model model;
        DQNEnvironment env = new TicTacToe();
        float exploration = 1.0f;
        float explorationDecay = 0.999f;
        float minExploration = 0.01f;
        int trainEvery = 3;
        float discount = 0.95f;
        Optimizer optimizer = new Adam(0.001f);
        Cost cost = new Huber();
        int replayBufferSize = 20000;
        int batchSize = 128;
        int agentBufferSize = 2;
        int opponentCopyRate = 600;
        int minRandomOpponentEpisodes = 600;
        float tau = 0.01f;
        float maxGradNorm = 1.0f;
        int minExperiences = 2000;
        int episodeMemorySize = 100;
        int testEpisodes = 5000;
        DQNTrainer dqnTrainer;
        FIFOBuffer<Episode> episodeBuffer = new(episodeMemorySize);

        Console.Clear();
        NNNLog.WriteLine("Welcome to the DQN Training Terminal (Enter Q to quit)");

        float activTau = 0.05f;
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
                new Dense(128, new LeakyReLU(activTau)),
                new Dense(128, new LeakyReLU(activTau)),
                new Dense(64, new LeakyReLU(activTau)),
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

        DQNTrainingLoop(dqnTrainer, env, model, ref episodeBuffer, testEpisodes);
        model = dqnTrainer.Agent;

        if (GetInput("Save model to a file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            SaveLoop(model);
        }
    }

    /// <summary>
    /// Sets up training hyperparameters and prompts the user through standard supervised training.
    /// </summary>
    static void StandardTraining()
    {
        Console.Clear();
        NNNLog.WriteLine("Welcome to the Supervised Training Terminal (Enter Q to quit)");

        NNNLog.WriteLine("\nLoading MNIST dataset...");
        var (trainImages, trainLabels) = MNISTLoader.GetTrainingData();
        var (testImages, testLabels) = MNISTLoader.GetTestData();
        NNNLog.WriteLine("MNIST dataset loaded");

        float tau = 0.05f;
        float convDropout = 0.15f;
        float denseDropout = 0.5f;
        Model model;
        if (GetInput("Load model from file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            // Load model from file
            string fileName = GetFileName();
            model = Saver.LoadModel(fileName);
        }
        else
        {
            model = new([
                new Conv(8, [5, 5], new LeakyReLU(tau)),
            new Conv(16, [5, 5], new LeakyReLU(tau), convDropout),
            new Dense(128, new LeakyReLU(tau), true, denseDropout),
            new Dense(10, new Linear())
            ], new([1, 28, 28, 1]));
        }

        Optimizer optimizer = new Adam(0.001f, weightDecay: 0.01f);
        float maxGradNorm = 0.5f;
        Cost cost = new SoftmaxCrossEntropy();
        Trainer trainer = new(model, optimizer, cost, maxGradNorm);
        float minLRFraction = 0.5f;

        var wrappedImages = new Tensor[testImages.Length];
        for (int i = 0; i < testImages.Length; i++)
        {
            wrappedImages[i] = Tensor.WrapBatch(testImages[i]);
        }

        Func<Model, int, bool> testFunc = (model, i) =>
        {
            using var predicts = model.Predict(wrappedImages[i]);
            return Tensor.ArgMax(predicts) == Tensor.ArgMax(testLabels[i]);
        };

        BatchBuffer batchBuffer = new(trainImages, trainLabels);
        int batchSize = 128;
        int testLength = testLabels.Length;

        StandardTrainingLoop(trainer, batchBuffer, batchSize, testFunc, true, minLRFraction, testLength);
        model = trainer.Model;

        if (GetInput("Save model to a file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            SaveLoop(model);
        }
    }

    /// <summary>
    /// Repeatedly prompts the user to train a model for a given number of DQN episodes.
    /// </summary>
    /// <param name="dqnTrainer">DQN trainer to use.</param>
    /// <param name="env">DQN environment to train in.</param>
    /// <param name="model">Model to train.</param>
    /// <param name="episodeBuffer">Buffer to store past training episodes in.</param>
    /// <param name="testEpisodes">Number of episodes to run per performance test.</param>
    static void DQNTrainingLoop(DQNTrainer dqnTrainer, DQNEnvironment env, Model model,
        ref FIFOBuffer<Episode> episodeBuffer, int testEpisodes)
    {
        // Train agent until user indicates to stop
        while (true)
        {
            if (GetInput("Run DQN Training episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
            {
                // Train agent for a given number of episodes
                int episodes = GetInteger("Enter number of episodes to train");
                int testEvery = GetInteger("Enter episodes per training progress test");
                NNNLog.WriteLine($"\n\nTraining for {episodes} episodes...");
                dqnTrainer.Train(ref episodeBuffer!, episodes, testEvery, testEpisodes);
                model = dqnTrainer.Agent;

                if (env is TicTacToe ticTacToe && GetInput("Play against model? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
                {
                    ticTacToe.Play(model);
                }

                if (env is Snake snake && GetInput("Watch agent play? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
                {
                    snake.Play(model);
                }

                ViewEpisodes(env, ref episodeBuffer);
            }
            else break;
        }
    }

    /// <summary>
    /// Repeatedly prompts the user to train a model for a given number of supervised epochs.
    /// </summary>
    /// <param name="trainer">Supervised trainer to use.</param>
    /// <param name="batchBuffer">Batch buffer storing training data.</param>
    /// <param name="batchSize">Number of training inputs to include per batch.</param>
    /// <param name="testFunc">Function to use to evaluate model performance.</param>
    /// <param name="testLength">Number of times to run the test function per performance test.</param>
    static void StandardTrainingLoop(Trainer trainer, BatchBuffer batchBuffer, int batchSize,
        Func<Model, int, bool> testFunc, bool decayLR, float minLRFraction, int testLength)
    {
        // Train model until user indicates to stop
        while (true)
        {
            if (GetInput("Run supervised training epochs? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
            {
                int epochs = GetInteger("Enter number of epochs to train");
                int testEvery = GetInteger("Enter epochs per training progress test");
                NNNLog.WriteLine($"\n\nTraining for {epochs} epochs...");
                trainer.Train(batchBuffer, batchSize, epochs, batchAllInputs: true, testFunc, decayLR,
                    minLRFraction, testEvery: testEvery, testLength: testLength);
            }
            else break;
        }
    }
}