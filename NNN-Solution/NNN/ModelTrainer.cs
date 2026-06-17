using NNN.Components.Activations;
using NNN.Components.Buffers;
using NNN.Components.Costs;
using NNN.Components.Environments;
using NNN.Components.Episodes;
using NNN.Components.Models;
using NNN.Components.Models.Layers;
using NNN.Components.Optimizers;
using NNN.Components.Trainers;
using NNN.Components.Utilities;
using NNN.Components.Utilities.SaveSystem;
using static NNN.Components.Utilities.UIUtils;

bool demoMode = false;

Model model;
NNN.Components.Environments.Environment env = new TicTacToe();
double exploration = 1.0;
double explorationDecay = 0.9995;
double minExploration = 0.01;
int trainEvery = 1;
double discount = 0.99;
Optimizer optimizer = new Adam(0.001);
Cost cost = new Huber();
int replayBufferSize = 10000;
int batchSize = 128;
int agentBufferSize = 2;
int opponentCopyRate = 600;
int minRandomOpponentEpisodes = 600;
double tau = 0.01;
double maxGradNorm = 1.0;
int minExperiences = 2000;
int episodeMemorySize = 100;
int testEpisodes = 5000;
DQNTrainer dqnTrainer;
FIFOBuffer<Episode> episodeBuffer = new(episodeMemorySize);

if (demoMode) DemoHandler.RunDemo();
else InteractionLoop();

#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning disable CS8602 // Dereference of a possibly null reference.
// Primary loop for model training UI - entry point
void InteractionLoop()
{
    Console.WriteLine("Welcome to the DQN Training Terminal (Enter Q to quit)");

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
            new Dense(256, new LeakyReLU()),
            new Dense(256, new LeakyReLU()),
            new Dense(128, new LeakyReLU()),
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

    TrainingLoop();

    if (GetInput("Save model to a file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        SaveLoop(model);
    }

    Console.WriteLine("\nPress any key to quit...");
    Console.ReadKey();
    System.Environment.Exit(0);
}

// UI loop for training models for a specified number of episodes
void TrainingLoop()
{
    // Train agent until user indicates to stop
    while (true)
    {
        if (GetInput("Run DQN Training episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            // Train agent for a given number of episodes
            int episodes = GetInteger("Enter number of episodes to train");
            int testEvery = GetInteger("Enter episodes per training progress test");
            Console.WriteLine($"\n\nTraining for {episodes} episodes...");
            dqnTrainer.Train(ref episodeBuffer, episodes, testEvery, testEpisodes);

            if (env is TicTacToe ticTacToe && GetInput("Play against model? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
            {
                ticTacToe.Play(model);
            }

            if (env is Snake snake && GetInput("Watch agent play? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
            {
                snake.Play(model);
            }

            ViewEpisodes();
        }
        else break;
    }
}

// UI loop for viewing and navigating through past training episodes
void ViewEpisodes()
{
    if (GetInput("Replay past episodes? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
    {
        // Allow user to view past episodes until user indicates to stop
        while (true)
        {
            Console.WriteLine();
            int episode = GetEpisodeSelection(episodeBuffer!);

            int step = 0;
            bool viewingEpisode = true;
            while (viewingEpisode)
            {
                // Render current selected step of the episode
                Console.Clear();
                env.Render(episodeBuffer[episode], step);

                // Navigate through episode based on user input
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

#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
#pragma warning restore CS8602 // Dereference of a possibly null reference.

/// <summary>
/// Key mappings for user navigation during training episode review.
/// </summary>
enum EpisodeNavigation
{
    Previous = ConsoleKey.LeftArrow,
    Next = ConsoleKey.RightArrow,
    Exit = ConsoleKey.Escape,
    Quit = ConsoleKey.Q
}