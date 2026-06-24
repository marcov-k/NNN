using NNN.Components.Autodiff;
using NNN.Components.Environments;
using NNN.Components.Utilities.DataLoaders;
using NNN.Components.Utilities.SaveSystem;
using static NNN.Components.Utilities.UIUtils;

namespace NNNDemo;

/// <summary>
/// Neural Network Notions demonstration program.
/// </summary>
public class NNNDemo
{
    /// <summary>
    /// Array of all environments with trained and implemented demonstrations.
    /// </summary>
    static readonly NNN.Components.Environments.Environment[] DQNDemoEnvs = [new TicTacToe(), new Snake()];
    /// <summary>
    /// Array of all standard supervised training demo functions.
    /// </summary>
    static readonly StandardDemo[] StandardDemos = [new("MNIST", "mnistdemo", RunMNISTDemo)];

    /// <summary>
    /// Record storing the name and function of a standard supervised training demo.
    /// </summary>
    /// <param name="DemoName">Display name of the demo.</param>
    /// <param name="DemoFunction">Function of the demo.</param>
    record StandardDemo(string DemoName, string FileName, Action<string> DemoFunction);

    public static void Main(string[] args)
    {
        RunDemo();
    }

    /// <summary>
    /// Runs the demonstration user interaction loop.
    /// </summary>
    static void RunDemo()
    {
        Console.WriteLine("Welcome to the Neural Network Nonsense library demonstration.");
        Console.WriteLine("Enter Q at any time to close the demonstration.");

        // Main interaction loop
        while (true)
        {
            if (GetIntegerInRange("Select demo version:\n1 - Standard\n2 - DQN", 1, 2) == 1)
            {
                RunStandardDemo();
            }
            else
            {
                RunDQNDemo();
            }

            if (GetInput("Continue viewing demos? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No])
            {
                break;
            }

            Console.Clear();
        }

        Console.WriteLine("\nPress any key to close...");
        Console.ReadKey();
        System.Environment.Exit(0);
    }

    /// <summary>
    /// Runs the DQN training demonstration user interaction loop.
    /// </summary>
    static void RunDQNDemo()
    {
        bool done = false;
        while (!done)
        {
            // Get environment index from the user
            string prompt = "Please select which DQN environment you would like to see a demo of:";
            for (int i = 0; i < DQNDemoEnvs.Length; i++)
            {
                prompt += $"\n{i + 1} - {DQNDemoEnvs[i].GetType().Name}";
            }

            int envIndex = GetIntegerInRange(prompt, 1, DQNDemoEnvs.Length);

            Console.WriteLine("\n");
            DQNDemoEnvs[envIndex - 1].PlayDemo();

            Console.Clear();
            if (GetInput("Continue viewing DQN demos? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) done = true;
        }
    }

    /// <summary>
    /// Runs the standard supervised training demonstration user interaction loop.
    /// </summary>
    static void RunStandardDemo()
    {
        bool done = false;
        while (!done)
        {
            // Get demo index from user
            string prompt = "Please select which model you would like to see a demo of:";
            for (int i = 0; i < StandardDemos.Length; i++)
            {
                prompt += $"\n{i + 1} - {StandardDemos[i].DemoName}";
            }

            int demoIndex = GetIntegerInRange(prompt, 1, StandardDemos.Length);

            Console.WriteLine("\n");
            StandardDemos[demoIndex - 1].DemoFunction(StandardDemos[demoIndex - 1].FileName);

            Console.Clear();
            if (GetInput("Continue viewing standard demos? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) done = true;
        }
    }

    /// <summary>
    /// Loads and runs the MNIST training demonstration.
    /// </summary>
    /// <param name="fileName">Name of the file to load the MNIST demo model from.</param>
    static void RunMNISTDemo(string fileName)
    {
        Console.WriteLine("\nLoading MNIST test dataset...");
        var (images, labels) = MNISTLoader.GetTestData();
        var wrappedImages = new Tensor[images.Length];
        for (int i = 0; i < images.Length; i++)
        {
            wrappedImages[i] = Tensor.WrapBatch(images[i]);
        }
        Console.WriteLine("Loaded MNIST test dataset");

        Console.WriteLine("\nLoading demo model...");
        var model = Saver.LoadModel(fileName);
        Console.WriteLine("Loaded demo model");

        bool done = false;
        while (!done)
        {
            int index = Random.Shared.Next(wrappedImages.Length);
            var image = images[index];
            var wrappedImage = wrappedImages[index];
            var label = labels[index];

            int predictLabel = Tensor.ArgMax(model.Predict(wrappedImage));
            Console.WriteLine($"\n\nImage index in dataset: {index}\n");
            DrawMNISTImage(image, label, 0.5);
            Console.WriteLine($"Model prediction: {predictLabel}");

            if (GetInput("View another image? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) done = true;
        }
    }
}