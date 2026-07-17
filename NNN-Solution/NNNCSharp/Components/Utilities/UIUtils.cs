using NNNCSharp.Components.Buffers;
using NNNCSharp.Components.DQNEnvironments;
using NNNCSharp.Components.Episodes;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Utilities.SaveSystem;
using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Utilities;

/// <summary>
/// Static class containing various UI utility functions.
/// </summary>
public static class UIUtils
{
    /// <summary>
    /// Standard user input mappings.
    /// </summary>
    public static readonly Dictionary<UserInput, string> userInputs = new() { { UserInput.Yes, "y" },
            { UserInput.No, "n" }, { UserInput.Quit, "q" } };

    /// <summary>
    /// Standard user inputs.
    /// </summary>
    public enum UserInput { Yes, No, Quit }

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

    /// <summary>
    /// Allows the user to view past DQN training episodes.
    /// </summary>
    /// <param name="env">DQN environment episodes took place in.</param>
    /// <param name="episodeBuffer">Buffer of stored episodes for viewing.</param>
    public static void ViewEpisodes(DQNEnvironment env, ref FIFOBuffer<Episode> episodeBuffer)
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
                    switch (input)
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
                            Environment.Exit(0);
                            break;
                    }
                }

                if (GetInput("View another episode? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) break;
            }
        }
    }

    /// <summary>
    /// Saves the given model to the file entered by the user.
    /// </summary>
    /// <param name="model">Model to be saved.</param>
    public static void SaveLoop(Model model)
    {
        string fileName;

        while (true)
        {
            fileName = GetInput("Enter file name");
            if (Saver.FileExists(fileName))
            {
                if (GetInput($"File with name \"{fileName}\" already exists. Overwrite existing file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]])
                    == userInputs[UserInput.Yes])
                {
                    Save(model, fileName);
                    break;
                }
            }
            else
            {
                Save(model, fileName);
                break;
            }
        }
    }

    /// <summary>
    /// Finalizes model save information.
    /// </summary>
    /// <param name="model">Model to save.</param>
    /// <param name="fileName">File to save to.</param>
    static void Save(Model model, string fileName)
    {
        string desc = string.Empty;
        if (GetInput("Include a short description of the model? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.Yes])
        {
            desc = GetInput("Enter description:");
        }

        Saver.SaveModel(model, fileName, desc);
        Console.WriteLine("\nModel saved");
    }

    /// <summary>
    /// Prompts the user to enter a file name.
    /// </summary>
    /// <returns>Name of an existing file entered by the user.</returns>
    public static string GetFileName()
    {
        string input;
        while (true)
        {
            input = GetInput("Enter file name");
            if (Saver.FileExists(input)) return input;
            else Console.WriteLine("\nFile not found");
        }
    }

    /// <summary>
    /// Prompts the user to enter an integer input in the given range.
    /// </summary>
    /// <param name="prompt">Prompt to display to the user.</param>
    /// <param name="min">Minimum value of the integer (inclusive).</param>
    /// <param name="max">Maximum value of the integer (inclusive).</param>
    /// <returns>Integer within the given range entered by the user.</returns>
    public static int GetIntegerInRange(string prompt, int min, int max)
    {
        while (true)
        {
            if (int.TryParse(GetInput(prompt), out int integer) && integer >= min && integer <= max) return integer;
            else Console.WriteLine("\nNot a valid number");
        }
    }

    /// <summary>
    /// Prompts the user to enter an integer input.
    /// </summary>
    /// <param name="prompt">Prompt to display to the user.</param>
    /// <returns>Valid integer entered by the user.</returns>
    public static int GetInteger(string prompt)
    {
        while (true)
        {
            if (int.TryParse(GetInput(prompt), out int integer)) return integer;
            else Console.WriteLine("\nNot a valid number");
        }
    }

    /// <summary>
    /// Gets a valid episode index from the user.
    /// </summary>
    /// <param name="episodeBuffer">Episode buffer which is being accessed.</param>
    /// <returns>Valid episode index entered by the user.</returns>
    public static int GetEpisodeSelection(FIFOBuffer<Episode> episodeBuffer)
    {
        while (true)
        {
            int index = GetInteger($"Enter episode number ({episodeBuffer.Count} episodes cached)");
            if (index > 0 && index <= episodeBuffer.Count) return index - 1;
            else Console.WriteLine("Invalid episode number");
        }
    }

    /// <summary>
    /// Prompts the user for an input.
    /// </summary>
    /// <param name="prompt">Prompt to display to the user.</param>
    /// <param name="options">Optional list of valid user inputs.</param>
    /// <returns>Valid input entered by the user.</returns>
    public static string GetInput(string prompt, List<string>? options = null)
    {
        options ??= [];
        for (int i = 0; i < options.Count; i++)
        {
            options[i] = options[i].ToLowerInvariant();
        }

        // Prompt the user until a valid input is given or the quit input is given
        string input;
        while (true)
        {
            Console.WriteLine($"\n{prompt}");
            input = Console.ReadLine()?.ToLowerInvariant() ?? string.Empty;

            if (input == userInputs[UserInput.Quit]) Environment.Exit(0); // terminate the program if the user chooses to quit
            else if (options.Count == 0 || options.Contains(input)) return input; // return input if valid or no valid inputs given
        }
    }

    /// <summary>
    /// Draws a MNIST dataset image in the console.
    /// </summary>
    /// <param name="image">Image data to draw.</param>
    /// <param name="label">Label of the image.</param>
    /// <param name="renderThreshold">Opacity threshold to draw a pixel at.</param>
    public static void DrawMNISTImage(Tensor image, Tensor label, double renderThreshold)
    {
        for (int row = 0; row < image.Dimensions[0]; row++)
        {
            for (int col = 0; col < image.Dimensions[1]; col++)
            {
                Console.Write(image[row, col] > renderThreshold ? "@" : " ");
            }

            Console.WriteLine();
        }
        Console.WriteLine($"Label: {Tensor.ArgMax(label)}");
    }
}
