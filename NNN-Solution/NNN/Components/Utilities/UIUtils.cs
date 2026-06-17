using NNN.Components.Buffers;
using NNN.Components.Episodes;
using NNN.Components.Models;
using NNN.Components.Utilities.SaveSystem;

namespace NNN.Components.Utilities;

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
                    Saver.SaveModel(model, fileName);
                    Console.WriteLine("\nModel saved");
                    break;
                }
            }
            else
            {
                Saver.SaveModel(model, fileName);
                Console.WriteLine("\nModel saved");
                break;
            }
        }
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

            if (input == userInputs[UserInput.Quit]) System.Environment.Exit(0); // terminate the program if the user chooses to quit
            else if (options.Count == 0 || options.Contains(input)) return input; // return input if valid or no valid inputs given
        }
    }
}
