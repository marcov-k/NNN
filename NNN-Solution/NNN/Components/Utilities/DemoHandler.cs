using NNN.Components.Environments;
using static NNN.Components.Utilities.UIUtils;

namespace NNN.Components.Utilities;

/// <summary>
/// Static class containing demonstration functionality.
/// </summary>
public static class DemoHandler
{
    /// <summary>
    /// Array of all environments with trained and implemented demonstrations.
    /// </summary>
    static readonly Environments.Environment[] DemoEnvs = [new TicTacToe(), new Snake()];

    /// <summary>
    /// Runs the demonstration user interaction loop.
    /// </summary>
    public static void RunDemo()
    {
        Console.WriteLine("Welcome to the Neural Network Nonsense library demonstration.");
        Console.WriteLine("Enter Q at any time to close the demonstration.");

        // Main interaction loop
        bool done = false;
        while (!done)
        {
            // Get demo index from the user
            int envIndex = -1;
            bool validIndex = false;
            while (!validIndex)
            {
                string prompt = "Please select which environment you would like to see a demo of:";
                for (int i = 0; i < DemoEnvs.Length; i++)
                {
                    prompt += $"\n{i + 1} - {DemoEnvs[i].GetType().Name}";
                }

                envIndex = GetInteger(prompt);

                if (envIndex > 0 && envIndex <= DemoEnvs.Length) validIndex = true;
                else Console.WriteLine("Invalid selection...");
            }

            Console.WriteLine("\n");
            DemoEnvs[envIndex - 1].PlayDemo();

            Console.Clear();
            if (GetInput("Continue viewing demos? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) done = true;
        }

        Console.WriteLine("Press any key to close...");
        Console.ReadKey();
    }
}
