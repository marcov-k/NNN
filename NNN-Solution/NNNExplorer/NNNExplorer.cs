using NNNCSharp.Components.Utilities.SaveSystem;
using static NNNCSharp.Components.Utilities.UIUtils;

namespace NNNExplorer;

/// <summary>
/// Neural Network Notions .nnn format file viewing program.
/// </summary>
public class NNNExplorer
{
    public static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            ExplorerLoop();
        }
        else
        {
            DisplayFile(args[0]);
        }

        Console.WriteLine("\nPress any key to close...");
        Console.ReadKey();
        Environment.Exit(0);
    }

    /// <summary>
    /// Loop for user interaction when opened without a file path.
    /// </summary>
    static void ExplorerLoop()
    {
        while (true)
        {
            string fileName = GetInput("Enter name of .nnn file to view (without extension)");
            string filePath = Saver.GetFullPath(fileName);
            Console.WriteLine("\n");
            DisplayFile(filePath);

            if (GetInput("View another file? y/n", [userInputs[UserInput.Yes], userInputs[UserInput.No]]) == userInputs[UserInput.No]) break;

            Console.WriteLine();
        }
    }

    /// <summary>
    /// Displays the .nnn file at the given path to the console.
    /// </summary>
    /// <param name="filePath">Path of the file to display.</param>
    static void DisplayFile(string filePath)
    {
        Console.WriteLine();

        Console.WriteLine($"Reading model stored in {Path.GetFileName(filePath)}\n");

        if (!File.Exists(filePath))
        {
            Console.WriteLine("File not found.\n");
            return;
        }

        using FileStream stream = new(filePath, FileMode.Open, FileAccess.Read);
        {
            int magic = FileUtils.ReadInt32(stream);
            if (magic != Saver.MagicNumber)
            {
                Console.WriteLine("Incorrect file type.\n");
                return;
            }

            string desc = FileUtils.ReadString(stream);
            ulong paramCount = FileUtils.ReadUInt64(stream);
            int layerCount = FileUtils.ReadInt32(stream);
            var layerData = new string[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layerData[i] = FileUtils.PrintLayer(stream);
            }

            Console.WriteLine($"Description: {desc}");
            Console.WriteLine($"Total Parameter Count: {paramCount}");
            Console.WriteLine($"Layers: {layerCount}");
            for (int i = 0; i < layerCount; i++)
            {
                Console.WriteLine($"\nLayer {i + 1}:\n{layerData[i]}\n");
            }
            Console.WriteLine();
        }
    }
}