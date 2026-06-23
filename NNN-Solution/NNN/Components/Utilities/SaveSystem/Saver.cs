using NNN.Components.Models;

namespace NNN.Components.Utilities.SaveSystem;

/// <summary>
/// Static class for handling saving and loading of trained neural networks.
/// </summary>
public static class Saver
{
    /// <summary>
    /// Directory to save to and load from.
    /// </summary>
    const string DirectoryName = "Models";
    /// <summary>
    /// Full directory path to save to and load from.
    /// </summary>
    static string DirectoryPath = string.Empty;
    /// <summary>
    /// File extension for neural network save files.
    /// </summary>
    const string Extension = ".nnn";
    /// <summary>
    /// Magic number for .nnn files.
    /// </summary>
    public const int MagicNumber = 776883790; // spells ".NNN"

    /// <summary>
    /// Saves the given model to a file with the given name.
    /// </summary>
    /// <param name="model">Model to save.</param>
    /// <param name="fileName">Name of file to save to.</param>
    /// <param name="desc">Short description to include in the file.</param>
    public static void SaveModel(Model model, string fileName, string desc = "")
    {
        string filePath = GetFullPath(fileName);

        using FileStream stream = new(filePath, FileMode.Create, FileAccess.Write);
        {
            FileUtils.WriteInt32(stream, MagicNumber);
            FileUtils.WriteString(stream, desc);
            FileUtils.WriteUInt64(stream, model.GetTotalParameterSize());
            FileUtils.WriteModel(stream, model);
            stream.Flush();
        }
    }

    /// <summary>
    /// Loads the model stored in the given file.
    /// </summary>
    /// <param name="fileName">Name of file to load from.</param>
    /// <returns>Model loaded from the given file.</returns>
    public static Model LoadModel(string fileName)
    {
        string filePath = GetFullPath(fileName);
        if (!File.Exists(filePath)) throw new ArgumentException($"File with name {fileName} not found.");

        Model model;
        using FileStream stream = new(filePath, FileMode.Open, FileAccess.Read);
        {
            int magic = FileUtils.ReadInt32(stream);
            if (magic != MagicNumber) throw new Exception($"Invalid magic number: found {magic} instead of {MagicNumber}");

            // Skip the file's description
            int descLength = FileUtils.ReadInt32(stream);
            stream.Position += descLength + sizeof(ulong); // skip description + total parameter count (ulong)

            model = FileUtils.ReadModel(stream);
        }

        return model;
    }

    /// <summary>
    /// Checks whether a neural network file with the given name exists.
    /// </summary>
    /// <param name="fileName">Name of file to search for.</param>
    /// <returns>Whether a neural network file with the given name exists.</returns>
    public static bool FileExists(string fileName)
    {
        string filePath = GetFullPath(fileName);

        if (File.Exists(filePath)) return true;
        else return false;
    }

    /// <summary>
    /// Gets the full file path to a .nnn file with the given name in the Models directory.
    /// </summary>
    /// <param name="fileName">Name of the file to get path to.</param>
    /// <returns>Full path to the given file.</returns>
    public static string GetFullPath(string fileName)
    {
        InitDirectory();

        return Path.Combine(DirectoryPath, fileName + Extension);
    }

    /// <summary>
    /// Initializes the directory and directory path if not yet initialized.
    /// </summary>
    static void InitDirectory()
    {
        if (string.IsNullOrEmpty(DirectoryPath))
        {
            string? exePath = Environment.ProcessPath;
            string? exeDirPath = Path.GetDirectoryName(exePath);
            if (!string.IsNullOrEmpty(exeDirPath))
            {
                DirectoryPath = Path.Combine(exeDirPath, DirectoryName);
                if (!Directory.Exists(DirectoryPath)) Directory.CreateDirectory(DirectoryPath);
            }
        }
    }
}
