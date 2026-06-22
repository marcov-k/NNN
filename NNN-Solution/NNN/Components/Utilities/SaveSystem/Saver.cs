using NNN.Components.Models;
using NNN.Components.Models.Layers;
using System.Text.Json;

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

#pragma warning disable CS8604 // Possible null reference argument.
    /// <summary>
    /// Saves the given model to a file with the given name.
    /// </summary>
    /// <param name="model">Model to save.</param>
    /// <param name="fileName">Name of file to save to.</param>
    public static void SaveModel(Model model, string fileName)
    {
        InitDirectory(); // ensure directory exists

        string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

        // Generate save data for the model
        var layers = new LayerData[model.Layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            var layer = model.Layers[i];

            switch (layer)
            {
                case Dense dense:
                    layers[i] = new(layerName: dense.GetType().AssemblyQualifiedName,
                        activation: dense.Activation.GetType().AssemblyQualifiedName,
                        dropout: dense.Dropout, biases: dense.Biases, neuronCount: dense.NeuronCount,
                        weights: dense.Weights);

                    break;
                case Conv conv:
                    layers[i] = new(layerName: conv.GetType().AssemblyQualifiedName,
                        activation: conv.Activation.GetType().AssemblyQualifiedName,
                        dropout: conv.Dropout, biases: conv.Biases, filterCount: conv.FilterCount,
                        kernelDims: conv.KernelDims, kernels: conv.Kernels);

                    break;
            }
        }
        ModelData modelData = new(layers);

        // Serialize the model as Json and save to file
        string json = JsonSerializer.Serialize(modelData);

        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads the model stored in the given file.
    /// </summary>
    /// <param name="fileName">Name of file to load from.</param>
    /// <returns>Model loaded from the given file.</returns>
    public static Model LoadModel(string fileName)
    {
        InitDirectory(); // ensure directory exists

        string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

        string json = File.ReadAllText(filePath); // load model Json

        // Reconstruct model from save data Json
        var modelData = JsonSerializer.Deserialize<ModelData>(json);
        Model model = new(modelData);

        return model;
    }
#pragma warning restore CS8604 // Possible null reference argument.

    /// <summary>
    /// Checks whether a neural network file with the given name exists.
    /// </summary>
    /// <param name="fileName">Name of file to search for.</param>
    /// <returns>Whether a neural network file with the given name exists.</returns>
    public static bool FileExists(string fileName)
    {
        InitDirectory(); // ensure directory exists

        string filePath = Path.Combine(DirectoryPath, fileName + Extension); // generate full file path

        if (File.Exists(filePath)) return true;
        else return false;
    }

    /// <summary>
    /// Initializes the directory and directory path if not yet initialized.
    /// </summary>
    static void InitDirectory()
    {
        if (string.IsNullOrEmpty(DirectoryPath))
        {
            string? exePath = System.Environment.ProcessPath;
            string? exeDirPath = Path.GetDirectoryName(exePath);
            if (!string.IsNullOrEmpty(exeDirPath))
            {
                DirectoryPath = Path.Combine(exeDirPath, DirectoryName);
                if (!Directory.Exists(DirectoryPath)) Directory.CreateDirectory(DirectoryPath);
            }
        }
    }
}
