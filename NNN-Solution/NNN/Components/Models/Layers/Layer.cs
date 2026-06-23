using NNN.Components.Activations;
using NNN.Components.Autodiff;
using NNN.Components.Utilities;
using NNN.Components.Utilities.SaveSystem;

namespace NNN.Components.Models.Layers;

/// <summary>
/// Base class for neural network layers.
/// </summary>
public abstract class Layer
{
    /// <summary>
    /// Tensor containing the bias parameters of the layer.
    /// </summary>
    public Tensor Biases { get; protected set; } = new();
    /// <summary>
    /// Activation function used by the layer.
    /// </summary>
    public Activation Activation { get; protected set; } = new Linear();
    /// <summary>
    /// Format of the output of the layer.
    /// </summary>
    public Tensor OutputFormat { get; protected set; } = new();
    /// <summary>
    /// Dropout rate of the layer.
    /// </summary>
    public double Dropout { get; protected set; } = 0.0;

    /// <summary>
    /// Parameterless constructor for model reconstruction from save data.
    /// </summary>
    public Layer() { }

    /// <summary>
    /// Initializes the layer's parameters for the given number of inputs.
    /// </summary>
    /// <param name="inputFormat">Expected format of input tensors.</param>
    public abstract void SetUpLayer(Tensor inputFormat);

    /// <summary>
    /// Processes the input using the layer's parameters and activation function.
    /// </summary>
    /// <param name="input">Input tensor to be processed.</param>
    /// <returns>Tensor containing the outputs of the layer's operations.</returns>
    public abstract Tensor Forward(Tensor input);

    /// <summary>
    /// Gets all of the layer's parameters.
    /// </summary>
    /// <returns>IEnumerable containing all of the layer's parameter tensors.</returns>
    public abstract IEnumerable<Tensor> GetParameters();

    /// <summary>
    /// Creates a new deep copy of the layer with the same parameters and activation function.
    /// </summary>
    /// <returns>New layer instance with identical parameters and activation function.</returns>
    public abstract Layer Copy();

    /// <summary>
    /// Writes any layer type-specific data to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    public abstract void WriteUniqueData(FileStream stream);

    /// <summary>
    /// Fills the layer instance with data at the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    public void BuildFromData(FileStream stream)
    {
        var activType = IDManager.GetActivationByID((byte)stream.ReadByte());
        Activation = (Activator.CreateInstance(activType) as Activation)!;
        Activation.BuildFromData(stream);

        Dropout = FileUtils.ReadDouble(stream);
        Biases = FileUtils.ReadTensor(stream);

        ReadUniqueData(stream);
    }

    /// <summary>
    /// Reads any layer type-specific data from the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    protected abstract void ReadUniqueData(FileStream stream);

    /// <summary>
    /// Converts the layer data from the file stream into a readable string.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Readable string of the layer data.</returns>
    public string PrintLayer(FileStream stream)
    {
        var activType = IDManager.GetActivationByID((byte)stream.ReadByte());
        var activation = Activator.CreateInstance(activType) as Activation;
        string activData = activation!.PrintActivation(stream);
        double dropout = FileUtils.ReadDouble(stream);
        string biases = $"Bias Tensor: {FileUtils.PrintTensor(stream)}";
        string additionalData = PrintUniqueLayer(stream);

        return $"{additionalData}\n{biases}\nDropout: {dropout}\nActivation: {activType.Name}{activData}";
    }

    /// <summary>
    /// Converts layer type-specific data from the file stream into a readable string.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Readable string of the layer type-specific data.</returns>
    protected abstract string PrintUniqueLayer(FileStream stream);
}
