using NNN.Components.Activations;
using NNN.Components.Autodiff;
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
    /// Reconstructs the layer from the save data.
    /// </summary>
    /// <param name="data">Save data containing the layer's saved parameters.</param>
    public abstract void BuildFromData(LayerData data);
}
