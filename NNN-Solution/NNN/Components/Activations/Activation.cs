using NNN.Components.Autodiff;

namespace NNN.Components.Activations;

/// <summary>
/// Base class for neural network activation functions.
/// </summary>
public abstract class Activation
{
    /// <summary>
    /// Applies the activation function to each element in the input tensor.
    /// </summary>
    /// <param name="input">Tensor to apply the activation function to.</param>
    /// <returns>Tensor containing the result of applying the activation function to the input tensor.</returns>
    public abstract Tensor Forward(Tensor input);

    /// <summary>
    /// Creates a new instance of the same activation function.
    /// </summary>
    /// <returns>New instance of the same activation function.</returns>
    public abstract Activation Copy(); // used to ensure function-specific parameters are persisted during copies

    /// <summary>
    /// Writes any activation type-specific data to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    public abstract void WriteUniqueData(FileStream stream);

    /// <summary>
    /// Fills the activation instance with data at the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    public abstract void BuildFromData(FileStream stream);

    /// <summary>
    /// Converts any activation type-specific data from the file stream into a readable string.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Readable string of the activation type-specific data.</returns>
    public abstract string PrintActivation(FileStream stream);
}
