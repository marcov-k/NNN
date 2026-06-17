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
}
