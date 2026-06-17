using NNN.Components.Autodiff;

namespace NNN.Components.Activations;

/// <summary>
/// Hyperbolic tangent activation function.
/// </summary>
public class Tanh : Activation
{
    /// <summary>
    /// Applies the Tanh activation function to each element in the input tensor.
    /// </summary>
    /// <param name="input">Tensor to apply the Tanh activation function to.</param>
    /// <returns>Tensor containing the result of applying the Tanh activation function to the input tensor.</returns>
    public override Tensor Forward(Tensor input) => Tensor.Tanh(input);

    /// <summary>
    /// Creates a new Tanh activation function instance.
    /// </summary>
    /// <returns>New Tanh activation function instance.</returns>
    public override Activation Copy() => new Tanh();
}
