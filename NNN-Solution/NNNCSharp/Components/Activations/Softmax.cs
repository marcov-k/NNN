using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Activations;

/// <summary>
/// Softmax activation function.
/// </summary>
public class Softmax : Activation
{
    /// <summary>
    /// Applies the Softmax activation function to each element in the input tensor.
    /// </summary>
    /// <param name="input">Tensor to apply the Softmax activation function to.</param>
    /// <returns>Tensor containing the result of applying the Softmax activation function to the input tensor.</returns>
    public override Tensor Forward(Tensor input) => Tensor.Softmax(input);

    /// <summary>
    /// Creates a new Softmax activation function instance.
    /// </summary>
    /// <returns>New Softmax activation function instance.</returns>
    public override Activation Copy() => new Softmax();

    public override void WriteUniqueData(FileStream stream) { }

    public override void BuildFromData(FileStream stream) { }

    public override string PrintActivation(FileStream stream) => string.Empty;
}
