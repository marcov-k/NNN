using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Activations;

/// <summary>
/// Linear activation function (does not modify input).
/// </summary>
public class Linear : Activation
{
    /// <summary>
    /// Applies the linear activation function to each element in the input tensor.
    /// </summary>
    /// <param name="input">Tensor to apply the linear activation function to.</param>
    /// <returns>Tensor containing the result of applying the linear activation function to the input tensor.</returns>
    public override Tensor Forward(Tensor input) => input; // does not modify input -> linear function

    /// <summary>
    /// Creates a new linear activation function instance.
    /// </summary>
    /// <returns>New linear activation function instance.</returns>
    public override Activation Copy() => new Linear();

    public override void WriteUniqueData(FileStream stream) { }

    public override void BuildFromData(FileStream stream) { }

    public override string PrintActivation(FileStream stream) => string.Empty;
}
