using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities.SaveSystem;

namespace NNNCSharp.Components.Activations;

/// <summary>
/// Leaky Rectified Linear Unit activation function.
/// </summary>
public class LeakyReLU : Activation
{
    /// <summary>
    /// Coefficient for negative inputs.
    /// </summary>
    double Tau;

    /// <summary>
    /// Creates a new Leaky ReLU activation function instance.
    /// </summary>
    /// <param name="tau">Coefficient for negative inputs.</param>
    public LeakyReLU(double tau = 0.01)
    {
        Tau = tau;
    }

    /// <summary>
    /// Parameterless constructor for model reconstruction from save data.
    /// </summary>
    public LeakyReLU() { }

    /// <summary>
    /// Applies the Leaky ReLU activation function to each element in the input tensor.
    /// </summary>
    /// <param name="input">Tensor to apply the Leaky ReLU activation function to.</param>
    /// <returns>Tensor containing the result of applying the Leaky ReLU activation function to the input tensor.</returns>
    public override Tensor Forward(Tensor input) => Tensor.LeakyReLU(input, Tau);

    /// <summary>
    /// Creates a new Leaky ReLU activation function instance with the same Tau value.
    /// </summary>
    /// <returns>New Leaky ReLU activation function instance with the same Tau value.</returns>
    public override Activation Copy() => new LeakyReLU(Tau);

    public override void WriteUniqueData(FileStream stream)
    {
        FileUtils.WriteDouble(stream, Tau);
    }

    public override void BuildFromData(FileStream stream)
    {
        Tau = FileUtils.ReadDouble(stream);
    }

    public override string PrintActivation(FileStream stream)
    {
        double tau = FileUtils.ReadDouble(stream);
        return $", Tau: {tau}";
    }
}
