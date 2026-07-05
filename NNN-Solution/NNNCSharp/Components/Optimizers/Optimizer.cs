using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Optimizers;

/// <summary>
/// Base class for optimizers used for updating model parameters.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
public abstract class Optimizer(double learningRate)
{
    /// <summary>
    /// Gradient scaling factor for parameter updates.
    /// </summary>
    public double LR { get; set; } = learningRate;

    /// <summary>
    /// Updates the parameter based on its gradient.
    /// </summary>
    /// <param name="parameter">Parameter tensor to be updated.</param>
    /// <param name="iterations">Number of training iterations which have been run.</param>
    public abstract void Step(Tensor parameter, int iterations);
}
