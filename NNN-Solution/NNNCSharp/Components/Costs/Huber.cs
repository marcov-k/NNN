using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Costs;

/// <summary>
/// Pseudo-Huber (smoothed) cost function.
/// </summary>
/// <param name="delta">Linear transition threshold.</param>
public class Huber(double delta = 1.0) : Cost
{
    /// <summary>
    /// Linear transition threshold.
    /// </summary>
    readonly double Delta = delta;

    public override Tensor CalculateCost(Tensor predictions, Tensor target)
    {
        return Tensor.Mean(CalculatePerSampleCost(predictions, target));
    }

    public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
    {
        var diff = target - predictions; // calculate per-prediction error

        // Apply pseudo-Huber function -> delta^2 * (sqrt(1 + (diff/delta)^2) - 1)
        var scaled = diff / Delta;
        var inner = Tensor.Pow(scaled, 2.0) + 1.0;
        return (Tensor.Pow(inner, 0.5) - 1.0) * (Delta * Delta);
    }
}
