using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Costs;

/// <summary>
/// Mean Squared Error cost function.
/// </summary>
public class MSE : Cost
{
    public override Tensor CalculateCost(Tensor predictions, Tensor target)
    {
        return Tensor.Mean(CalculatePerSampleCost(predictions, target));
    }

    public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
    {
        return Tensor.MSE(predictions, target);
    }
}
