using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Costs;

public class SoftmaxCrossEntropy : Cost
{
    public override Tensor CalculateCost(Tensor predictions, Tensor target)
    {
        return Tensor.Mean(CalculatePerSampleCost(predictions, target));
    }

    public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
    {
        return Tensor.SoftmaxCrossEntropy(predictions, target);
    }
}
