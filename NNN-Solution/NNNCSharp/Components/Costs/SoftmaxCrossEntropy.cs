using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Costs
{
    /// <summary>
    /// Softmax Cross-Entropy cost function.
    /// </summary>
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
}
