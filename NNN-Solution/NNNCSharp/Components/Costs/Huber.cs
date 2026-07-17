using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Costs
{
    /// <summary>
    /// Pseudo-Huber (smoothed) cost function.
    /// </summary>
    /// <param name="delta">Linear transition threshold.</param>
    public class Huber : Cost
    {
        /// <summary>
        /// Linear transition threshold.
        /// </summary>
        readonly double Delta;

        public Huber(double delta = 1.0)
        {
            Delta = delta;
        }

        public override Tensor CalculateCost(Tensor predictions, Tensor target)
        {
            return Tensor.Mean(CalculatePerSampleCost(predictions, target));
        }

        public override Tensor CalculatePerSampleCost(Tensor predictions, Tensor target)
        {
            return Tensor.Huber(predictions, target, Delta);
        }
    }
}
