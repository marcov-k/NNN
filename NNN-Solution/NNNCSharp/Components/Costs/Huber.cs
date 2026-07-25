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
        readonly float Delta;

        public Huber(float delta = 1.0f)
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
