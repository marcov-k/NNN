using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Interop;
using System;
using System.Collections.Generic;

namespace NNNCSharp.Components.Optimizers
{
    /// <summary>
    /// Adaptive Moment Estimation optimizer.
    /// </summary>
    /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
    /// <param name="beta1">Exponential decay rate of first moment estimates.</param>
    /// <param name="beta2">Exponential decay rate of second moment estimates.</param>
    /// <param name="epsilon">Epsilon value to use.</param>
    /// <param name="weightDecay">Weight decay value to use.</param>
    public class Adam : Optimizer
    {
        /// <summary>
        /// Exponential decay rate of first moment estimates.
        /// </summary>
        readonly double Beta1;
        /// <summary>
        /// Precalculated 1 - beta1 value.
        /// </summary>
        readonly double OneMinusBeta1;
        /// <summary>
        /// Exponential decay rate of second moment estimates.
        /// </summary>
        readonly double Beta2;
        /// <summary>
        /// Precalculated 1 - beta2 value.
        /// </summary>
        readonly double OneMinusBeta2;
        /// <summary>
        /// Epsilon value to use.
        /// </summary>
        readonly double Epsilon;
        /// <summary>
        /// Weight decay to use.
        /// </summary>
        readonly double WeightDecay;

        public Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double epsilon = 1e-8, double weightDecay = 0.0) : base(learningRate)
        {
            Beta1 = beta1;
            OneMinusBeta1 = 1.0 - beta1;
            Beta2 = beta2;
            OneMinusBeta2 = 1.0 - beta2;
            Epsilon = epsilon;
            WeightDecay = weightDecay;
        }

        /// <summary>
        /// Dictionary of per-parameter persistent buffers for first and second moments.
        /// </summary>
        readonly Dictionary<Tensor, (double[] m, double[] v)> _state = new();

        public override void Step(Tensor parameter, int iteration)
        {
            // Create a new persistent moment buffer if necessary
            if (!_state.TryGetValue(parameter, out var moments))
            {
                moments = (new double[parameter.ElementCount], new double[parameter.ElementCount]);
                _state[parameter] = moments;
            }
            var (m, v) = moments;

            NativeMethods.optimizers_adam(parameter.Handle, LR, iteration, m, v, m.Length, Beta1, OneMinusBeta1, Beta2,
                OneMinusBeta2, Epsilon, WeightDecay);
            GC.KeepAlive(parameter);
        }
    }
}
