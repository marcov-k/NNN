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
        readonly float Beta1;
        /// <summary>
        /// Precalculated 1 - beta1 value.
        /// </summary>
        readonly float OneMinusBeta1;
        /// <summary>
        /// Exponential decay rate of second moment estimates.
        /// </summary>
        readonly float Beta2;
        /// <summary>
        /// Precalculated 1 - beta2 value.
        /// </summary>
        readonly float OneMinusBeta2;
        /// <summary>
        /// Epsilon value to use.
        /// </summary>
        readonly float Epsilon;
        /// <summary>
        /// Weight decay to use.
        /// </summary>
        readonly float WeightDecay;

        public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
            float epsilon = 1e-8f, float weightDecay = 0.0f) : base(learningRate)
        {
            Beta1 = beta1;
            OneMinusBeta1 = 1.0f - beta1;
            Beta2 = beta2;
            OneMinusBeta2 = 1.0f - beta2;
            Epsilon = epsilon;
            WeightDecay = weightDecay;
        }

        /// <summary>
        /// Dictionary of per-parameter persistent buffers for first and second moments.
        /// </summary>
        readonly Dictionary<Tensor, (float[] m, float[] v)> _state = new();

        public override void Step(Tensor parameter, int iteration)
        {
            // Create a new persistent moment buffer if necessary
            if (!_state.TryGetValue(parameter, out var moments))
            {
                moments = (new float[parameter.ElementCount], new float[parameter.ElementCount]);
                _state[parameter] = moments;
            }
            var (m, v) = moments;

            NativeMethods.optimizers_adam(parameter.Handle, LR, iteration, m, v, m.Length, Beta1, OneMinusBeta1, Beta2,
                OneMinusBeta2, Epsilon, WeightDecay);
            GC.KeepAlive(parameter);
        }
    }
}
