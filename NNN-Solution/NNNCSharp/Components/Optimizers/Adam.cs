using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Interop;
using System;
using System.Collections.Generic;

namespace NNNCSharp.Components.Optimizers
{
    /// <summary>
    /// Adaptive Moment Estimation optimizer.
    /// </summary>
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
        /// <summary>
        /// Dictionary of per-parameter persistent buffers for first and second moments.
        /// </summary>
        readonly Dictionary<Tensor, (IntPtr m, IntPtr v)> _state = new();

        /// <summary>
        /// Creates a new Adam optimizer instance.
        /// </summary>
        /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
        /// <param name="beta1">Exponential decay rate of first moment estimates.</param>
        /// <param name="beta2">Exponential decay rate of second moment estimates.</param>
        /// <param name="epsilon">Epsilon value to use.</param>
        /// <param name="weightDecay">Weight decay value to use.</param>
        public Adam(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f,
            float epsilon = 1e-8f, float weightDecay = 0.0f) : base(learningRate)
        {
            Beta1 = beta1;
            OneMinusBeta1 = 1.0f - beta1;
            Beta2 = beta2;
            OneMinusBeta2 = 1.0f - beta2;
            Epsilon = epsilon;
            WeightDecay = weightDecay;
        }

        ~Adam() // free each moment vector allocation via C++
        {
            foreach (var (m, v) in _state.Values)
            {
                NativeMethods.free_aligned(m);
                NativeMethods.free_aligned(v);
            }
        }

        public override void Step(Tensor parameter, int iteration)
        {
            // Create a new persistent 32-byte aligned moment buffer if necessary
            if (!_state.TryGetValue(parameter, out var moments))
            {
                // Allocate moment vectors via C++ aligned allocator
                int byteCount = parameter.ElementCount * sizeof(float);
                IntPtr m = NativeMethods.alloc_aligned((UIntPtr)byteCount, (UIntPtr)32);
                IntPtr v = NativeMethods.alloc_aligned((UIntPtr)byteCount, (UIntPtr)32);
                unsafe // zero out newly allocated moment vector memory
                {
                    new Span<byte>((void*)m, byteCount).Clear();
                    new Span<byte>((void*)v, byteCount).Clear();
                }
                moments = (m, v);
                _state[parameter] = moments;
            }

            NativeMethods.optimizers_adam(parameter.Handle, LR, iteration, moments.m, moments.v, parameter.ElementCount,
                Beta1, OneMinusBeta1, Beta2, OneMinusBeta2, Epsilon, WeightDecay);
            GC.KeepAlive(parameter);
        }
    }
}
