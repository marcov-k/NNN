using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Interop;
using System;

namespace NNNCSharp.Components.Optimizers
{
    /// <summary>
    /// Stochastic Gradient Descent optimizer.
    /// </summary>
    /// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
    public class SGD: Optimizer
    {
        public SGD(double learningRate) : base(learningRate) { }

        public override void Step(Tensor parameter, int iterations)
        {
            NativeMethods.optimizers_sgd(parameter.Handle, LR);
            GC.KeepAlive(parameter);
        }
    }
}
