using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
public class SGD(double learningRate) : Optimizer(learningRate)
{
    public override void Step(Tensor parameter, int iterations)
    {
        NativeMethods.optimizers_sgd(parameter.Handle, LR);
        GC.KeepAlive(parameter);
    }
}
