using NNN.Components.Autodiff;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNN.Components.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
public class SGD(double learningRate) : Optimizer(learningRate)
{
    public override void Step(Tensor parameter, int iterations)
    {
        var paramVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Data.AsSpan());
        var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Grad.AsSpan());
        for (int i = 0; i < paramVecs.Length; i++)
        {
            paramVecs[i] -= gradVecs[i] * LRVec;
        }

        for (int i = paramVecs.Length * VectorSize; i < parameter.ElementCount; i++)
        {
            parameter[i] -= parameter.Grad[i] * LR;
        }
    }
}
