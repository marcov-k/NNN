using NNNCSharp.Components.Interop;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
public class SGD(double learningRate) : Optimizer(learningRate)
{
    public override void Step(Tensor parameter, int iterations)
    {
        var paramVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Data);
        var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Grad);
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
