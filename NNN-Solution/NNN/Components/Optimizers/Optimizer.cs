using NNN.Components.Autodiff;
using System.Numerics;

namespace NNN.Components.Optimizers;

/// <summary>
/// Base class for optimizers used for updating model parameters.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
public abstract class Optimizer(double learningRate)
{
    /// <summary>
    /// Gradient scaling factor for parameter updates.
    /// </summary>
    public double LR
    {
        get => _lr;
        set
        {
            _lr = value;
            LRVec = new(value);
        }
    }
    /// <summary>
    /// Internal learning rate property.
    /// </summary>
    protected double _lr = learningRate;
    /// <summary>
    /// Preallocated vectorization of learning rate.
    /// </summary>
    protected Vector<double> LRVec = new(learningRate);
    /// <summary>
    /// Size of vectors in the current CPU architecture.
    /// </summary>
    protected static readonly int VectorSize = Vector<double>.Count;

    /// <summary>
    /// Updates the parameter based on its gradient.
    /// </summary>
    /// <param name="parameter">Parameter tensor to be updated.</param>
    /// <param name="iterations">Number of training iterations which have been run.</param>
    public abstract void Step(Tensor parameter, int iterations);
}
