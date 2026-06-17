using NNN.Components.Autodiff;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNN.Components.Optimizers;

/// <summary>
/// Adaptive Moment Estimation optimizer.
/// </summary>
/// <param name="learningRate">Gradient scaling factor for parameter updates.</param>
/// <param name="beta1">Exponential decay rate of first moment estimates.</param>
/// <param name="beta2">Exponential decay rate of second moment estimates.</param>
/// <param name="epsilon">Epsilon value to use.</param>
public class Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : Optimizer(learningRate)
{
    /// <summary>
    /// Exponential decay rate of first moment estimates.
    /// </summary>
    readonly double Beta1 = beta1;
    /// <summary>
    /// Precalculated 1 - beta1 value.
    /// </summary>
    readonly double OneMinusBeta1 = 1.0 - beta1;
    /// <summary>
    /// Preallocated vectorization of beta1 value.
    /// </summary>
    readonly Vector<double> Beta1Vec = new(beta1);
    /// <summary>
    /// Preallocated vectorization of 1 - beta1 value.
    /// </summary>
    readonly Vector<double> OneMinusBeta1Vec = new(1.0 - beta1);
    /// <summary>
    /// Exponential decay rate of second moment estimates.
    /// </summary>
    readonly double Beta2 = beta2;
    /// <summary>
    /// Precalculated 1 - beta2 value.
    /// </summary>
    readonly double OneMinusBeta2 = 1.0 - beta2;
    /// <summary>
    /// Preallocated vectorization of beta2 value.
    /// </summary>
    readonly Vector<double> Beta2Vec = new(beta2);
    /// <summary>
    /// Preallocated vectorization of 1 - beta2 value.
    /// </summary>
    readonly Vector<double> OneMinusBeta2Vec = new(1.0 - beta2);
    /// <summary>
    /// Epsilon value to use.
    /// </summary>
    readonly double Epsilon = epsilon;
    /// <summary>
    /// Preallocated vectorization of epsilon value.
    /// </summary>
    readonly Vector<double> EpsilonVec = new(epsilon);

    /// <summary>
    /// Dictionary of per-parameter persistent buffers for first and second moments.
    /// </summary>
    readonly Dictionary<Tensor, (double[] m, double[] v)> _state = [];

    public override void Step(Tensor parameter, int iteration)
    {
        // Create a new persistent moment buffer if necessary
        if (!_state.TryGetValue(parameter, out var moments))
        {
            moments = (new double[parameter.ElementCount], new double[parameter.ElementCount]);
            _state[parameter] = moments;
        }

        // Update moments and parameter values
        double biasCorrection1 = 1.0 - Math.Pow(Beta1, iteration + 1);
        var biasCorr1Vec = new Vector<double>(biasCorrection1);

        double biasCorrection2 = 1.0 - Math.Pow(Beta2, iteration + 1);
        var biasCorr2Vec = new Vector<double>(biasCorrection2);
        var (m, v) = moments;

        var mVecs = MemoryMarshal.Cast<double, Vector<double>>(m.AsSpan());
        var vVecs = MemoryMarshal.Cast<double, Vector<double>>(v.AsSpan());
        var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Grad.AsSpan());
        var paramVecs = MemoryMarshal.Cast<double, Vector<double>>(parameter.Data.AsSpan());

        for (int i = 0; i < paramVecs.Length; i++)
        {
            mVecs[i] = (Beta1Vec * mVecs[i]) + (OneMinusBeta1Vec * gradVecs[i]);
            vVecs[i] = (Beta2Vec * vVecs[i]) + (OneMinusBeta2Vec * (gradVecs[i] * gradVecs[i]));

            var mHatVec = mVecs[i] / biasCorr1Vec;
            var vHatVec = vVecs[i] / biasCorr2Vec;

            paramVecs[i] -= (LRVec * mHatVec) / (Vector.SquareRoot(vHatVec) + EpsilonVec);
        }

        for (int i = paramVecs.Length * VectorSize; i < parameter.ElementCount; i++)
        {
            double grad = parameter.Grad[i];

            // Update parameter moments
            m[i] = Beta1 * m[i] + (OneMinusBeta1 * grad);
            v[i] = Beta2 * v[i] + (OneMinusBeta2 * (grad * grad));

            // Correct moment estimate bias due to 0 initialization
            double mHat = m[i] / biasCorrection1;
            double vHat = v[i] / biasCorrection2;

            parameter.Data[i] -= (LR * mHat) / (Math.Sqrt(vHat) + Epsilon); // update parameter based on moments
        }
    }
}
