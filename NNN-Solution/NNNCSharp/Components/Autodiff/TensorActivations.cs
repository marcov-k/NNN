using NNNCSharp.Components.Utilities;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Autodiff;

public partial class Tensor
{
    // Activation functions

    /// <summary>
    /// Applies the Rectified Linear Unit function to the input tensor.
    /// </summary>
    /// <param name="t">Tensor to apply ReLU function to.</param>
    /// <returns>Result tensor of applying ReLU function to the input tensor.</returns>
    public static Tensor ReLU(Tensor t) // ReLU(t) = t (t > 0), 0 (t <= 0)
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var vzero = Vector<double>.Zero; // preallocate 0's vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized ReLU result
        for (int i = 0; i < tVecs.Length; i++)
        {
            rVecs[i] = Vector.Max(tVecs[i], vzero);
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Max(t[i], 0.0);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = ReLU(t) -> dr/dt = 1 (t > 0), 0 (t <= 0)
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize inputs and gradients
                var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vzero = Vector<double>.Zero; // preallocate 0's vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients of parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    var mask = Vector.GreaterThan(tvVecs[i], vzero);
                    tgVecs[i] += Vector.ConditionalSelect(mask, rgVecs[i], vzero);
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    t.Grad[i] += (t[i] > 0.0 ? 1.0 : 0.0) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Applies the Leaky Rectified Linear Unit function to the input tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Leaky ReLU function to.</param>
    /// <param name="tau">Tau coefficient to use.</param>
    /// <returns>Result tensor of applying Leaky ReLU function to the input tensor.</returns>
    public static Tensor LeakyReLU(Tensor t, double tau) // Leaky ReLU(t) = t (t > 0), Tau * t (t <= 0)
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var vzero = Vector<double>.Zero; // preallocate 0's vector
        var vtau = new Vector<double>(tau); // splat scalar into vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized Leaky ReLU result
        for (int i = 0; i < tVecs.Length; i++)
        {
            var mask = Vector.GreaterThan(tVecs[i], vzero);
            rVecs[i] = Vector.ConditionalSelect(mask, tVecs[i], vtau * tVecs[i]);
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = t[i] > 0.0 ? t[i] : tau * t[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = Leaky ReLU(t) -> dr/dt = 1 (t > 0), Tau (t <= 0)
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize inputs and gradients
                var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vzero = Vector<double>.Zero; // preallocate 0's vector
                var vone = Vector<double>.One; // preallocate 1's vector
                var vtau = new Vector<double>(tau); // splat scalar into vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients of parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    var mask = Vector.GreaterThan(tvVecs[i], vzero);
                    tgVecs[i] += Vector.ConditionalSelect(mask, vone, vtau) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    t.Grad[i] += (t[i] > 0.0 ? 1.0 : tau) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Applies the sigmoid activation function to the input tensor.
    /// </summary>
    /// <param name="t">Tensor to apply sigmoid function to.</param>
    /// <returns>Result tensor of applying sigmoid function to the input tensor.</returns>
    public static Tensor Sigmoid(Tensor t) // Sigmoid(t) = 1 / (1 + e^(-t))
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var vone = Vector<double>.One; // preallocate 1's vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized sigmoid result
        for (int i = 0; i < tVecs.Length; i++)
        {
            rVecs[i] = vone / (vone + Vector.Exp(-tVecs[i]));
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = MathUtils.Sigmoid(t[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = Sigmoid(t) -> dr/dt = Sigmoid(t) * (1 - Sigmoid(t))
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize inputs, results, and gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vone = Vector<double>.One; // preallocate 1's vector
                var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    // Reuse forward results as Sigmoid(t)
                    tgVecs[i] += rvVecs[i] * (vone - rvVecs[i]) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    // Reuse forward results as Sigmoid(t)
                    t.Grad[i] += result[i] * (1.0 - result[i]) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Applies the hyperbolic tangent activation function to the input tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Tanh function to.</param>
    /// <returns>Result tensor of applying Tanh function to the input tensor.</returns>
    public static Tensor Tanh(Tensor t) // Tanh(t) = (e^2t - 1) / (e^2t + 1)
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var vone = Vector<double>.One; // preallocate 1's vector
        var vtwo = new Vector<double>(2.0); // preallocate 2's vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectoried Tanh result
        for (int i = 0; i < tVecs.Length; i++)
        {
            var e2t = Vector.Exp(vtwo * tVecs[i]);
            rVecs[i] = (e2t - vone) / (e2t + vone);
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = MathUtils.Tanh(t[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = Tanh(t) -> dr/dt = 1 - Tanh^2(t)
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize inputs, results, and gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vone = Vector<double>.One; // preallocate 1's tensor
                var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients of parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    // Reuse forward results as Tanh(t)
                    tgVecs[i] += (vone - (rvVecs[i] * rvVecs[i])) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    // Reuse forward results as Tanh(t)
                    t.Grad[i] += (1.0 - (result[i] * result[i])) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Applies the softmax activation function to the input tensor.
    /// </summary>
    /// <param name="t">Tensor to apply softmax function to.</param>
    /// <returns>Result tensor of applying softmax function to the input tensor.</returns>
    public static Tensor Softmax(Tensor t) // Softmax(z_i) = e^z_i / sum(j = 1 to n) [e^z_j]
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        int classes = t.Dimensions[^1]; // number of classification probabilities per output
        int batchSize = t.ElementCount / classes; // number of batches of separate outputs

        // Apply softmax function to each batch
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * classes; // linear offset of current batch

            // Vectorize inputs and results
            var tSlice = t.Data.AsSpan(offset, classes);
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(tSlice);
            var rSlice = result.Data.AsSpan(offset, classes);
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(rSlice);

            // Find vectorized maximum value in batch
            var vmax = new Vector<double>(double.MinValue);
            for (int i = 0; i < tVecs.Length; i++)
            {
                vmax = Vector.Max(vmax, tVecs[i]);
            }

            // Find single maximum value in vectorized maximum
            double max = double.MinValue;
            for (int lane = 0; lane < VectorSize; lane++)
            {
                max = Math.Max(max, vmax[lane]);
            }

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < classes; i++)
            {
                max = Math.Max(max, tSlice[i]);
            }

            // Calculate vectorized exponentiated values and exponentiated sum
            var vmaxSplat = new Vector<double>(max); // splat scalar into vector
            var acc = Vector<double>.Zero; // sum vector
            for (int i = 0; i < tVecs.Length; i++)
            {
                rVecs[i] = Vector.Exp(tVecs[i] - vmaxSplat);
                acc += rVecs[i];
            }
            double sum = Vector.Sum(acc);

            // Clean up unvectorized tail
            for (int i = tVecs.Length * VectorSize; i < classes; i++)
            {
                rSlice[i] = Math.Exp(tSlice[i] - max);
                sum += rSlice[i];
            }

            // Calculate vectorized normalized values
            var vsumSplat = new Vector<double>(sum); // splat scalar into vector
            for (int i = 0; i < rVecs.Length; i++)
            {
                rVecs[i] /= vsumSplat;
            }

            // Clean up unvectorized tail
            for (int i = rVecs.Length * VectorSize; i < classes; i++)
            {
                rSlice[i] /= sum;
            }
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = softmax(t) -> grad_t_i = r_i * (grad_r_i - sum_j(grad_r_j * r_j)) -> Vector-Jacobian product
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Calculate gradient per batch
                for (int b = 0; b < batchSize; b++)
                {
                    int offset = b * classes; // linear offset of current batch

                    // Vectorize inputs, results, and gradients
                    var tgSlice = t.Grad.AsSpan(offset, classes);
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(tgSlice);
                    var rvSlice = result.Data.AsSpan(offset, classes);
                    var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(rvSlice);
                    var rgSlice = result.Grad.AsSpan(offset, classes);
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);

                    // Calculate vectorized dot product of result gradient and forward softmax result -> sum_j(grad_r_j * r_j)
                    var vdot = Vector<double>.Zero; // product/sum vector
                    for (int i = 0; i < rvVecs.Length; i++)
                    {
                        vdot += rgVecs[i] * rvVecs[i];
                    }
                    double dot = Vector.Sum(vdot);

                    // Clean up unvectorized tail
                    for (int i = rvVecs.Length * VectorSize; i < classes; i++)
                    {
                        dot += rgSlice[i] * rvSlice[i];
                    }

                    // Calculate vectorized gradients for parent
                    var vdotSplat = new Vector<double>(dot); // splat scalar into vector
                    for (int i = 0; i < tgVecs.Length; i++)
                    {
                        tgVecs[i] += rvVecs[i] * (rgVecs[i] - vdotSplat);
                    }

                    // Clean up unvectorized tail
                    for (int i = tgVecs.Length * VectorSize; i < classes; i++)
                    {
                        tgSlice[i] += rvSlice[i] * (rgSlice[i] - dot);
                    }
                }
            };
        }

        return result;
    }
}
