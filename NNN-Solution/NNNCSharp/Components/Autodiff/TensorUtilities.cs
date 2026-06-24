using NNNCSharp.Components.Episodes;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Autodiff;

public partial class Tensor
{
    // Other neural network utility functions

    /// <summary>
    /// Calculates the flat sum of all elements in the input tensor.
    /// </summary>
    /// <param name="t">Tensor to calculate sum of.</param>
    /// <returns>Tensor containing the flat sum of the input tensor.</returns>
    public static Tensor Sum(Tensor t)
    {
        Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var acc = Vector<double>.Zero; // sum vector

        // Calculate vectorized sum
        for (int i = 0; i < tVecs.Length; i++)
        {
            acc += tVecs[i];
        }
        double sum = Vector.Sum(acc);

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < t.ElementCount; i++)
        {
            sum += t[i];
        }

        result[0] = sum;

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = sum(t) -> grad_t_i = grad_r
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vrg = new Vector<double>(result.Grad[0]); // splat scalar into vector

                // Calculate vectorized gradients for parent
                for (int i = 0; i < tgVecs.Length; i++)
                {
                    tgVecs[i] += vrg;
                }

                // Clean up unvectorized tail
                for (int i = tgVecs.Length * VectorSize; i < t.ElementCount; i++)
                {
                    t.Grad[i] += result.Grad[0];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Calculates the flat mean average of all elements in the input tensor.
    /// </summary>
    /// <param name="t">Tensor to calculate mean of.</param>
    /// <returns>Tensor containing the flat mean average of the input tensor.</returns>
    public static Tensor Mean(Tensor t)
    {
        Tensor result = GetResultTensor(t, [1], t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var acc = Vector<double>.Zero; // sum vector

        // Calculate vectorized sum
        for (int i = 0; i < tVecs.Length; i++)
        {
            acc += tVecs[i];
        }
        double sum = Vector.Sum(acc);

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < t.ElementCount; i++)
        {
            sum += t[i];
        }

        result[0] = sum / t.ElementCount; // calculate mean

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = mean(t) -> dr/dt = 1 / n
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                double rg = result.Grad[0] / t.ElementCount; // precalculate 1 / n * grad_r
                var vrg = new Vector<double>(rg); // splat scalar into vector

                // Calculate vectorized gradients for parent
                for (int i = 0; i < tgVecs.Length; i++)
                {
                    tgVecs[i] += vrg;
                }

                // Clean up unvectorized tail
                for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                {
                    t.Grad[i] += rg;
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Performs the Softmax Cross-Entropy function between the given tensor and a target tensor.
    /// </summary>
    /// <param name="t">Tensor to perform the Softmax Cross-Entropy function on.</param>
    /// <param name="target">Target tensor to use.</param>
    /// <returns>Result tensor of performing the Softmax Cross-Entropy function between the given and target tensors.</returns>
    public static Tensor SoftmaxCrossEntropy(Tensor t, Tensor target)
    {
        int classes = t.Dimensions[^1];
        int batchSize = t.ElementCount / classes;

        var result = GetResultTensor(t, [batchSize, 1], t.RequiresGrad);

        var probs = new double[t.ElementCount];

        // Apply Softmax Cross-Entropy per batch
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * classes;

            var tSlice = t.Data.AsSpan(offset, classes);
            var tVecs = MemoryMarshal.Cast<double, Vector<double>>(tSlice);

            // Calculate max
            var vmax = new Vector<double>(double.MinValue);
            for (int i = 0; i < tVecs.Length; i++)
            {
                vmax = Vector.Max(vmax, tVecs[i]);
            }

            double max = double.MinValue;
            for (int lane = 0; lane < VectorSize; lane++)
            {
                max = Math.Max(max, vmax[lane]);
            }

            for (int i = tVecs.Length * VectorSize; i < classes; i++)
            {
                max = Math.Max(max, tSlice[i]);
            }

            var pSlice = probs.AsSpan(offset, classes);
            var pVecs = MemoryMarshal.Cast<double, Vector<double>>(pSlice);

            // Calculate exponent sum
            var vmaxSplat = new Vector<double>(max); // splat scalar into vector
            var acc = Vector<double>.Zero;
            for (int i = 0; i < tVecs.Length; i++)
            {
                pVecs[i] = Vector.Exp(tVecs[i] - vmaxSplat);
                acc += pVecs[i];
            }
            double sumExp = Vector.Sum(acc);

            for (int i = tVecs.Length * VectorSize; i < classes; i++)
            {
                pSlice[i] = Math.Exp(tSlice[i] - max);
                sumExp += pSlice[i];
            }

            double logSumExp = Math.Log(sumExp) + max;

            // Normalize values by exponent sum
            var vsumSplat = new Vector<double>(sumExp); // splat scalar into vector
            for (int i = 0; i < pVecs.Length; i++)
            {
                pVecs[i] /= vsumSplat;
            }

            for (int i = pVecs.Length * VectorSize; i < classes; i++)
            {
                pSlice[i] /= sumExp;
            }

            // Calculate dot product of class values
            var gSlice = target.Data.AsSpan(offset, classes);
            var gVecs = MemoryMarshal.Cast<double, Vector<double>>(gSlice);
            var vdot = Vector<double>.Zero;
            for (int i = 0; i < tVecs.Length; i++)
            {
                vdot += gVecs[i] * tVecs[i];
            }
            double dot = Vector.Sum(vdot);

            for (int i = tVecs.Length * VectorSize; i < classes; i++)
            {
                dot += gSlice[i] * tSlice[i];
            }

            result[b] = logSumExp - dot;
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange(t);

            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Calculate gradient per batch
                for (int b = 0; b < batchSize; b++)
                {
                    int offset = b * classes;

                    double rg = result.Grad[b];

                    // Vectorize values
                    var vrg = new Vector<double>(rg); // splat scalar into vector
                    var pSlice = probs.AsSpan(offset, classes);
                    var pVecs = MemoryMarshal.Cast<double, Vector<double>>(pSlice);
                    var gSlice = target.Data.AsSpan(offset, classes);
                    var gVecs = MemoryMarshal.Cast<double, Vector<double>>(gSlice);
                    var tgSlice = t.Grad.AsSpan(offset, classes);
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(tgSlice);

                    // Calculate vectorized gradients
                    for (int i = 0; i < pVecs.Length; i++)
                    {
                        tgVecs[i] += (pVecs[i] - gVecs[i]) * vrg;
                    }

                    // Clean up unvectorized tail
                    for (int i = pVecs.Length * VectorSize; i < classes; i++)
                    {
                        tgSlice[i] += (pSlice[i] - gSlice[i]) * rg;
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Masks predicted Q-Values based on action selected per experience.
    /// </summary>
    /// <param name="qValues">Q-Values to mask.</param>
    /// <param name="batch">Batch of corresponding experiences.</param>
    /// <returns>Tensor containing the masked predicted Q-Values per experience.</returns>
    public static Tensor MaskActions(Tensor qValues, List<Experience> batch)
    {
        int batchSize = batch.Count;
        int actionCount = qValues.Dimensions[^1];

        Tensor result = GetResultTensor(qValues, [batchSize, 1], qValues.RequiresGrad);

        // Find corresponding Q-Values of experience actions
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = qValues[i * actionCount + batch[i].Action];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(qValues);

            // Gradient calculation function for r = Mask(q) -> dr_i/dq_i = 1
            result._backward = () =>
            {
                if (!qValues.RequiresGrad) return;

                for (int i = 0; i < batchSize; i++)
                {
                    qValues.Grad[i * actionCount + batch[i].Action] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Finds the index of the largest value in the given tensor.
    /// </summary>
    /// <param name="t">Tensor to find maximum of.</param>
    /// <returns>Index of the maximum value in the input tensor.</returns>
    public static int ArgMax(Tensor t)
    {
        int index = 0;
        double max = t[0];

        for (int i = 1; i < t.ElementCount; i++)
        {
            if (t[i] > max)
            {
                max = t[i];
                index = i;
            }
        }

        return index;
    }

    /// <summary>
    /// Transposes the given tensor based on the given axes permutation order.
    /// </summary>
    /// <param name="t">Tensor to transpose.</param>
    /// <param name="axes">Axes permutation order to transpose by.</param>
    /// <returns>Transpose of the input tensor based on the given axes permutation order.</returns>
    public static Tensor Transpose(Tensor t, int[]? axes = null)
    {
        // Default permutation order: reverse all axes
        if (axes is null)
        {
            axes = new int[t.Rank];
            for (int i = 0; i < axes.Length; i++)
            {
                axes[i] = axes.Length - 1 - i;
            }
        }

        // Build result dimensions
        var resultDims = new int[t.Rank];
        for (int i = 0; i < axes.Length; i++)
        {
            resultDims[i] = t.Dimensions[axes[i]];
        }

        Tensor result = GetResultTensor(t, resultDims, t.RequiresGrad);

        // Preallocate index mappings on stack - avoid garbage collector
        Span<int> srcIndices = stackalloc int[t.Rank];
        Span<int> dstIndices = stackalloc int[axes.Length];

        // Remap each element based on permutation order
        for (int i = 0; i < t.ElementCount; i++)
        {
            t.GetFullIndices(i, srcIndices);
            for (int j = 0; j < axes.Length; j++)
            {
                dstIndices[j] = srcIndices[axes[j]];
            }
            result[result.LinearIndex(dstIndices)] = t[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Inverse permutation order
            var invAxes = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                invAxes[axes[i]] = i;
            }

            // Gradient calculation function for r = Transpose(t) -> grad_t = Transpose(grad_r)
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Preallocate index mappings on stack - avoid garbage collector
                Span<int> srcIndices = stackalloc int[result.Rank];
                Span<int> dstIndices = stackalloc int[invAxes.Length];

                // Remap each gradient based on inverse permutation order
                for (int i = 0; i < result.ElementCount; i++)
                {
                    result.GetFullIndices(i, srcIndices);
                    for (int j = 0; j < invAxes.Length; j++)
                    {
                        dstIndices[j] = srcIndices[invAxes[j]];
                    }
                    t.Grad[t.LinearIndex(dstIndices)] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Broadcasts a tensor into the given new dimensions.
    /// </summary>
    /// <param name="t">Tensor to broadcast.</param>
    /// <param name="targetDims">Dimensions to broadcast into.</param>
    /// <returns>Tensor with target dimensions and data from the input tensor.</returns>
    public static Tensor Broadcast(Tensor t, int[] targetDims)
    {
        // T.Dimensions must be a suffix of targetDims (t -> [16], targetDims = [32, 16])

        Tensor result = GetResultTensor(t, targetDims, t.RequiresGrad);

        // Shortcut for 1D input
        if (t.Rank == 1)
        {
            int stride = t.ElementCount;
            int rows = result.ElementCount / stride;
            for (int r = 0; r < rows; r++)
            {
                t.Data.AsSpan(0, stride).CopyTo(result.Data.AsSpan(r * stride, stride));
            }
        }
        else
        {
            // Preallocate index mappings on stack - avoid garbage collector
            Span<int> indices = stackalloc int[result.Rank];
            Span<int> srcIndices = stackalloc int[t.Rank];

            // Copy input tensor data into result tensor
            for (int i = 0; i < result.ElementCount; i++)
            {
                result.GetFullIndices(i, indices);
                indices[^t.Rank..].CopyTo(srcIndices);
                result[i] = t[t.LinearIndex(srcIndices)];
            }
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = Broadcast(t) -> grad_t = sum(broadcasted axes) [grad_r]
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Shortcut for 1D input
                if (t.Rank == 1)
                {
                    int stride = t.ElementCount;
                    int rows = result.ElementCount / stride;

                    // Vectorize gradients
                    var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());

                    // Calculate gradients for parent per broadcasted dimension
                    for (int r = 0; r < rows; r++)
                    {
                        // Vectorize gradients
                        var rgSlice = result.Grad.AsSpan(r * stride, stride);
                        var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(rgSlice);

                        // Calculate vectorized gradients for parent
                        for (int i = 0; i < tgVecs.Length; i++)
                        {
                            tgVecs[i] += rgVecs[i];
                        }

                        // Clean up unvectorized tail
                        for (int i = tgVecs.Length * VectorSize; i < stride; i++)
                        {
                            t.Grad[i] += rgSlice[i];
                        }
                    }
                }
                else
                {
                    // Preallocate index mappings on stack - avoid garbage collector
                    Span<int> indices = stackalloc int[result.Rank];
                    Span<int> srcIndices = stackalloc int[t.Rank];

                    // Calculate gradients for parent sequentially
                    for (int i = 0; i < result.ElementCount; i++)
                    {
                        result.GetFullIndices(i, indices);
                        indices[^t.Rank..].CopyTo(srcIndices);
                        t.Grad[t.LinearIndex(srcIndices)] += result.Grad[i];
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Flattens a tensor beyond the given axis.
    /// </summary>
    /// <param name="t">Tensor to flatten.</param>
    /// <param name="startAxis">First axis to flatten.</param>
    /// <returns>Result tensor of flattening the input tensor beyond the given axis.</returns>
    public static Tensor Flatten(Tensor t, int startAxis = 0)
    {
        // Calculate size of flat dimension
        int flatSize = 1;
        for (int i = startAxis; i < t.Rank; i++) flatSize *= t.Dimensions[i];

        // Build new dimensions
        var newDims = new int[startAxis + 1];
        for (int i = 0; i < startAxis; i++) newDims[i] = t.Dimensions[i];
        newDims[^1] = flatSize;

        return Reshape(t, newDims); // use general reshape function
    }

    /// <summary>
    /// Reshapes a tensor into the given dimensions without rearranging linear data.
    /// </summary>
    /// <param name="t">Tensor to reshape.</param>
    /// <param name="newDims">Dimensions to reshape to.</param>
    /// <returns>Result tensor of reshaping the input tensor into the given dimensions.</returns>
    public static Tensor Reshape(Tensor t, int[] newDims)
    {
        Tensor result = GetResultTensor(t, newDims, t.RequiresGrad);

        Array.Copy(t.Data, result.Data, t.ElementCount); // copy linear data

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = Reshape(t) -> dr_i/dt_i = 1
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < tgVecs.Length; i++)
                {
                    tgVecs[i] += rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                {
                    t.Grad[i] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Wraps a tensor into a new batch with 1 experience entry.
    /// </summary>
    /// <param name="t">Tensor to wrap into batch.</param>
    /// <returns>Batch tensor containing the input tensor.</returns>
    public static Tensor WrapBatch(Tensor t)
    {
        // Add batch dimension
        var batchDims = new int[t.Rank + 1];
        batchDims[0] = 1;
        Array.Copy(t.Dimensions, 0, batchDims, 1, t.Rank);

        Tensor batch = GetResultTensor(t, batchDims, t.RequiresGrad);

        Array.Copy(t.Data, batch.Data, t.ElementCount); // copy linear data

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            batch._parents.Add(t);

            // Gradient calculation function for b = WrapBatch(t) -> db_i/dt_i = 1
            batch._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(batch.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < tgVecs.Length; i++)
                {
                    tgVecs[i] += bgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                {
                    t.Grad[i] += batch.Grad[i];
                }
            };
        }

        return batch;
    }

    /// <summary>
    /// Clips a tensor's values to be within a specified range.
    /// </summary>
    /// <param name="t">Tensor to clip.</param>
    /// <param name="min">Minimum value to clip to.</param>
    /// <param name="max">Maximum value to clip to.</param>
    /// <returns>Result tensor containing the input tensor's values clipped to between the given min and max.</returns>
    public static Tensor Clip(Tensor t, double min, double max)
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var vmin = new Vector<double>(min); // splat scalar into vector
        var vmax = new Vector<double>(max); // splat scalar into vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized clamp result
        for (int i = 0; i < tVecs.Length; i++)
        {
            rVecs[i] = Vector.Min(Vector.Max(tVecs[i], vmin), vmax);
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Clamp(t[i], min, max);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function r = Clip(t) -> dr/dt = 1 (min < t < max), 0 (t <= min or t >= max)
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize inputs and gradients
                var tvVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var vmin = new Vector<double>(min); // splat scalar into vector
                var vmax = new Vector<double>(max); // splat scalar into vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < tgVecs.Length; i++)
                {
                    var inRange = Vector.BitwiseAnd(
                        Vector.GreaterThanOrEqual(tvVecs[i], vmin),
                        Vector.LessThanOrEqual(tvVecs[i], vmax));
                    tgVecs[i] += Vector.ConditionalSelect(inRange, rgVecs[i], Vector<double>.Zero);
                }

                // Clean up unvectorized tail
                for (int i = tgVecs.Length * VectorSize; i < t.GradCount; i++)
                {
                    t.Grad[i] += (t[i] >= min && t[i] <= max) ? result.Grad[i] : 0;
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Creates a dropout mask for a dense layer with the given dropout rate.
    /// </summary>
    /// <param name="result">Result tensor to apply dropout to.</param>
    /// <param name="dropout">Dropout rate to apply.</param>
    /// <returns>Dropout mask for the given result tensor.</returns>
    public static Tensor GetDenseDropoutMask(Tensor result, double dropout)
    {
        // Do not apply dropout if in inference mode
        if (Inference) return Scalar(1.0, result.Dimensions, requiresGrad: false);

        Tensor mask = new(result.Dimensions, requiresGrad: false);
        double scale = 1.0 / (1.0 - dropout);

        // Randomly select each element to be dropped or not based on dropout rate
        for (int i = 0; i < mask.ElementCount; i++)
        {
            double rand = Random.Shared.NextDouble();
            mask[i] = rand < dropout ? 0.0 : scale;
        }
        return mask;
    }

    /// <summary>
    /// Creates a spatial dropout mask for a convolutional layer with the given dropout rate.
    /// </summary>
    /// <param name="result">Result tensor to apply dropout to.</param>
    /// <param name="dropout">Dropout rate to apply.</param>
    /// <returns>Dropout mask for the given result tensor.</returns>
    public static Tensor GetSpatialDropoutMask(Tensor result, double dropout)
    {
        // Do not apply dropout if in inference mode
        if (Inference) return Scalar(1.0, result.Dimensions, requiresGrad: false);

        Tensor mask = new(result.Dimensions, requiresGrad: false);
        double scale = 1.0 / (1.0 - dropout);

        // Compute strides and sizes
        int batches = result.Dimensions[0];
        int batchSize = result.ElementCount / batches;
        int channels = result.Dimensions[^1];
        int spatialSize = batchSize / channels;

        var channelVals = new double[channels];
        // Apply dropout independently to each batch
        for (int b = 0; b < batches; b++)
        {
            int batchOffset = b * batchSize;

            // Select which channels to drop for the given batch
            for (int c = 0; c < channels; c++)
            {
                channelVals[c] = Random.Shared.NextDouble() < dropout ? 0.0 : scale;
            }

            // Apply dropout to each spatial position
            for (int s = 0; s < spatialSize; s++)
            {
                int spatialOffset = batchOffset + (s * channels);

                // Apply dropout to each channel in the current spatial position
                for (int c = 0; c < channels; c++)
                {
                    mask[spatialOffset + c] = channelVals[c];
                }
            }
        }

        return mask;
    }

    /// <summary>
    /// Gets the tensor instance to write the result of the current function/operation to.
    /// </summary>
    /// <param name="owner">Tensor being used as an input in the function/operation.</param>
    /// <param name="dims">Dimensions of the result tensor.</param>
    /// <param name="requiresGrad">Whether it is necessary to calculate the result tensor's gradient.</param>
    /// <returns>Tensor instance into which to write the result of the function/operation.</returns>
    static Tensor GetResultTensor(Tensor owner, int[] dims, bool requiresGrad)
    {
        // Prepare input tensor for forward pass of the operation
        owner.PrepareForward();

        // Whether the operation should already be represented in the autograd graph
        bool newOp = owner._results.Count <= owner._opIndex;

        Tensor result;
        if (newOp) // create new tensor instance to write result into
        {
            result = new(dims, requiresGrad);
            owner._results.Add(result);
        }
        else
        {
            // Find tensor instance used to store result of operation with the same instance during last forward pass
            result = owner._results[owner._opIndex];

            // Check whether the stored instance has the required dimensions
            bool shapeMismatch = result.Rank != dims.Length;
            if (!shapeMismatch)
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    if (result.Dimensions[i] != dims[i])
                    {
                        shapeMismatch = true;
                        break;
                    }
                }
            }

            // Create new tensor instance if stored instance does not match required dimensions
            if (shapeMismatch)
            {
                result = new(dims, requiresGrad);
                owner._results[owner._opIndex] = result;
            }
            else
            {
                // Clear out previous autograd graph data from stored instance
                result._parents.Clear();
                result._backward = delegate { };
            }
        }

        owner._opIndex++;

        return result;
    }
}
