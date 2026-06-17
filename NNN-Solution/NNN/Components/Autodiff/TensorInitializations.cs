using NNN.Components.Utilities;

namespace NNN.Components.Autodiff;

public partial class Tensor
{
    // Initialization functions

    /// <summary>
    /// Creates a new tensor instance with the given dimensions.
    /// </summary>
    /// <param name="dims">Array containing the dimensions of the new tensor.</param>
    /// <param name="requiresGrad">Whether it is necessary to calculate the tensor's gradient.</param>
    public Tensor(int[] dims, bool requiresGrad = false)
    {
        // Calculate shape data
        Dimensions = (int[])dims.Clone();
        Strides = ComputeStrides(dims);

        // Calculate linear size
        int size = 1;
        foreach (var dim in dims) size *= dim;

        Data = new double[size]; // initialize linear array of values

        // Initialize linear array of gradients if necessary
        RequiresGrad = requiresGrad;
        if (RequiresGrad) Grad = new double[size];
    }

    /// <summary>
    /// Parameterless constructor for Json serializer.
    /// </summary>
    public Tensor() { }

    /// <summary>
    /// Initializes a new weights tensor using He Initialization.
    /// </summary>
    /// <param name="inputCount">Number of inputs to the layer.</param>
    /// <param name="neuronCount">Number of neurons in the layer.</param>
    /// <returns>Weights tensor with values initialized using He Initialization.</returns>
    public static Tensor InitWeights(int inputCount, int neuronCount)
    {
        Tensor weights = new([inputCount, neuronCount], true);

        double stdDev = Math.Sqrt(2.0 / inputCount);
        for (int i = 0; i < weights.ElementCount; i++)
        {
            weights[i] = MathUtils.NextGaussian(0, stdDev);
        }

        return weights;
    }

    /// <summary>
    /// Initializes a new bias tensor with non-zero values.
    /// </summary>
    /// <param name="neuronCount">Number of neurons in the layer.</param>
    /// <returns>Bias tensor with non-zero values.</returns>
    public static Tensor InitBiases(int neuronCount) => Scalar(0.01, [neuronCount], true);

    public static Tensor InitKernels(int filterCount, int[] kernelDims, int inputChannels)
    {
        var dims = new int[kernelDims.Length + 2];
        dims[0] = filterCount;
        Array.Copy(kernelDims, 0, dims, 1, kernelDims.Length);
        dims[^1] = inputChannels;

        int fanIn = inputChannels;
        foreach (var dim in kernelDims)
        {
            fanIn *= dim;
        }

        Tensor kernels = new(dims, true);

        double stdDev = Math.Sqrt(2.0 / fanIn);
        for (int i = 0; i < kernels.ElementCount; i++)
        {
            kernels[i] = MathUtils.NextGaussian(0, stdDev);
        }

        return kernels;
    }

    /// <summary>
    /// Creates a tensor instance filled with a single scalar value.
    /// </summary>
    /// <param name="value">Scalar value to fill the new tensor with.</param>
    /// <param name="dims">Dimensions of the new tensor.</param>
    /// <param name="requiresGrad">Whether it is necessary to calculate the gradient of the new tensor.</param>
    /// <returns>New tensor instance filled with the given scalar value.</returns>
    public static Tensor Scalar(double value, int[] dims, bool requiresGrad = false)
    {
        Tensor t = new(dims, requiresGrad);
        Array.Fill(t.Data, value);
        return t;
    }

    /// <summary>
    /// Creates a deep copy of the current tensor.
    /// </summary>
    /// <returns>Deep copy tensor detached from the existing autograd graph.</returns>
    public Tensor Copy()
    {
        Tensor copy = new(Dimensions, false);
        Array.Copy(Data, copy.Data, ElementCount);
        return copy;
    }
}
