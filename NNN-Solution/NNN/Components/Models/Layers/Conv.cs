using NNN.Components.Activations;
using NNN.Components.Autodiff;
using NNN.Components.Utilities.SaveSystem;

namespace NNN.Components.Models.Layers;

/// <summary>
/// Convolutional neural network layer.
/// </summary>
public class Conv : Layer
{
    /// <summary>
    /// Number of filters in the layer.
    /// </summary>
    public int FilterCount { get; private set; }
    /// <summary>
    /// Dimensions of the kernels used by the layer.
    /// </summary>
    public int[] KernelDims { get; private set; } = [];
    /// <summary>
    /// Tensor storing the kernels of the layer.
    /// </summary>
    public Tensor Kernels { get; private set; } = new();

    /// <summary>
    /// Creates a new convolutional layer instance.
    /// </summary>
    /// <param name="filterCount">Number of filters in the layer.</param>
    /// <param name="kernelDims">Dimensions of the kernels used by the layer.</param>
    /// <param name="activation">Activation function of the new layer.</param>
    public Conv(int filterCount, int[] kernelDims, Activation activation)
    {
        FilterCount = filterCount;
        KernelDims = kernelDims;
        Activation = activation;
    }

    /// <summary>
    /// Creates a new convolutional layer instance.
    /// </summary>
    /// <param name="filterCount">Number of filters in the layer.</param>
    /// <param name="kernelDims">Dimensions of the kernels used by the layer.</param>
    /// <param name="kernels">Kernels tensor of the new layer.</param>
    /// <param name="biases">Bias tensor of the new layer.</param>
    /// <param name="activation">Activation function of the new layer.</param>
    public Conv(int filterCount, int[] kernelDims, Tensor kernels, Tensor biases, Activation activation)
    {
        FilterCount = filterCount;
        KernelDims = kernelDims;
        Kernels = kernels;
        Biases = biases;
        Activation = activation;
    }

    /// <summary>
    /// Parameterless constructor for model reconstruction from save data.
    /// </summary>
    public Conv() { }

    // Base Layer API overrides

    public override void SetUpLayer(Tensor inputFormat)
    {
        // Initialize parameters
        Kernels = Tensor.InitKernels(FilterCount, KernelDims, inputFormat.Dimensions[^1]);
        Biases = Tensor.InitBiases(FilterCount);

        // Compute output dimensions
        var outputDims = new int[inputFormat.Rank];
        outputDims[0] = 1;
        for (int i = 0; i < KernelDims.Length; i++)
        {
            outputDims[i + 1] = inputFormat.Dimensions[i + 1] - KernelDims[i] + 1;
        }
        outputDims[^1] = FilterCount;
        OutputFormat = new(outputDims);
    }

    public override Tensor Forward(Tensor input)
    {
        // Apply convolution to the input
        var output = Tensor.Convolve(input, Kernels, Biases);
        return Activation.Forward(output);
    }

    public override IEnumerable<Tensor> GetParameters()
    {
        yield return Kernels;
        yield return Biases;
    }

    public override Layer Copy()
    {
        return new Conv(FilterCount, [.. KernelDims], Kernels.Copy(), Biases.Copy(), Activation.Copy());
    }

    public override void BuildFromData(LayerData data)
    {
        if (data.FilterCount is not null) FilterCount = data.FilterCount.Value;
        if (data.KernelDims is not null) KernelDims = data.KernelDims;

        if (data.Kernels is not null) Kernels = data.Kernels;
        Kernels.RestoreGrad();

        if (data.Biases is not null) Biases = data.Biases;
        Biases.RestoreGrad();

        var activType = Type.GetType(data.Activation);
        if (activType is not null)
        {
            Activation = Activator.CreateInstance(activType) as Activation ?? new Linear();
        }
    }
}
