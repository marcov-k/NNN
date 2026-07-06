using NNNCSharp.Components.Activations;
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities;
using NNNCSharp.Components.Utilities.SaveSystem;

namespace NNNCSharp.Components.Models.Layers;

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
    /// <param name="dropout">Dropout rate of the new layer.</param>
    public Conv(int filterCount, int[] kernelDims, Activation activation, double dropout = 0.0)
    {
        FilterCount = filterCount;
        KernelDims = kernelDims;
        Activation = activation;
        Dropout = dropout;
    }

    /// <summary>
    /// Creates a new convolutional layer instance.
    /// </summary>
    /// <param name="filterCount">Number of filters in the layer.</param>
    /// <param name="kernelDims">Dimensions of the kernels used by the layer.</param>
    /// <param name="kernels">Kernels tensor of the new layer.</param>
    /// <param name="biases">Bias tensor of the new layer.</param>
    /// <param name="activation">Activation function of the new layer.</param>
    /// <param name="dropout">Dropout rate of the new layer.</param>
    public Conv(int filterCount, int[] kernelDims, Tensor kernels, Tensor biases, Activation activation, double dropout)
    {
        FilterCount = filterCount;
        KernelDims = kernelDims;
        Kernels = kernels;
        Biases = biases;
        Activation = activation;
        Dropout = dropout;
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
        var output = Tensor.Convolve(input, Kernels);
        output += Tensor.Broadcast(Biases, output.Dimensions.ToArray());
        output = Activation.Forward(output);

        if (Dropout > 0.0) output *= Tensor.GetSpatialDropoutMask(output, Dropout);

        return output;
    }

    public override IEnumerable<Tensor> GetParameters()
    {
        yield return Kernels;
        yield return Biases;
    }

    public override Layer Copy()
    {
        return new Conv(FilterCount, [.. KernelDims], Kernels.Copy(), Biases.Copy(), Activation.Copy(), Dropout);
    }

    public override void WriteUniqueData(FileStream stream)
    {
        FileUtils.WriteInt32(stream, FilterCount);
        FileUtils.WriteTensor(stream, Kernels);
    }

    protected override void ReadUniqueData(FileStream stream)
    {
        FilterCount = FileUtils.ReadInt32(stream);
        Kernels = FileUtils.ReadTensor(stream);
        KernelDims = Kernels.Dimensions[1..^1].ToArray();
    }

    protected override string PrintUniqueLayer(FileStream stream)
    {
        int filterCount = FileUtils.ReadInt32(stream);
        var kernels = FileUtils.ReadTensor(stream);
        var kernelDims = kernels.Dimensions[1..^1];

        string kernelDimsString = "Kernel Dimensions: [";
        for (int i = 0; i < kernelDims.Length; i++)
        {
            kernelDimsString += kernelDims[i];
            if (i < kernelDims.Length - 1) kernelDimsString += ", ";
        }
        kernelDimsString += "]";

        string kernelsString = "Kernels Tensor: Dimensions: [";
        for (int i = 0; i < kernels.Dimensions.Length; i++)
        {
            kernelsString += kernels.Dimensions[i];
            if (i < kernels.Dimensions.Length - 1) kernelsString += ", ";
        }
        kernelsString += $"], # of parameter values: {kernels.ElementCount}";

        return $"Filters: {filterCount}\n{kernelDimsString}\n{kernelsString}";
    }

    public override void Dispose()
    {
        base.Dispose();
        Kernels.Dispose();
    }
}
