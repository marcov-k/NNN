using NNN.Components.Autodiff;

namespace NNN.Components.Utilities.SaveSystem;

/// <summary>
/// Class for storing save data for a single layer of a trained model.
/// </summary>
/// <param name="neuronCount">Number of neurons in the layer.</param>
/// <param name="layerName">Name of the specific layer subclass.</param>
/// <param name="weights">Weights parameter of the layer.</param>
/// <param name="biases">Bias parameter of the layer.</param>
/// <param name="activation">Name of the activation function of the layer.</param>
[Serializable]
public class LayerData(string layerName, string activation, Tensor biases, int? neuronCount = null, Tensor? weights = null,
    int? filterCount = null, int[]? kernelDims = null, Tensor? kernels = null)
{
    /// <summary>
    /// Name of the specific layer subclass.
    /// </summary>
    public string LayerName { get; set; } = layerName;
    /// <summary>
    /// Name of the activation function of the layer.
    /// </summary>
    public string Activation { get; set; } = activation;
    /// <summary>
    /// Bias parameter of the layer.
    /// </summary>
    public Tensor Biases { get; set; } = biases;
    /// <summary>
    /// Number of neurons in the layer.
    /// </summary>
    public int? NeuronCount { get; set; } = neuronCount;
    /// <summary>
    /// Weights parameter of the layer.
    /// </summary>
    public Tensor? Weights { get; set; } = weights;
    /// <summary>
    /// Number of filters used by the layer.
    /// </summary>
    public int? FilterCount { get; set; } = filterCount;
    /// <summary>
    /// Dimensions of the layer's kernels.
    /// </summary>
    public int[]? KernelDims { get; set; } = kernelDims;
    /// <summary>
    /// Kernels parameter of the layer.
    /// </summary>
    public Tensor? Kernels { get; set; } = kernels;
}
