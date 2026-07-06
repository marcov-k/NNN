using NNNCSharp.Components.Activations;
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities.SaveSystem;

namespace NNNCSharp.Components.Models.Layers;

/// <summary>
/// Fully connected neural network layer.
/// </summary>
public class Dense : Layer
{
    /// <summary>
    /// Number of neurons in the layer.
    /// </summary>
    public int NeuronCount { get; private set; }
    /// <summary>
    /// Tensor containing the weights parameters of the layer.
    /// </summary>
    public Tensor Weights { get; private set; } = new();
    /// <summary>
    /// Whether the layer flattens its input prior to applying weights.
    /// </summary>
    public bool Flatten { get; private set; }

    /// <summary>
    /// Creates a new dense neural network layer instance.
    /// </summary>
    /// <param name="neuronCount">Number of neurons in the new layer.</param>
    /// <param name="activation">Activation function of the new layer.</param>
    /// <param name="dropout">Dropout rate of the new layer.</param>
    public Dense(int neuronCount, Activation activation, bool flatten = false, double dropout = 0.0)
    {
        NeuronCount = neuronCount;
        Activation = activation;
        Flatten = flatten;
        Dropout = dropout;
    }

    /// <summary>
    /// Creates a new dense neural network layer instance.
    /// </summary>
    /// <param name="neuronCount">Number of neurons in the new layer.</param>
    /// <param name="weights">Weights tensor of the new layer.</param>
    /// <param name="biases">Bias tensor of the new layer.</param>
    /// <param name="activation">Activation function of the new layer.</param>
    /// <param name="dropout">Dropout rate of the new layer.</param>
    public Dense(int neuronCount, Tensor weights, Tensor biases, Activation activation, bool flatten, double dropout)
    {
        NeuronCount = neuronCount;
        Weights = weights;
        Biases = biases;
        Activation = activation;
        Flatten = flatten;
        Dropout = dropout;
    }

    /// <summary>
    /// Parameterless constructor for model reconstruction from save data.
    /// </summary>
    public Dense() { }

    // Base Layer API overrides

    public override void SetUpLayer(Tensor inputFormat)
    {
        Weights = Tensor.InitWeights(inputFormat.ElementCount, NeuronCount);
        Biases = Tensor.InitBiases(NeuronCount);
        OutputFormat = new([1, NeuronCount]);
    }

    public override Tensor Forward(Tensor input)
    {
        var output = Flatten ? Tensor.Flatten(input, 1) : input;
        output ^= Weights;
        output += Tensor.Broadcast(Biases, output.Dimensions.ToArray());
        output = Activation.Forward(output);

        if (Dropout > 0.0)
        {
            output *= Tensor.GetDenseDropoutMask(output, Dropout);
        }

        return output;
    }

    public override IEnumerable<Tensor> GetParameters()
    {
        yield return Weights;
        yield return Biases;
    }

    public override Layer Copy()
    {
        return new Dense(NeuronCount, Weights.Copy(), Biases.Copy(), Activation.Copy(), Flatten, Dropout);
    }

    public override void WriteUniqueData(FileStream stream)
    {
        FileUtils.WriteInt32(stream, NeuronCount);
        FileUtils.WriteTensor(stream, Weights);
        FileUtils.WriteBool(stream, Flatten);
    }

    protected override void ReadUniqueData(FileStream stream)
    {
        NeuronCount = FileUtils.ReadInt32(stream);
        Weights = FileUtils.ReadTensor(stream);
        Flatten = FileUtils.ReadBool(stream);
    }

    protected override string PrintUniqueLayer(FileStream stream)
    {
        int neuronCount = FileUtils.ReadInt32(stream);
        string weights = $"Weights Tensor: {FileUtils.PrintTensor(stream)}";
        bool flatten = FileUtils.ReadBool(stream);
        return $"Neurons: {neuronCount}\n{weights}\nFlatten: {flatten}";
    }

    public override void Dispose()
    {
        base.Dispose();
        Weights.Dispose();
    }
}
