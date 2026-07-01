using NNNCSharp.Components.Interop;
using NNNCSharp.Components.Models.Layers;
using NNNCSharp.Components.Utilities.SaveSystem;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Models;

/// <summary>
/// Neural network model class.
/// </summary>
public class Model
{
    // Externally accessible properties
    /// <summary>
    /// Hidden layers and output layer of the model.
    /// </summary>
    public Layer[] Layers { get; private set; }
    /// <summary>
    /// List of all layer parameter tensors.
    /// </summary>
    public List<Tensor> Parameters
    {
        get
        {
            _parameters ??= [.. GetParameters()]; // initialize internal property if not yet initialized
            return _parameters;
        }
    }
    /// <summary>
    /// Number of parameter tensors across all layers.
    /// </summary>
    public int ParameterCount => Parameters.Count;

    // Internal properties
    /// <summary>
    /// Internal list of all player parameters.
    /// </summary>
    List<Tensor>? _parameters;

    // Utilities
    static readonly int VectorSize = Vector<double>.Count;

    /// <summary>
    /// Initializes a new model with the given layers.
    /// </summary>
    /// <param name="layers">List of hidden and output layers.</param>
    public Model(Layer[] layers)
    {
        Layers = layers;
    }

    /// <summary>
    /// Initializes a new model with the given layers and initializes parameters for the given input format.
    /// </summary>
    /// <param name="layers">List of hidden and output layers.</param>
    /// <param name="inputFormat">Tensor representing the expected input format.</param>
    public Model(Layer[] layers, Tensor inputFormat)
    {
        Layers = layers;
        SetUpLayers(inputFormat);
    }

    /// <summary>
    /// Initializes layer parameters to support the given input format.
    /// </summary>
    /// <param name="inputFormat">Tensor representing the expected input format.</param>
    public void SetUpLayers(Tensor inputFormat)
    {
        var format = inputFormat.Copy();

        // Initialize parameters of each layer
        foreach (var layer in Layers)
        {
            layer.SetUpLayer(format);
            format = layer.OutputFormat; // every subsequent layer receives output dimensions of the previous layer
        }
    }

    /// <summary>
    /// Sends the input forward through the neural network while building autograd graph.
    /// </summary>
    /// <param name="input">Input tensor to be processed.</param>
    /// <returns>Tensor containing the model's predicted outputs.</returns>
    public Tensor Forward(Tensor input)
    {
        Tensor.BeginForward();

        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Sends the input forward through the neural network without building an autograd graph.
    /// </summary>
    /// <param name="input">Input tensor to be processed.</param>
    /// <returns>Tensor containing the model's predicted outputs.</returns>
    public Tensor Predict(Tensor input)
    {
        Tensor.Inference = true;
        var output = Forward(input);
        Tensor.Inference = false;
        return output;
    }

    /// <summary>
    /// Gets all of the parameters in the model's hidden layers and output layer.
    /// </summary>
    /// <returns>IEnumerable containing all parameter tensors.</returns>
    public IEnumerable<Tensor> GetParameters()
    {
        foreach (var layer in Layers)
        {
            foreach (var param in layer.GetParameters())
            {
                yield return param;
            }
        }
    }

    /// <summary>
    /// Clips all of the model's parameters if necessary.
    /// </summary>
    /// <param name="maxNorm">Maximum total magnitude of gradients without clipping.</param>
    public void ClipGradients(double maxNorm)
    {
        var handles = Parameters.Select(p => p.Handle).ToArray();
        NativeMethods.optimizers_clip_gradients(handles, handles.Length, maxNorm);
    }

    /// <summary>
    /// Gets the total number of parameter values across all layers.
    /// </summary>
    /// <returns>Total number of parameter values.</returns>
    public ulong GetTotalParameterSize()
    {
        ulong paramSize = 0;
        foreach (var param in Parameters)
        {
            paramSize += (ulong)param.ElementCount;
        }
        return paramSize;
    }

    /// <summary>
    /// Creates an identical deep copy of the model.
    /// </summary>
    /// <returns>New model instance with identical layers and parameters.</returns>
    public Model Copy()
    {
        var layers = new Layer[Layers.Length];

        for (int i = 0; i < Layers.Length; i++)
        {
            layers[i] = Layers[i].Copy();
        }

        return new(layers);
    }
}
