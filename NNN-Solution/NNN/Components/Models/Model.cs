using NNN.Components.Autodiff;
using NNN.Components.Models.Layers;
using NNN.Components.Utilities.SaveSystem;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNN.Components.Models;

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
    /// List of all layer parameters.
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
    /// Number of parameters across all layers.
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
    /// Constructs a new model instance from the given save data.
    /// </summary>
    /// <param name="data">Save data representing the model architecture and training.</param>
    public Model(ModelData data)
    {
        Layers = new Layer[data.Layers.Length];

        BuildFromData(data);
    }

    /// <summary>
    /// Reconstructs the model to match the architecture and training of the model stored in the save data.
    /// </summary>
    /// <param name="data"></param>
    void BuildFromData(ModelData data)
    {
        InvalidateParameters(); // ensure any previous parameters are cleared

        // Reconstruct each layer from the save data
        LayerData layerData;
        Type? layerType;
        for (int i = 0; i < data.Layers.Length; i++)
        {
            layerData = data.Layers[i];

            // Create new instance of saved layer type
            layerType = Type.GetType(layerData.LayerName);
            if (layerType is not null)
            {
                var layer = Activator.CreateInstance(layerType) as Layer;
                layer?.BuildFromData(layerData);
                if (layer is not null) Layers[i] = layer;
            }
        }
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
        // Calculate total magnitude of gradients
        double totalNorm = 0.0;
        foreach (var param in Parameters)
        {
            var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(param.Grad.AsSpan());
            var acc = Vector<double>.Zero;
            for (int i = 0; i < gradVecs.Length; i++)
            {
                acc += gradVecs[i] * gradVecs[i];
            }
            totalNorm += Vector.Sum(acc);

            for (int i = gradVecs.Length * VectorSize; i < param.GradCount; i++)
            {
                double grad = param.Grad[i];

                totalNorm += grad * grad;
            }
        }
        totalNorm = Math.Sqrt(totalNorm);

        // Scale (clip) gradients if necessary
        if (totalNorm > maxNorm)
        {
            double scale = maxNorm / (totalNorm + 1e-8);
            var scaleVec = new Vector<double>(scale);

            foreach (var param in Parameters)
            {
                var gradVecs = MemoryMarshal.Cast<double, Vector<double>>(param.Grad.AsSpan());
                for (int i = 0; i < gradVecs.Length; i++)
                {
                    gradVecs[i] *= scaleVec;
                }

                for (int i = gradVecs.Length * VectorSize; i < param.GradCount; i++)
                {
                    param.Grad[i] *= scale;
                }
            }
        }
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

    /// <summary>
    /// Clears the model's current stored parameter references.
    /// </summary>
    void InvalidateParameters() => _parameters = null;
}
