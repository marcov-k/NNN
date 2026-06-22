using NNN.Components.Activations;
using NNN.Components.Models.Layers;

namespace NNN.Components.Utilities;

/// <summary>
/// Static class storing IDs for saving and loading.
/// </summary>
public static class IDManager
{
    /// <summary>
    /// IDs for Layer subclasses.
    /// </summary>
    static readonly Type[] LayerIDs = [typeof(Dense), typeof(Conv)];
    /// <summary>
    /// IDs for Activation subclasses.
    /// </summary>
    static readonly Type[] ActivationIDs = [typeof(Linear), typeof(Sigmoid),
        typeof(Tanh), typeof(ReLU), typeof(LeakyReLU), typeof(Softmax)];

    /// <summary>
    /// Gets the byte ID of a layer.
    /// </summary>
    /// <param name="layer">Layer to get ID of.</param>
    /// <returns>Byte ID of the given layer.</returns>
    public static byte GetLayerID(Layer layer) => (byte)LayerIDs.IndexOf(layer.GetType());

    /// <summary>
    /// Gets the layer type with the given ID.
    /// </summary>
    /// <param name="id">ID of the layer type.</param>
    /// <returns>Type of the layer with the given ID.</returns>
    public static Type GetLayerByID(byte id) => LayerIDs[id];

    /// <summary>
    /// Gets the byte ID of an activation.
    /// </summary>
    /// <param name="activation">Activation to get ID of.</param>
    /// <returns>Byte ID of the given activation.</returns>
    public static byte GetActivationID(Activation activation) => (byte)ActivationIDs.IndexOf(activation.GetType());

    /// <summary>
    /// Gets the activation type with the given ID.
    /// </summary>
    /// <param name="id">ID of the activation type.</param>
    /// <returns>Type of the activation with the given ID.</returns>
    public static Type GetActivationByID(byte id) => ActivationIDs[id];
}
