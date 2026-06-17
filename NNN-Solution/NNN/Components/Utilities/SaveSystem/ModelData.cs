namespace NNN.Components.Utilities.SaveSystem;

/// <summary>
/// Class for storing save data for a trained model.
/// </summary>
/// <param name="layers">Array of save data for layers constituting the model.</param>
[Serializable]
public class ModelData(LayerData[] layers)
{
    /// <summary>
    /// Array of save data for layers constituting the model.
    /// </summary>
    public LayerData[] Layers { get; set; } = layers;
}
