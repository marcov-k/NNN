using NNN.Components.Autodiff;
using NNN.Components.Utilities;

namespace NNN.Components.Buffers;

/// <summary>
/// Buffer for creating batches during standard supervised training.
/// </summary>
/// <param name="data">Full training inputs.</param>
/// <param name="targets">Full training targets.</param>
public class BatchBuffer(Tensor[] data, Tensor[] targets)
{
    /// <summary>
    /// Array of all training inputs.
    /// </summary>
    readonly Tensor[] Data = data;
    /// <summary>
    /// Array of all training targets.
    /// </summary>
    readonly Tensor[] Targets = targets;
    /// <summary>
    /// Dimensions of an unbatched training input.
    /// </summary>
    readonly int[] DataDims = data[0].Dimensions;
    /// <summary>
    /// Dimensions of an unbatched training target.
    /// </summary>
    readonly int[] TargetDims = targets[0].Dimensions;

    /// <summary>
    /// Randomly selects a training batch from the full training inputs and targets.
    /// </summary>
    /// <param name="batchSize">Number of input and target pairs to include in the batch.</param>
    /// <returns>Batched input and corresponding target tensors.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Batch size outside valid range.</exception>
    public (Tensor batchInputs, Tensor batchTargets) GetBatch(int batchSize)
    {
        if (batchSize <= 0 || batchSize > Data.Length) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size out of range.");

        // Compute batched input tensor dimensions
        var batchDims = new int[Data[0].Rank + 1];
        batchDims[0] = batchSize;
        Array.Copy(DataDims, 0, batchDims, 1, DataDims.Length);

        Tensor batchInputs = new(batchDims);

        // Fill batched input tensor with randomly selected inputs
        int itemLength = Data[0].ElementCount;
        var batchItems = ArrayUtils.GetRandomElements(Data, batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(batchItems[i].Element.Data, 0, batchInputs.Data, i * itemLength, itemLength);
        }

        // Compute batched target tensor dimensions
        var targetDims = new int[Targets[0].Rank + 1];
        targetDims[0] = batchSize;
        Array.Copy(TargetDims, 0, targetDims, 1, TargetDims.Length);

        Tensor batchTargets = new(targetDims);

        // Fill batched target tensor with targets corresponding to selected inputs
        int targetLength = Targets[0].ElementCount;
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(Targets[batchItems[i].OriginalIndex].Data, 0, batchTargets.Data, i * targetLength, targetLength);
        }

        return (batchInputs, batchTargets);
    }
}
