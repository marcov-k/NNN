using NNNCSharp.Components.Interop;
using NNNCSharp.Components.Utilities;

namespace NNNCSharp.Components.Buffers;

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
    readonly int[] DataDims = data[0].Dimensions.ToArray();
    /// <summary>
    /// Dimensions of an unbatched training target.
    /// </summary>
    readonly int[] TargetDims = targets[0].Dimensions.ToArray();
    /// <summary>
    /// Persistent array of batch input tensors.
    /// </summary>
    Tensor[]? BatchInputs = null;
    /// <summary>
    /// Persistent array of batch target tensors.
    /// </summary>
    Tensor[]? BatchTargets = null;

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

        // Compute batched target tensor dimensions
        var targetDims = new int[Targets[0].Rank + 1];
        targetDims[0] = batchSize;
        Array.Copy(TargetDims, 0, targetDims, 1, TargetDims.Length);

        // Allocate persistent batch arrays if not yet allocated
        if (BatchInputs is null)
        {
            BatchInputs = new Tensor[1];
            BatchInputs[0] = new(batchDims);
        }
        if (BatchTargets is null)
        {
            BatchTargets = new Tensor[1];
            BatchTargets[0] = new(targetDims);
        }

        // Fill batched input tensor with randomly selected inputs
        int itemLength = Data[0].ElementCount;
        var batchItems = ArrayUtils.GetRandomElements(Data, batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            batchItems[i].Element.Data[0..itemLength].CopyTo(BatchInputs[0].Data.Slice(i * itemLength, itemLength));
        }

        // Fill batched target tensor with targets corresponding to selected inputs
        int targetLength = Targets[0].ElementCount;
        for (int i = 0; i < batchSize; i++)
        {
            Targets[batchItems[i].OriginalIndex].Data[0..targetLength].CopyTo(BatchTargets[0].Data.Slice(i * targetLength, targetLength));
        }

        return (BatchInputs[0], BatchTargets[0]);
    }

    public (Tensor[] BatchInputs, Tensor[] BatchTargets) GetBatches(int batchSize)
    {
        if (batchSize <= 0 || batchSize > Data.Length) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size out of range.");

        var shuffledData = ArrayUtils.GetRandomElements(Data, Data.Length);
        var shuffledTargets = new Tensor[shuffledData.Length];
        for (int i = 0; i < shuffledData.Length; i++)
        {
            shuffledTargets[i] = Targets[shuffledData[i].OriginalIndex];
        }

        var batchInputDims = new int[Data[0].Rank + 1];
        batchInputDims[0] = batchSize;
        Array.Copy(DataDims, 0, batchInputDims, 1, DataDims.Length);

        var batchTargetDims = new int[Targets[0].Rank + 1];
        batchTargetDims[0] = batchSize;
        Array.Copy(TargetDims, 0, batchTargetDims, 1, TargetDims.Length);

        int fullBatchCount = Data.Length / batchSize;
        int tailBatchLength = Data.Length % batchSize;
        int batchCount = tailBatchLength > 0 ? fullBatchCount + 1 : fullBatchCount;

        var tailBatchInputDims = new int[batchInputDims.Length];
        tailBatchInputDims[0] = tailBatchLength;
        Array.Copy(DataDims, 0, tailBatchInputDims, 1, DataDims.Length);

        var tailBatchTargetDims = new int[batchTargetDims.Length];
        tailBatchTargetDims[0] = tailBatchLength;
        Array.Copy(TargetDims, 0, tailBatchTargetDims, 1, TargetDims.Length);

        // Allocate persistent batch arrays if not yet allocated
        if (BatchInputs is null || BatchInputs.Length < batchCount)
        {
            BatchInputs = new Tensor[batchCount];
            for (int i = 0; i < fullBatchCount; i++)
            {
                BatchInputs[i] = new(batchInputDims);
            }
            if (tailBatchLength > 0)
            {
                BatchInputs[fullBatchCount] = new(tailBatchInputDims);
            }
        }
        if (BatchTargets is null || BatchTargets.Length < batchCount)
        {
            BatchTargets = new Tensor[batchCount];
            for (int i = 0; i < fullBatchCount; i++)
            {
                BatchTargets[i] = new(batchTargetDims);
            }
            if (tailBatchLength > 0)
            {
                BatchTargets[fullBatchCount] = new(tailBatchTargetDims);
            }
        }

        int itemLength = Data[0].ElementCount;
        int targetLength = Targets[0].ElementCount;
        for (int b = 0; b < fullBatchCount; b++)
        {
            int batchOffset = b * batchSize;
            for (int i = 0; i < batchSize; i++)
            {
                shuffledData[batchOffset + i].Element.Data[0..itemLength].CopyTo(BatchInputs[b].Data.Slice(i * itemLength, itemLength));
                shuffledTargets[batchOffset + i].Data[0..targetLength].CopyTo(BatchTargets[b].Data.Slice(i * targetLength, targetLength));
            }
        }
        if (tailBatchLength > 0)
        {
            int batchOffset = fullBatchCount * batchSize;
            for (int i = 0; i < tailBatchLength; i++)
            {
                shuffledData[batchOffset + i].Element.Data[0..itemLength].CopyTo(BatchInputs[fullBatchCount].Data.Slice(i * itemLength, itemLength));
                shuffledTargets[batchOffset + i].Data[0..targetLength].CopyTo(BatchTargets[fullBatchCount].Data.Slice(i * targetLength, targetLength));
            }
        }

        return (BatchInputs, BatchTargets);
    }
}
