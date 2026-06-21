using NNN.Components.Autodiff;

namespace NNN.Components.Buffers;

public class BatchBuffer(Tensor[] data)
{
    readonly Tensor[] Data = data;
    readonly int[] DataDims = data[0].Dimensions;

    public Tensor GetBatch(int batchSize)
    {
        if (batchSize <= 0 || batchSize > Data.Length) throw new ArgumentException("Batch size out of range.");

        var batchDims = new int[Data.Rank + 1];
        batchDims[0] = batchSize;
        Array.Copy(DataDims, 0, batchDims, 1, DataDims.Length);

        Tensor batch = new(batchDims);

        int itemLength = Data[0].ElementCount;
        var batchItems = Random.Shared.GetItems(Data, batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(batchItems[i].Data, 0, batch.Data, i * itemLength, itemLength);
        }

        return batch;
    }
}
