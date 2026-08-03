using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities;
using System;

namespace NNNCSharp.Components.Buffers
{
    /// <summary>
    /// Buffer for creating batches during standard supervised training.
    /// </summary>
    public class BatchBuffer
    {
        /// <summary>
        /// Array of all training inputs.
        /// </summary>
        readonly Tensor[] Data;
        /// <summary>
        /// Array of all training targets.
        /// </summary>
        readonly Tensor[] Targets;
        /// <summary>
        /// Dimensions of an unbatched training input.
        /// </summary>
        readonly int[] DataDims;
        /// <summary>
        /// Dimensions of an unbatched training target.
        /// </summary>
        readonly int[] TargetDims;
        /// <summary>
        /// Persistent array of batch input tensors.
        /// </summary>
        Tensor[]? BatchInputs = null;
        /// <summary>
        /// Persistent array of batch target tensors.
        /// </summary>
        Tensor[]? BatchTargets = null;

        /// <summary>
        /// Creates a new ReplayBuffer instance containing the given training data.
        /// </summary>
        /// <param name="data">Complete training inputs.</param>
        /// <param name="targets">Complete training targets.</param>
        public BatchBuffer(Tensor[] data, Tensor[] targets)
        {
            Data = data;
            Targets = targets;
            DataDims = data[0].Dimensions.ToArray();
            TargetDims = targets[0].Dimensions.ToArray();
        }

        ~BatchBuffer() // release all native C++ memory used by the persistent input and target arrays
        {
            if (BatchInputs is not null)
            {
                foreach (var tensor in BatchInputs) tensor.Dispose();
            }
            if (BatchTargets is not null)
            {
                foreach (var tensor in BatchTargets) tensor.Dispose();
            }
        }

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

            // Prepare batch tensors
            if (BatchInputs is null) // allocate persistent input batch array if not yet allocated
            {
                BatchInputs = new Tensor[1];
                BatchInputs[0] = new(batchDims);
            }
            else if (!Tensor.DimensionsMatch(BatchInputs[0].Dimensions, batchDims)) // ensure batch tensor has required dimensions
            {
                BatchInputs[0].Dispose(); // safely release native C++ memory and allocate new batch tensor
                BatchInputs[0] = new(batchDims);
            }

            if (BatchTargets is null) // allocate persistent target batch array if not yet allocated
            {
                BatchTargets = new Tensor[1];
                BatchTargets[0] = new(targetDims);
            }
            else if (!Tensor.DimensionsMatch(BatchTargets[0].Dimensions, targetDims)) // ensure batch tensor has required dimensions
            {
                BatchTargets[0].Dispose(); // safely release native C++ memory and allocate new batch tensor
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

        /// <summary>
        /// Randomly samples the maximum possible number of training batches without repeating inputs.
        /// </summary>
        /// <param name="batchSize">Number of input and target pairs to include per batch.</param>
        /// <returns>Arrays of batched inputs and targets, arranged with corresponding indices.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Requested batch size exceeds total number of input and target pairs.</exception>
        public (Tensor[] batchInputs, Tensor[] batchTargets) GetBatches(int batchSize)
        {
            if (batchSize <= 0 || batchSize > Data.Length) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size out of range.");

            // Shuffle input and target tensors while ensuring corresponding input-target pairs retain identical indices
            var shuffledData = ArrayUtils.GetRandomElements(Data, Data.Length);
            var shuffledTargets = new Tensor[shuffledData.Length];
            for (int i = 0; i < shuffledData.Length; i++)
            {
                shuffledTargets[i] = Targets[shuffledData[i].OriginalIndex];
            }

            // Compute batched input and target dimensions
            var batchInputDims = new int[Data[0].Rank + 1];
            batchInputDims[0] = batchSize;
            Array.Copy(DataDims, 0, batchInputDims, 1, DataDims.Length);

            var batchTargetDims = new int[Targets[0].Rank + 1];
            batchTargetDims[0] = batchSize;
            Array.Copy(TargetDims, 0, batchTargetDims, 1, TargetDims.Length);

            // Compute number of full-size batches and length of incomplete tail batch
            int fullBatchCount = Data.Length / batchSize;
            int tailBatchLength = Data.Length % batchSize;
            int batchCount = tailBatchLength > 0 ? fullBatchCount + 1 : fullBatchCount;

            // Compute tail batch input and target dimensions
            var tailBatchInputDims = new int[batchInputDims.Length];
            tailBatchInputDims[0] = tailBatchLength;
            Array.Copy(DataDims, 0, tailBatchInputDims, 1, DataDims.Length);

            var tailBatchTargetDims = new int[batchTargetDims.Length];
            tailBatchTargetDims[0] = tailBatchLength;
            Array.Copy(TargetDims, 0, tailBatchTargetDims, 1, TargetDims.Length);

            // Prepare batch tensor arrays
            if (BatchInputs is null || BatchInputs.Length < batchCount) // allocate persistent input batch array if not yet allocated or incorrect size
            {
                if (BatchInputs is not null) // safely release native C++ memory from previous batch
                {
                    foreach (var input in BatchInputs) input.Dispose();
                }

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
            else if (BatchInputs is not null) // ensure each batch tensor has required dimensions
            {
                for (int i = 0; i < fullBatchCount; i++)
                {
                    if (!Tensor.DimensionsMatch(BatchInputs[i].Dimensions, batchInputDims))
                    {
                        BatchInputs[i].Dispose(); // safely release native C++ memory and allocate new batch tensor
                        BatchInputs[i] = new(batchInputDims);
                    }
                }
                if (tailBatchLength > 0 && !Tensor.DimensionsMatch(BatchInputs[fullBatchCount].Dimensions, tailBatchInputDims))
                {
                    BatchInputs[fullBatchCount].Dispose(); // safely release native C++ memory and allocate new batch tensor
                    BatchInputs[fullBatchCount] = new(tailBatchInputDims);
                }
            }

            if (BatchTargets is null || BatchTargets.Length < batchCount) // allocate persistent target batch array if not yet allocated or incorrect size
            {
                if (BatchTargets is not null) // safely release native C++ memory from previous batch
                {
                    foreach (var target in BatchTargets) target.Dispose();
                }

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
            else if (BatchTargets is not null) // ensure each batch tensor has required dimensions
            {
                for (int i = 0; i < fullBatchCount; i++)
                {
                    if (!Tensor.DimensionsMatch(BatchTargets[i].Dimensions, batchTargetDims))
                    {
                        BatchTargets[i].Dispose(); // safely release native C++ memory and allocate new batch tensor
                        BatchTargets[i] = new(batchTargetDims);
                    }
                }
                if (tailBatchLength > 0 && !Tensor.DimensionsMatch(BatchTargets[fullBatchCount].Dimensions, tailBatchTargetDims))
                {
                    BatchTargets[fullBatchCount].Dispose(); // safely release native C++ memory and allocate new batch tensor
                    BatchTargets[fullBatchCount] = new(tailBatchTargetDims);
                }
            }

            // Fill batch input and target tensors
            int itemLength = Data[0].ElementCount;
            int targetLength = Targets[0].ElementCount;
            for (int b = 0; b < fullBatchCount; b++)
            {
                int batchOffset = b * batchSize;
                for (int i = 0; i < batchSize; i++)
                {
                    // Copy data from single input-target pair into corresponding location in batch input/target tensors
                    shuffledData[batchOffset + i].Element.Data[0..itemLength].CopyTo(BatchInputs![b].Data.Slice(i * itemLength, itemLength));
                    shuffledTargets[batchOffset + i].Data[0..targetLength].CopyTo(BatchTargets![b].Data.Slice(i * targetLength, targetLength));
                }
            }
            if (tailBatchLength > 0)
            {
                int batchOffset = fullBatchCount * batchSize;
                for (int i = 0; i < tailBatchLength; i++)
                {
                    // Copy data from single input-target pair into corresponding location in batch input/target tensors
                    shuffledData[batchOffset + i].Element.Data[0..itemLength].CopyTo(BatchInputs![fullBatchCount].Data.Slice(i * itemLength, itemLength));
                    shuffledTargets[batchOffset + i].Data[0..targetLength].CopyTo(BatchTargets![fullBatchCount].Data.Slice(i * targetLength, targetLength));
                }
            }

            return (BatchInputs!, BatchTargets!);
        }
    }
}
