using NNNCSharp.Components.Activations;
using NNNCSharp.Components.Autodiff;
using NNNCSharp.Components.Utilities.SaveSystem;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NNNCSharp.Components.Models.Layers
{
    /// <summary>
    /// Convolutional neural network layer.
    /// </summary>
    public class Conv : Layer
    {
        /// <summary>
        /// Number of filters in the layer.
        /// </summary>
        public int FilterCount { get; private set; }
        /// <summary>
        /// Dimensions of the kernels used by the layer.
        /// </summary>
        public int[] KernelDims { get; private set; } = new int[0];
        /// <summary>
        /// Tensor storing the kernels of the layer.
        /// </summary>
        public Tensor Kernels { get; private set; } = new();
        /// <summary>
        /// Type of padding to use.
        /// </summary>
        Padding PaddingType;
        /// <summary>
        /// Padding to add at start of every forward pass.
        /// </summary>
        int[] PaddingDims = new int[0];

        public enum Padding { Valid, Same }

        /// <summary>
        /// Creates a new convolutional layer instance.
        /// </summary>
        /// <param name="filterCount">Number of filters in the layer.</param>
        /// <param name="kernelDims">Dimensions of the kernels used by the layer.</param>
        /// <param name="activation">Activation function of the layer.</param>
        /// <param name="padding">Type of padding used by the layer.</param>
        /// <param name="dropout">Dropout rate of the layer.</param>
        public Conv(int filterCount, int[] kernelDims, Activation activation, Padding padding = Padding.Valid, float dropout = 0.0f)
            : base(activation, dropout)
        {
            FilterCount = filterCount;
            KernelDims = kernelDims;
            PaddingType = padding;
        }

        /// <summary>
        /// Creates a new convolutional layer instance.
        /// </summary>
        /// <param name="filterCount">Number of filters in the layer.</param>
        /// <param name="kernelDims">Dimensions of the kernels used by the layer.</param>
        /// <param name="kernels">Kernels tensor of the layer.</param>
        /// <param name="biases">Bias tensor of the layer.</param>
        /// <param name="activation">Activation function of the layer.</param>
        /// <param name="dropout">Dropout rate of the layer.</param>
        public Conv(int filterCount, int[] kernelDims, Tensor kernels, Tensor biases, Activation activation, Padding padding, int[] paddingDims, float dropout)
            : base(activation, dropout)
        {
            FilterCount = filterCount;
            KernelDims = kernelDims;
            Kernels.Dispose(); // release native C++ memory used by default allocation
            Kernels = kernels;
            Biases.Dispose(); // release native C++ memory used by default allocation
            Biases = biases;
            PaddingType = padding;
            PaddingDims = paddingDims;
        }

        /// <summary>
        /// Parameterless constructor for model reconstruction from save data.
        /// </summary>
        public Conv() { }

        // Base Layer API overrides

        public override void SetUpLayer(Tensor inputFormat)
        {
            // Initialize parameters
            Kernels.Dispose(); // release native C++ memory used by default allocation
            Kernels = Tensor.InitKernels(FilterCount, KernelDims, inputFormat.Dimensions[^1]);
            Biases.Dispose(); // release native C++ memory used by default allocation
            Biases = Tensor.InitBiases(FilterCount);
            ComputePadding(inputFormat);

            // Compute output dimensions
            var outputDims = new int[inputFormat.Rank];
            outputDims[0] = 1;

            if (PaddingType == Padding.Valid)
            {
                for (int i = 0; i < KernelDims.Length; i++)
                {
                    outputDims[i + 1] = inputFormat.Dimensions[i + 1] - KernelDims[i] + 1;
                }
            }
            else
            {
                inputFormat.Dimensions[1..^1].CopyTo(outputDims.AsSpan(1, KernelDims.Length));
            }
            outputDims[^1] = FilterCount;
            OutputFormat.Dispose(); // release native C++ memory used by default allocation
            OutputFormat = new(outputDims);
        }

        void ComputePadding(Tensor inputFormat)
        {
            PaddingDims = new int[KernelDims.Length * 2];
            if (PaddingType == Padding.Same)
            {
                for (int i = 0; i < KernelDims.Length; i++)
                {
                    int total = KernelDims[i] - 1;
                    PaddingDims[i * 2] = total / 2;
                    PaddingDims[i * 2 + 1] = total - total / 2;
                }
            }
        }

        public override Tensor Forward(Tensor input)
        {
            // Compute convolution and bias addition result
            // Release all native C++ memory used by intermediate tensor instances via 'using'
            // (intermediate tensor instances are kept alive by native C++ autograd graph until no longer needed)
            Tensor convInput = input;
            Tensor? padded = null;
            if (PaddingType == Padding.Same)
            {
                padded = Tensor.Pad(input, PaddingDims);
                convInput = padded;
            }
            using var conv = Tensor.Convolve(convInput, Kernels);
            padded?.Dispose();
            using var biasBroadcast = Tensor.Broadcast(Biases, conv.Dimensions.ToArray());
            using var biasAdd = conv + biasBroadcast;
            var output = Activation.Forward(biasAdd);

            if (Dropout > 0.0)
            {
                using var dropoutMask = Tensor.GetSpatialDropoutMask(output, Dropout);
                var dropoutOutput = output * dropoutMask;
                output.Dispose(); // release native C++ memory used by intermediate tensor instance
                output = dropoutOutput;
            }

            return output;
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            yield return Kernels;
            yield return Biases;
        }

        public override Layer Copy()
        {
            return new Conv(FilterCount, KernelDims.ToArray(), Kernels.Copy(), Biases.Copy(), Activation.Copy(), PaddingType, PaddingDims, Dropout);
        }

        internal override void WriteUniqueData(FileStream stream)
        {
            FileUtils.WriteInt32(stream, FilterCount);
            FileUtils.WriteTensor(stream, Kernels);
            FileUtils.WriteInt32(stream, (int)PaddingType);
            FileUtils.WriteInt32Array(stream, PaddingDims);
        }

        protected override void ReadUniqueData(FileStream stream)
        {
            FilterCount = FileUtils.ReadInt32(stream);
            Kernels.Dispose(); // release native C++ memory used by previous allocation
            Kernels = FileUtils.ReadTensor(stream);
            KernelDims = Kernels.Dimensions[1..^1].ToArray();
            PaddingType = (Padding)FileUtils.ReadInt32(stream);
            PaddingDims = FileUtils.ReadInt32Array(stream);
        }

        protected override string PrintUniqueLayer(FileStream stream)
        {
            int filterCount = FileUtils.ReadInt32(stream);
            var kernels = FileUtils.ReadTensor(stream);
            var kernelDims = kernels.Dimensions[1..^1];
            Padding paddingType = (Padding)FileUtils.ReadInt32(stream);
            _ = FileUtils.ReadInt32Array(stream); // skip padding dimensions

            string kernelDimsString = "Kernel Dimensions: [";
            for (int i = 0; i < kernelDims.Length; i++)
            {
                kernelDimsString += kernelDims[i];
                if (i < kernelDims.Length - 1) kernelDimsString += ", ";
            }
            kernelDimsString += "]";

            string kernelsString = "Kernels Tensor: Dimensions: [";
            for (int i = 0; i < kernels.Dimensions.Length; i++)
            {
                kernelsString += kernels.Dimensions[i];
                if (i < kernels.Dimensions.Length - 1) kernelsString += ", ";
            }
            kernelsString += $"], # of parameter values: {kernels.ElementCount}";

            string paddingString = $"Padding: {paddingType}";

            return $"Filters: {filterCount}\n{kernelDimsString}\n{kernelsString}\n{paddingString}";
        }

        public override void Dispose() // release native C++ memory used by shared layer tensor allocations and kernels tensor allocation
        {
            base.Dispose();
            Kernels.Dispose();
        }
    }
}
