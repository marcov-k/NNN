using NNN;

NDArray test = new(3, 3, 3, 3);

for (int i = 0; i < test.ElementCount; i++)
{
    test[i] = new(i);
}

Console.WriteLine($"Initial values: {test}");

var result = test ^ test;

Console.WriteLine($"Result values: {result}");

namespace NNN
{
    public class NDArray
    {
        readonly Number[] Data;
        public readonly int[] Dimensions;
        readonly int[] Multipliers;

        public int Rank => Dimensions.Length;
        public int ElementCount => Data.Length;

        public static NDArray operator +(NDArray a, NDArray b)
        {
            AssertElementwiseDims(a, b);

            NDArray output = new(a.Dimensions);
            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] + b[i];
            }

            return output;
        }

        public static NDArray operator *(NDArray a, NDArray b)
        {
            AssertElementwiseDims(a, b);

            NDArray output = new(a.Dimensions);
            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] * b[i];
            }

            return output;
        }

        public static NDArray operator ^(NDArray a, NDArray b)
        {
            AssertMultiplicationDims(a, b);

            var resultDims = new int[a.Rank];
            for (int i = 0; i < a.Rank - 1; i ++)
            {
                resultDims[i] = a.Dimensions[i];
            }
            resultDims[^1] = b.Dimensions[^1];

            NDArray output = new(resultDims);
            if (a.Rank > 2)
            {
                // Recursively reduce arrays down to batches of 2D matrices

                var reducedA = a.ReduceDimensions();
                var reducedB = b.ReduceDimensions();

                for (int i = 0; i < reducedA.Length; i++)
                {
                    output.InsertSubArray(i, reducedA[i] ^ reducedB[i]);
                }
            }
            else
            {
                // Standard 2D matrix multiplication
                for (int rowA = 0; rowA < output.GetLength(0); rowA++)
                {
                    for (int colB = 0; colB < output.GetLength(1); colB++)
                    {
                        output[rowA, colB] = new(0);
                        for (int i = 0; i < a.GetLength(1); i++)
                        {
                            output[rowA, colB] += a[rowA, i] * b[i, colB];
                        }
                    }
                }
            }

            return output;
        }

        public NDArray(params int[] dimensions)
        {
            Dimensions = (int[])dimensions.Clone();
            Multipliers = new int[Dimensions.Length];

            int totalSize = 1;
            for (int i = Rank - 1; i >= 0; i--)
            {
                Multipliers[i] = totalSize;
                totalSize *= Dimensions[i];
            }

            Data = new Number[totalSize];
        }

        public int GetLinearIndex(int[] indices)
        {
            int linearIndex = 0;
            for (int i = 0; i < Rank; i++)
            {
                linearIndex += indices[i] * Multipliers[i];
            }
            return linearIndex;
        }

        public int[] GetFullIndices(int index)
        {
            var indices = new int[Rank];

            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = index % Dimensions[i];
                index /= Dimensions[i];
            }

            return indices;
        }

        public override string ToString()
        {
            string output = string.Empty;
            for (int i = 0; i < ElementCount - 1; i++)
            {
                output += $"{Data[i].Value}, ";
            }
            output += Data[^1].Value;
            return output;
        }

        public NDArray[] ReduceDimensions()
        {
            var output = new NDArray[Dimensions[0]];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = ExtractSubArray(i);
            }

            return output;
        }

        NDArray ExtractSubArray(int firstDimIndex)
        {
            var extractDims = new int[Rank - 1];
            for (int i = 1; i < Rank; i++)
            {
                extractDims[i - 1] = Dimensions[i];
            }

            NDArray output = new(extractDims);

            for (int i = 0; i < output.ElementCount; i++)
            {
                var extractIndices = output.GetFullIndices(i);

                var parentIndices = new int[extractIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = extractIndices[j - 1];
                }

                output[extractIndices] = this[parentIndices];
            }

            return output;
        }

        public void InsertSubArray(int firstDimIndex, NDArray subArray)
        {
            for (int i = 0; i < subArray.ElementCount; i++)
            {
                var subIndices = subArray.GetFullIndices(i);

                var parentIndices = new int[subIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = subIndices[j - 1];
                }

                this[parentIndices] = subArray[subIndices];
            }
        }

        public static NDArray Transpose(NDArray array, int[]? axes = null)
        {
            if (axes == null)
            {
                axes = new int[array.Rank];
                for (int i = 0; i < array.Rank; i++)
                {
                    axes[i] = array.Rank - 1 - i;
                }
            }

            AssertTranspositionAxes(array.Dimensions, axes);

            var outputDims = RemapIndices(array.Dimensions, axes);

            NDArray output = new(outputDims);

            for (int i = 0; i < array.ElementCount; i++)
            {
                output[RemapIndices(array.GetFullIndices(i), axes)] = array[i];
            }

            return output;
        }

        public static int[] RemapIndices(int[] indices, int[] axes)
        {
            var output = new int[indices.Length];

            for (int i = 0; i < axes.Length; i++)
            {
                output[i] = indices[axes[i]];
            }

            return output;
        }

        public Number this[params int[] indices]
        {
            get => Data[GetLinearIndex(indices)];
            set => Data[GetLinearIndex(indices)] = value;
        }

        public Number this[int trueIndex]
        {
            get => Data[trueIndex];
            set => Data[trueIndex] = value;
        }

        public int GetLength(int dimension) => Dimensions[dimension];

        static void AssertElementwiseDims(NDArray a, NDArray b)
        {
            if (a.Rank != b.Rank) throw new ArgumentException("Array dimensions mismatch");
            else
            {
                for (int i = 0; i < a.Rank; i++)
                {
                    if (a.GetLength(i) != b.GetLength(i)) throw new ArgumentException("Array dimensions mismatch");
                }
            }
        }

        static void AssertMultiplicationDims(NDArray a, NDArray b)
        {
            if (a.Rank != b.Rank) throw new ArgumentException("Invalid array dimensions");
            else
            {
                if (a.Dimensions[^1] != b.Dimensions[^2]) throw new ArgumentException("Invalid array dimensions");
                for (int i = 0; i < a.Rank - 2; i++)
                {
                    if (a.GetLength(i) != b.GetLength(i)) throw new ArgumentException("Invalid array dimensions");
                }
            }
        }

        static void AssertTranspositionAxes(int[] dimensions, int[] axes)
        {
            if (axes.Length != dimensions.Length) throw new ArgumentException("Axes must match array dimensions");
            else
            {
                foreach (var axis in axes)
                {
                    if (axis >= dimensions.Length) throw new ArgumentException("Axes must match array dimensions");
                }
            }
        }
    }

    public class Number(float value, List<Number>? dependsOn = null, string creationOp = "")
    {
        public float Value { get; private set; } = value;
        public float Gradient { get; private set; } = 0;
        readonly List<Number> DependsOn = dependsOn ?? [];
        readonly string CreationOp = creationOp;

        public static Number operator +(Number a, Number b)
        {
            return new(value: a.Value + b.Value, dependsOn: [a, b], creationOp: "+");
        }

        public static Number operator +(Number a, float b)
        {
            return a + new Number(b);
        }

        public static Number operator +(float a, Number b)
        {
            return new Number(a) + b;
        }

        public static Number operator *(Number a, Number b)
        {
            return new(value: a.Value * b.Value, dependsOn: [a, b], creationOp: "*");
        }

        public static Number operator *(Number a, float b)
        {
            return a * new Number(b);
        }

        public static Number operator *(float a, Number b)
        {
            return new Number(a) * b;
        }

        public void Backward(float? backwardGrad = null)
        {
            Gradient = (backwardGrad != null) ? Gradient + backwardGrad.Value : 1;

            switch (CreationOp)
            {
                case "+":
                    DependsOn[0].Backward(Gradient);
                    DependsOn[1].Backward(Gradient);

                    break;
                case "*":
                    var newGrad = DependsOn[1].Value * Gradient;
                    DependsOn[0].Backward(newGrad);

                    newGrad = DependsOn[0].Value * Gradient;
                    DependsOn[1].Backward(newGrad);

                    break;
            }
        }

        public void ZeroGradient()
        {
            Gradient = 0;

            if (DependsOn.Count > 0)
            {
                DependsOn[0].ZeroGradient();
                DependsOn[1].ZeroGradient();
            }
        }

        public override string ToString()
        {
            return $"Value = {Value}; Gradient = {Gradient}";
        }

        public static Number Max(Number a, Number b)
        {
            return (a.Value > b.Value) ? a : b;
        }

        public static Number Min(Number a, Number b)
        {
            return (a.Value < b.Value) ? a : b;
        }

        public static Number Mean(List<Number> inputs)
        {
            Number sum = new(0);
            foreach (var input in inputs)
            {
                sum += input;
            }
            return sum * (1f / inputs.Count);
        }
    }
}