using NNN;

var output = Number.Mean([new(5), new(6), new(7)]);

Console.WriteLine($"Output: {output}");

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

            //TODO: Implement matrix multiplication

            NDArray output = new(0);

            return output;
        }

        public NDArray(params int[] dimensions)
        {
            Dimensions = (int[])dimensions.Clone();
            Multipliers = new int[dimensions.Length];

            int totalSize = 1;
            for (int i = Dimensions.Length - 1; i >= 0; i--)
            {
                Multipliers[i] = totalSize;
                totalSize *= dimensions[i];
            }

            Data = new Number[totalSize];
        }

        int GetLinearIndex(int[] indices)
        {
            int linearIndex = 0;
            for (int i = 0; i < Rank; i++)
            {
                linearIndex += indices[i] * Multipliers[i];
            }
            return linearIndex;
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