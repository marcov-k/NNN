using NNN;

NDArray inputs = new(10, 1);
NDArray targets = new(10, 1);

for (int i = 0; i < inputs.ElementCount; i++)
{
    float value = i * 2;
    inputs[i] = new(value);
    targets[i] = new(value * value);
}

(inputs, float inNorm) = NDArray.Normalize(inputs);
(targets, float outNorm) = NDArray.Normalize(targets);

Model model = new([new Dense(5, new Sigmoid()), new Dense(5, new Sigmoid()), new Dense(1, new Linear())], inputs);
Trainer trainer = new(model, optimizer: new SGD(learningRate: 0.01f), cost: new MSE());

trainer.Train(inputs, targets, 50000);

NDArray testInputs = new(30, 1);
NDArray testTargets = new(30, 1);
for (int i = 0; i < testInputs.ElementCount; i++)
{
    float value = i * 0.5f;
    testInputs[i] = new(value);
    testTargets[i] = new(value * value);
}

testInputs = NDArray.Normalize(testInputs, inNorm).normalizedArray;

var predictions = model.Forward(testInputs);
predictions = NDArray.UnnormalizeArray(predictions, outNorm);

foreach (var target in testTargets.ToLinearArray())
{
    target.Value = MathF.Round(target.Value, MidpointRounding.AwayFromZero);
}
foreach (var prediction in predictions.ToLinearArray())
{
    prediction.Value = MathF.Round(prediction.Value, MidpointRounding.AwayFromZero);
}

Console.WriteLine();
Console.WriteLine($"Inputs:   {NDArray.UnnormalizeArray(testInputs, inNorm)}");
Console.WriteLine($"Targets:  {testTargets}");
Console.WriteLine($"Predicts: {predictions}");

namespace NNN
{
    public class Trainer(Model model, Optimizer optimizer, Cost cost)
    {
        readonly Model Model = model;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;

        public void Train(NDArray inputs, NDArray targets, int epochs)
        {
            int logEvery = Math.Max(1, epochs / 500);
            for (int e = 0; e < epochs; e++)
            {
                Model.ZeroGrad();
                var predictions = Model.Forward(inputs);
                var loss = Cost.CalculateCost(predictions, targets);
                loss.Backward();

                Model.Optimize(Optimizer);

                if (e % logEvery == 0 || e == epochs - 1)
                {
                    Console.WriteLine($"Epoch {e} : Loss = {loss.Value}");
                }
            }
        }
    }

    public class Optimizer(float learningRate)
    {
        protected readonly float LR = learningRate;

        public virtual void Step(Number parameter)
        {
            throw new NotImplementedException();
        }
    }

    public class SGD(float learningRate) : Optimizer(learningRate)
    {
        public override void Step(Number parameter)
        {
            parameter.Value -= parameter.Gradient * LR;
        }
    }

    public class Model
    {
        readonly Layer[] Layers;

        public Model(Layer[] layers, NDArray inputFormat)
        {
            Layers = layers;
            SetUpLayers(inputFormat);
        }

        public void SetUpLayers(NDArray inputFormat)
        {
            int inputs = inputFormat.GetLength(1);
            foreach (var layer in Layers)
            {
                layer.SetUpLayer(inputs);
                inputs = layer.NeuronCount;
            }
        }

        public NDArray Forward(NDArray input)
        {
            var output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public void Optimize(Optimizer optimizer)
        {
            foreach (var layer in Layers)
            {
                layer.Optimize(optimizer);
            }
        }

        public void ZeroGrad()
        {
            foreach (var layer in Layers)
            {
                layer.ZeroGrad();
            }
        }
    }

    public class Layer(int neuronCount)
    {
        public int NeuronCount { get; protected set; } = neuronCount;

        public virtual void SetUpLayer(int inputCount)
        {
            throw new NotImplementedException();
        }

        public virtual NDArray Forward(NDArray input)
        {
            throw new NotImplementedException();
        }

        public virtual void Optimize(Optimizer optimizer)
        {
            throw new NotImplementedException();
        }

        public virtual void ZeroGrad()
        {
            throw new NotImplementedException();
        }
    }

    public class Dense(int neuronCount, Activation activation) : Layer(neuronCount)
    {
        NDArray Weights = new();
        NDArray Biases = new();
        readonly Activation Activation = activation;

        public override void SetUpLayer(int inputCount)
        {
            Weights = NDArray.InitWeights(inputCount, NeuronCount);
            Biases = NDArray.InitBias(NeuronCount);
        }

        public override NDArray Forward(NDArray input)
        {
            var output = input ^ Weights;
            output += NDArray.Broadcast(Biases, output.GetLength(0));
            output = Activation.Forward(output);
            return output;
        }

        public override void Optimize(Optimizer optimizer)
        {
            foreach (var weight in Weights.ToLinearArray())
            {
                optimizer.Step(weight);
            }
            foreach (var bias in Biases.ToLinearArray())
            {
                optimizer.Step(bias);
            }
        }

        public override void ZeroGrad()
        {
            foreach (var weight in Weights.ToLinearArray())
            {
                weight.ZeroGradient();
            }
            foreach (var bias in Biases.ToLinearArray())
            {
                bias.ZeroGradient();
            }
        }
    }

    public class Activation
    {
        public virtual NDArray Forward(NDArray input)
        {
            throw new NotImplementedException();
        }
    }

    public class LeakyReLU(float tau = 0.01f) : Activation
    {
        readonly float Tau = tau;

        public override NDArray Forward(NDArray input)
        {
            NDArray output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                if (input[i].Value >= 0) output[i] = input[i];
                else output[i] = input[i] * Tau;
            }

            return output;
        }
    }

    public class Tanh : Activation
    {
        public override NDArray Forward(NDArray input)
        {
            NDArray output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Tanh(input[i]);
            }

            return output;
        }
    }

    public class Sigmoid : Activation
    {
        public override NDArray Forward(NDArray input)
        {
            NDArray output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Sigmoid(input[i]);
            }

            return output;
        }
    }

    public class Linear : Activation
    {
        public override NDArray Forward(NDArray input)
        {
            return input;
        }
    }

    public class Cost
    {
        public virtual Number CalculateCost(NDArray input, NDArray target)
        {
            throw new NotImplementedException();
        }
    }

    public class MSE : Cost
    {
        public override Number CalculateCost(NDArray input, NDArray target)
        {
            var diff = input - target;
            diff *= diff;
            return NDArray.Mean(diff);
        }
    }

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

        public static NDArray operator -(NDArray a, NDArray b)
        {
            AssertElementwiseDims(a, b);

            NDArray output = new(a.Dimensions);
            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] - b[i];
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

        public static Number Mean(NDArray input)
        {
            return Number.Mean(input.ToLinearArray());
        }

        public static NDArray InitWeights(int inputCount, int neuronCount)
        {
            NDArray output = new(inputCount, neuronCount);

            for (int i = 0; i < output.ElementCount; i++)
            {
                float weight = MathUtils.NextGaussian(0, MathF.Sqrt(2f / inputCount));
                output[i] = new(weight);
            }

            return output;
        }

        public static NDArray InitBias(int neuronCount)
        {
            NDArray output = new(neuronCount);

            for (int i = 0; i < output.ElementCount; i++)
            {
                output[i] = new(0.01f);
            }

            return output;
        }

        public static NDArray Broadcast(NDArray array, int firstDimLength)
        {
            var outputDims = new int[array.Rank + 1];
            outputDims[0] = firstDimLength;
            for (int i = 1; i < outputDims.Length; i++)
            {
                outputDims[i] = array.Dimensions[i - 1];
            }

            NDArray output = new(outputDims);

            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < array.ElementCount; j++)
                {
                    var inputIndices = array.GetFullIndices(j);

                    var outputIndices = new int[output.Rank];
                    outputIndices[0] = i;
                    for (int k = 1; k < outputIndices.Length; k++)
                    {
                        outputIndices[k] = inputIndices[k - 1];
                    }

                    output[outputIndices] = array[inputIndices];
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

        public Number[] ToLinearArray()
        {
            return Data;
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

        public static (NDArray normalizedArray, float normalizeFactor) Normalize(NDArray array, float? normalizeFactor = null)
        {
            float maxValue = array[0].Value;
            if (normalizeFactor == null)
            {
                foreach (var value in array.ToLinearArray())
                {
                    maxValue = MathF.Max(maxValue, value.Value);
                }
            }
            else maxValue = normalizeFactor.Value;

            NDArray output = new(array.Dimensions);
            for (int i = 0; i < array.ElementCount; i++)
            {
                output[i] = new(array[i].Value / maxValue);
            }

            return (output, maxValue);
        }

        public static NDArray UnnormalizeArray(NDArray array, float normalizeFactor)
        {
            NDArray output = new(array.Dimensions);
            for (int i = 0; i < array.ElementCount; i++)
            {
                output[i] = new(array[i].Value * normalizeFactor);
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
        public float Value { get; set; } = value;
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

        public static Number operator -(Number a, Number b)
        {
            return a + (b * -1);
        }

        public static Number operator -(Number a, float b)
        {
            return a - new Number(b);
        }

        public static Number operator -(float a, Number b)
        {
            return new Number(a) - b;
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

        public static Number operator /(Number a, Number b)
        {
            return new(value: a.Value / b.Value, dependsOn: [a, b], creationOp: "/");
        }

        public static Number operator /(Number a, float b)
        {
            return a / new Number(b);
        }

        public static Number operator /(float a, Number b)
        {
            return new Number(a) / b;
        }

        public static Number operator ^(Number a, Number b)
        {
            return new(value: MathF.Pow(a.Value, b.Value), dependsOn: [a, b], creationOp: "^");
        }

        public static Number operator ^(Number a, float b)
        {
            return a ^ new Number(b);
        }

        public static Number operator ^(float a, Number b)
        {
            return new Number(a) ^ b;
        }

        public void Backward(float? backwardGrad = null)
        {
            float newGrad;

            Gradient = (backwardGrad != null) ? Gradient + backwardGrad.Value : 1;

            switch (CreationOp)
            {
                case "+":
                    DependsOn[0].Backward(Gradient);
                    DependsOn[1].Backward(Gradient);

                    break;
                case "*":
                    newGrad = DependsOn[1].Value * Gradient;
                    DependsOn[0].Backward(newGrad);

                    newGrad = DependsOn[0].Value * Gradient;
                    DependsOn[1].Backward(newGrad);

                    break;
                case "/":
                    newGrad = Gradient * (1f / DependsOn[1].Value);
                    DependsOn[0].Backward(newGrad);

                    newGrad = Gradient * (-DependsOn[0].Value / MathF.Pow(DependsOn[1].Value, 2));
                    DependsOn[1].Backward(newGrad);

                    break;
                case "^":
                    newGrad = Gradient * DependsOn[1].Value * MathF.Pow(DependsOn[0].Value, DependsOn[1].Value - 1);
                    DependsOn[0].Backward(newGrad);

                    newGrad = Gradient * MathF.Pow(DependsOn[0].Value, DependsOn[1].Value) * MathF.Log(DependsOn[0].Value);
                    DependsOn[1].Backward(newGrad);

                    break;
            }
        }

        public void ZeroGradient()
        {
            Gradient = 0;
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

        public static Number Mean(Number[] inputs)
        {
            Number sum = new(0);
            foreach (var input in inputs)
            {
                sum += input;
            }
            return sum * (1f / inputs.Length);
        }

        public static Number Tanh(Number x)
        {
            return ((MathF.E ^ (2 * x)) - 1) / ((MathF.E ^ (2 * x)) + 1);
        }

        public static Number Sigmoid(Number x)
        {
            return 1 / (1 + (MathF.E ^ (-1 * x)));
        }
    }

    public static class MathUtils
    {
        static readonly Random random = new();

        public static float NextGaussian()
        {
            double u1 = 1.0f - random.NextDouble();
            double u2 = 1.0f - random.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return (float)randStdNormal;
        }

        public static float NextGaussian(float mean, float stdDev)
        {
            float randStdNormal = NextGaussian();
            float randNormal = mean + stdDev * randStdNormal;
            return randNormal;
        }
    }
}