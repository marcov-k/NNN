using NNN;

Model model;

Tensor inputs = new(20, 1);
Tensor targets = new(20, 1);

for (int i = 0; i < inputs.ElementCount; i++)
{
    float value = i * 2;
    inputs[i] = new(value);
    targets[i] = new(value * value);
}

(inputs, float inNorm) = Tensor.Normalize(inputs);
(targets, float outNorm) = Tensor.Normalize(targets);

Tensor testInputs = new(80, 1);
Tensor testTargets = new(80, 1);
for (int i = 0; i < testInputs.ElementCount; i++)
{
    float value = i * 0.5f;
    testInputs[i] = new(value);
    testTargets[i] = new(value * value);
}

testInputs = Tensor.Normalize(testInputs, inNorm).normalizedArray;

foreach (var target in testTargets.ToLinearArray())
{
    target.Value = MathF.Round(target.Value, MidpointRounding.AwayFromZero);
}

InteractionLoop();

void InteractionLoop()
{
    Console.WriteLine("Welcome to the Neural Network Nonsense Terminal (Enter Q to quit)");

    string fileName;

    string input = GetInput("Load model from file? y/n", ["y", "n"]);
    if (input == "y")
    {
        fileName = GetFileName();
        model = Saver.LoadModel(fileName);
        TestModel();
        TrainingLoop();
    }
    else
    {
        model = new([new Dense(10, new Sigmoid()), new Dense(10, new Sigmoid()), new Dense(1, new Linear())], inputs);
        TrainingLoop();
    }

    input = GetInput("Save model to a file? y/n", ["y", "n"]);
    if (input == "y")
    {
        SaveLoop();
    }

    Console.WriteLine("\nPress any key to quit...");
    Console.ReadKey();
    Environment.Exit(0);
}

string GetInput(string prompt, List<string>? options = null)
{
    options ??= [];
    for (int i = 0; i < options.Count; i++)
    {
        options[i] = options[i].ToLowerInvariant();
    }

    string input;
    while (true)
    {
        Console.WriteLine($"\n{prompt}");
        input = Console.ReadLine()?.ToLowerInvariant() ?? "";

        if (input == "q") Environment.Exit(0);
        else if (options.Count == 0 || options.Contains(input)) return input;
    }
}

string GetFileName()
{
    string input;
    while (true)
    {
        input = GetInput("Enter file name");
        if (Saver.FileExists(input)) return input;
        else Console.WriteLine("\nFile not found");
    }
}

int GetInteger(string prompt)
{
    string input;
    while (true)
    {
        input = GetInput(prompt);
        if (int.TryParse(input, out int integer)) return integer;
        else Console.WriteLine("\nNot a valid number");
    }
}

void TrainingLoop()
{
    Trainer trainer = new(model, optimizer: new SGD(learningRate: 0.01f), cost: new MSE());
    string input;
    int epochs;

    while (true)
    {
        input = GetInput("Train model? y/n", ["y", "n"]);
        if (input == "y")
        {
            epochs = GetInteger("Enter number of training epochs");
            trainer.Train(inputs, targets, epochs);
            TestModel();
        }
        else break;
    }
}

void TestModel()
{
    var predictions = model.Forward(testInputs);
    predictions = Tensor.UnnormalizeArray(predictions, outNorm);

    foreach (var prediction in predictions.ToLinearArray())
    {
        prediction.Value = MathF.Round(prediction.Value, MidpointRounding.AwayFromZero);
    }

    var diff = testTargets - predictions;
    var avgDiff = Number.Mean(diff.Data).Value;

    Console.WriteLine($"\nRunning test data...\n");
    Console.WriteLine($"Inputs:   {Tensor.UnnormalizeArray(testInputs, inNorm)}");
    Console.WriteLine($"Targets:  {testTargets}");
    Console.WriteLine($"Predicts: {predictions}");
    Console.WriteLine($"Average difference: {avgDiff:F2}");
}

void SaveLoop()
{
    string fileName;
    string input;

    while (true)
    {
        fileName = GetInput("Enter file name");
        if (Saver.FileExists(fileName))
        {
            input = GetInput($"File with name \"{fileName}\" already exists. Overwrite existing file? y/n", ["y", "n"]);
            if (input == "y")
            {
                Saver.SaveModel(model, fileName);
                Console.WriteLine("\nModel saved");
                break;
            }
        }
        else
        {
            Saver.SaveModel(model, fileName);
            Console.WriteLine("\nModel saved");
            break;
        }
    }
}

namespace NNN
{
    using System.Diagnostics;
    using System.Text.Json;

    public class Trainer(Model model, Optimizer optimizer, Cost cost)
    {
        readonly Model Model = model;
        readonly Optimizer Optimizer = optimizer;
        readonly Cost Cost = cost;

        public void Train(Tensor inputs, Tensor targets, int epochs)
        {
            Stopwatch timer = new();

            int logEvery = Math.Max(100, MathUtils.RoundToInterval(epochs / 500f, 100));
            Tensor predictions;
            Number loss;

            timer.Start();
            for (int e = 0; e < epochs; e++)
            {
                Model.ZeroGrad();
                predictions = Model.Forward(inputs);
                loss = Cost.CalculateCost(predictions, targets);
                loss.Backward();

                Model.Optimize(Optimizer);

                if (e % logEvery == 0 || e == epochs - 1)
                {
                    Console.WriteLine($"Epoch {e} : Loss = {loss.Value} : Time elapsed = {timer.ElapsedMilliseconds}ms : Time per epoch = {((float)timer.ElapsedMilliseconds / logEvery):F2}ms");
                    timer.Restart();
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
        public Layer[] Layers { get; private set; }

        public Model(Layer[] layers, Tensor inputFormat)
        {
            Layers = layers;
            SetUpLayers(inputFormat);
        }

        public Model(Saver.ModelData data)
        {
            Layers = new Layer[data.Layers.Length];

            BuildFromData(data);
        }

        void BuildFromData(Saver.ModelData data)
        {
            Saver.LayerData layerData;
            Type? layerType;
            for (int i = 0; i < data.Layers.Length; i++)
            {
                layerData = data.Layers[i];

                layerType = Type.GetType(layerData.LayerName);
                if (layerType != null)
                {
                    var layer = Activator.CreateInstance(layerType) as Layer;
                    layer?.BuildFromData(layerData);
                    if (layer != null) Layers[i] = layer;
                }
            }
        }

        public void SetUpLayers(Tensor inputFormat)
        {
            int inputs = inputFormat.GetLength(1);
            foreach (var layer in Layers)
            {
                layer.SetUpLayer(inputs);
                inputs = layer.NeuronCount;
            }
        }

        public Tensor Forward(Tensor input)
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

    public class Layer
    {
        public int NeuronCount { get; protected set; }

        public Layer(int neuronCount)
        {
            NeuronCount = neuronCount;
        }

        public Layer() { }

        public virtual void SetUpLayer(int inputCount)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor Forward(Tensor input)
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

        public virtual void BuildFromData(Saver.LayerData data)
        {
            throw new NotImplementedException();
        }
    }

    public class Dense : Layer
    {
        public Tensor Weights { get; private set; } = new();
        public Tensor Biases { get; private set; } = new();
        public Activation Activation { get; private set; } = new();

        public Dense(int neuronCount, Activation activation)
        {
            NeuronCount = neuronCount;
            Activation = activation;
        }

        public Dense() { }

        public override void SetUpLayer(int inputCount)
        {
            Weights = Tensor.InitWeights(inputCount, NeuronCount);
            Biases = Tensor.InitBias(NeuronCount);
        }

        public override Tensor Forward(Tensor input)
        {
            var output = input ^ Weights;
            output += Tensor.Broadcast(Biases, output.GetLength(0));
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

        public override void BuildFromData(Saver.LayerData data)
        {
            NeuronCount = data.NeuronCount;
            Weights = data.Weights;
            Biases = data.Biases;

            var activType = Type.GetType(data.Activation);
            if (activType != null)
            {
                Activation = Activator.CreateInstance(activType) as Activation ?? new();
            }
        }
    }

    public class Activation
    {
        public virtual Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }
    }

    public class LeakyReLU(float tau = 0.01f) : Activation
    {
        readonly float Tau = tau;

        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

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
        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Tanh(input[i]);
            }

            return output;
        }
    }

    public class Sigmoid : Activation
    {
        public override Tensor Forward(Tensor input)
        {
            Tensor output = new(input.Dimensions);

            for (int i = 0; i < input.ElementCount; i++)
            {
                output[i] = Number.Sigmoid(input[i]);
            }

            return output;
        }
    }

    public class Linear : Activation
    {
        public override Tensor Forward(Tensor input)
        {
            return input;
        }
    }

    public class Cost
    {
        public virtual Number CalculateCost(Tensor input, Tensor target)
        {
            throw new NotImplementedException();
        }
    }

    public class MSE : Cost
    {
        public override Number CalculateCost(Tensor input, Tensor target)
        {
            var diff = input - target;
            diff *= diff;
            return Tensor.Mean(diff);
        }
    }

    [Serializable]
    public class Tensor
    {
        public Number[] Data { get; set; }
        public int[] Dimensions { get; set; }
        public int[] Multipliers { get; set; }

        public int Rank => Dimensions.Length;
        public int ElementCount => Data.Length;

        public static Tensor operator +(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] + b[i];
            }

            return output;
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] - b[i];
            }

            return output;
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            AssertElementwiseDims(a, b);

            Tensor output = new(a.Dimensions);

            for (int i = 0; i < a.ElementCount; i++)
            {
                output[i] = a[i] * b[i];
            }

            return output;
        }

        public static Tensor operator ^(Tensor a, Tensor b)
        {
            AssertMultiplicationDims(a, b);

            var resultDims = new int[a.Rank];
            for (int i = 0; i < a.Rank - 1; i ++)
            {
                resultDims[i] = a.Dimensions[i];
            }
            resultDims[^1] = b.Dimensions[^1];

            Tensor output = new(resultDims);
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

        public static Number Mean(Tensor input)
        {
            return Number.Mean(input.ToLinearArray());
        }

        public static Tensor InitWeights(int inputCount, int neuronCount)
        {
            Tensor output = new(inputCount, neuronCount);

            float weight;
            for (int i = 0; i < output.ElementCount; i++)
            {
                weight = MathUtils.NextGaussian(0, MathF.Sqrt(2f / inputCount));
                output[i] = new(weight);
            }

            return output;
        }

        public static Tensor InitBias(int neuronCount)
        {
            Tensor output = new(neuronCount);

            for (int i = 0; i < output.ElementCount; i++)
            {
                output[i] = new(0.01f);
            }

            return output;
        }

        public static Tensor Broadcast(Tensor array, int firstDimLength)
        {
            var outputDims = new int[array.Rank + 1];
            outputDims[0] = firstDimLength;
            for (int i = 1; i < outputDims.Length; i++)
            {
                outputDims[i] = array.Dimensions[i - 1];
            }

            Tensor output = new(outputDims);

            int[] inputIndices;
            int[] outputIndices;
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < array.ElementCount; j++)
                {
                    inputIndices = array.GetFullIndices(j);

                    outputIndices = new int[output.Rank];
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

        public Tensor(params int[] dimensions)
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

        public Tensor[] ReduceDimensions()
        {
            var output = new Tensor[Dimensions[0]];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = ExtractSubArray(i);
            }

            return output;
        }

        Tensor ExtractSubArray(int firstDimIndex)
        {
            var extractDims = new int[Rank - 1];
            for (int i = 1; i < Rank; i++)
            {
                extractDims[i - 1] = Dimensions[i];
            }

            Tensor output = new(extractDims);

            int[] extractIndices;
            int[] parentIndices;
            for (int i = 0; i < output.ElementCount; i++)
            {
                extractIndices = output.GetFullIndices(i);

                parentIndices = new int[extractIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = extractIndices[j - 1];
                }

                output[extractIndices] = this[parentIndices];
            }

            return output;
        }

        public void InsertSubArray(int firstDimIndex, Tensor subArray)
        {
            int[] subIndices;
            int[] parentIndices;
            for (int i = 0; i < subArray.ElementCount; i++)
            {
                subIndices = subArray.GetFullIndices(i);

                parentIndices = new int[subIndices.Length + 1];

                parentIndices[0] = firstDimIndex;
                for (int j = 1; j < parentIndices.Length; j++)
                {
                    parentIndices[j] = subIndices[j - 1];
                }

                this[parentIndices] = subArray[subIndices];
            }
        }

        public static Tensor Transpose(Tensor array, int[]? axes = null)
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

            Tensor output = new(outputDims);

            for (int i = 0; i < array.ElementCount; i++)
            {
                output[RemapIndices(array.GetFullIndices(i), axes)] = array[i];
            }

            return output;
        }

        public static (Tensor normalizedArray, float normalizeFactor) Normalize(Tensor array, float? normalizeFactor = null)
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

            Tensor output = new(array.Dimensions);
            for (int i = 0; i < array.ElementCount; i++)
            {
                output[i] = new(array[i].Value / maxValue);
            }

            return (output, maxValue);
        }

        public static Tensor UnnormalizeArray(Tensor array, float normalizeFactor)
        {
            Tensor output = new(array.Dimensions);
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

        static void AssertElementwiseDims(Tensor a, Tensor b)
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

        static void AssertMultiplicationDims(Tensor a, Tensor b)
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

    [Serializable]
    public class Number
    {
        public float Value { get; set; }
        public float Gradient = 0;
        public List<Number> DependsOn = [];
        public string CreationOp = "";

        public Number(float value, List<Number>? dependsOn = null, string creationOp = "")
        {
            Value = value;
            DependsOn = dependsOn ?? [];
            CreationOp = creationOp;
        }

        public Number() { }

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
                    newGrad = Gradient * DependsOn[1].Value;
                    DependsOn[0].Backward(newGrad);

                    newGrad = Gradient * DependsOn[0].Value;
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

        public static int RoundToInterval(float value, int interval)
        {
            return (int)MathF.Round(value / interval, MidpointRounding.AwayFromZero) * interval;
        }
    }

    public static class Saver
    {
        const string Extension = ".nnn";

#pragma warning disable CS8604 // Possible null reference argument.
        public static void SaveModel(Model model, string fileName)
        {
            fileName += Extension;

            var layers = new LayerData[model.Layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                var layer = model.Layers[i];

                switch (layer)
                {
                    case Dense dense:
                        LayerData layerData = new(dense.NeuronCount, dense.GetType().AssemblyQualifiedName, dense.Weights,
                            dense.Biases, dense.Activation.GetType().AssemblyQualifiedName);
                        layers[i] = layerData;

                        break;
                }
            }

            ModelData modelData = new(layers);

            string json = JsonSerializer.Serialize(modelData);

            File.WriteAllText(fileName, json);
        }

        public static Model LoadModel(string fileName)
        {
            fileName += Extension;

            string json = File.ReadAllText(fileName);

            var modelData = JsonSerializer.Deserialize<ModelData>(json);

            Model model = new(modelData);

            return model;
        }
#pragma warning restore CS8604 // Possible null reference argument.

        public static bool FileExists(string fileName)
        {
            fileName += Extension;

            if (File.Exists(fileName)) return true;
            else return false;
        }

        [Serializable]
        public class ModelData(LayerData[] layers)
        {
            public LayerData[] Layers { get; set; } = layers;
        }

        [Serializable]
        public class LayerData(int neuronCount, string layerName, Tensor weights, Tensor biases, string activation)
        {
            public int NeuronCount { get; set; } = neuronCount;
            public string LayerName { get; set; } = layerName;
            public Tensor Weights { get; set; } = weights;
            public Tensor Biases { get; set; } = biases;
            public string Activation { get; set; } = activation;
        }
    }
}