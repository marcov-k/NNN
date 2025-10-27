using CsvHelper;
using ILGPU;
using ILGPU.Runtime;
using NNN;
using NumSharp;
using Python.Runtime;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;

bool useGPU = false;

var mnistData = await MNISTLoader.LoadMNIST();

var (xTrain, yTrain, xTest, yTest) = (np.array(mnistData.XTrain, np.float64), np.array(mnistData.YTrain, np.float64),
    np.array(mnistData.XTest, np.float64), np.array(mnistData.YTest, np.float64));

int randomSeed = 190119;

xTrain = xTrain.reshape(-1, 28 * 28);
xTest = xTest.reshape(-1, 28 * 28);

(xTrain, xTest) = (xTrain - np.mean(xTrain), xTest - np.mean(xTest));
(xTrain, xTest) = (xTrain / np.std(xTrain), xTest / np.std(xTest));

Stopwatch stopwatch = new Stopwatch();
stopwatch.Start();
var numTrainLabels = yTrain.shape[0];
var trainLabels = np.zeros((numTrainLabels, 10));
int trainTaskCount = 12;
List<Task> labelsTasks = new List<Task>();
Console.WriteLine();
for (int i = 0; i < trainTaskCount; i++)
{
    int index = i;
    Task labelsTask = Task.Run(() =>
    {
        Console.WriteLine("Start train task...");
        int start = (int)Math.Round((double)(index * (numTrainLabels / trainTaskCount)));
        int end = (int)Math.Round((double)((index + 1) * (numTrainLabels / trainTaskCount)));
        for (int j = start; j < end; j++)
        {
            trainLabels[j][yTrain[j]] = 1;
        }
        Console.WriteLine($"Finish train task, processed labels from {start} to {end - 1}");
    });
    labelsTasks.Add(labelsTask);
}

var numTestLabels = yTest.shape[0];
var testLabels = np.zeros((numTestLabels, 10));
int testTaskCount = (int)Math.Round(trainTaskCount / 6.0);
if (testTaskCount < 1) { testTaskCount = 1; }
for (int i = 0; i < testTaskCount; i++)
{
    int index = i;
    Task labelsTask = Task.Run(() =>
    {
        Console.WriteLine("Start test task...");
        int start = (int)Math.Round((double)(index * (numTestLabels / testTaskCount)));
        int end = (int)Math.Round((double)((index + 1) * (numTestLabels / testTaskCount)));
        for (int j = start; j < end; j++)
        {
            testLabels[j][yTest[j]] = 1;
        }
        Console.WriteLine($"Finish test task, processed labels from {start} to {end - 1}");
    });
    labelsTasks.Add(labelsTask);
}

await Task.WhenAll(labelsTasks);
labelsTasks.Clear();
stopwatch.Stop();
Console.WriteLine($"\nTasks took a total of {stopwatch.ElapsedMilliseconds / 1000.0} seconds to complete...");

NeuralNetwork model = new NeuralNetwork(layers: [new Dense(neurons: 89, activation: new Tanh()),
    new Dense(neurons: 10, activation: new Sigmoid())], loss: new MeanSquaredError(), seed: randomSeed);

Trainer trainer = new Trainer(model, new SGD(lr: 0.01));

await Task.Run(async() =>
{
    await trainer.Fit(xTrain, trainLabels, xTest, testLabels, epochs: 2, 
        evalEvery: 1, batchSize: 60, seed: randomSeed, useGPU: useGPU);
});

CalcAccuracyModel(model, xTest, testLabels, useGPU);

Console.WriteLine($"\nPress any key to close...");
Console.ReadKey();

static void CalcAccuracyModel(NeuralNetwork model, NDArray testSet, NDArray yTest, bool useGPU = false)
{
    NDArray predictions = model.Forward(testSet, useGPU);
    NDArray predictedLabels = np.argmax(predictions, axis: 1);
    NDArray yTestLabels = np.argmax(yTest, axis: 1);
    int[] mask = (predictedLabels == yTestLabels).ToArray<bool>().Select(x => x ? 1 : 0).ToArray();
    double correct = mask.Sum();
    double accuracy = correct * 100.0 / testSet.Shape[0];
    Console.WriteLine($"\nThe model validation accuracy is: {accuracy:F2}%");
}

namespace NNN
{
    public class Trainer
    {
        // Trains a neural network

        protected NeuralNetwork Net { get; set; }
        protected Optimizer Optim { get; set; }
        protected double BestLoss { get; set; }

        public async Task Fit(NDArray xTrain, NDArray yTrain, NDArray xTest, NDArray yTest, int epochs = 100, int evalEvery = 10,
            int batchSize = 32, int seed = 1, bool restart = true, bool earlyStopping = true, bool useGPU = false)
        {
            // Fits neural network on training data for certain number of epochs
            // Every "evalEvery" epochs, evaluates neural network on testing data
            np.random.seed(seed);
            Optim.MaxEpochs = epochs;
            Optim.SetupDecay();

            if (restart)
            {
                foreach (var layer in Net.Layers)
                {
                    layer.First = true;
                }
                BestLoss = double.MaxValue;
            }

            var setupBatch = Helpers.GenerateSetupBatch(xTrain, yTrain).xBatch;
            Net.SetupLayers(setupBatch);

            var lastModel = Net.Copy();
            for (int e = 0; e < epochs; e++)
            {
                if ((e + 1) % evalEvery == 0)
                {
                    lastModel = Net.Copy();
                }

                (xTrain, yTrain) = Helpers.PermuteData(xTrain, yTrain);

                var batchGenerator = Helpers.GenerateBatches(xTrain, yTrain, batchSize);

                Stopwatch stopwatch = new Stopwatch();
                int i = 1;
                foreach (var (xBatch, yBatch) in batchGenerator)
                {
                    stopwatch.Start();
                    Net.TrainBatch(xBatch, yBatch, useGPU);
                    Optim.Step();
                    stopwatch.Stop();
                    Console.WriteLine($"Batch {i}/{batchGenerator.Count()} finished...");
                    Console.WriteLine($"Batch took {stopwatch.ElapsedMilliseconds / 1000.0} seconds...");
                    stopwatch.Reset();
                    i++;
                }

                if ((e + 1) % evalEvery == 0)
                {
                    var testPreds = Net.Forward(xTest, useGPU);
                    var loss = Net.Loss.Forward(testPreds, yTest);

                    if (earlyStopping)
                    {
                        if (loss < BestLoss)
                        {
                            Console.WriteLine($"Validation loss after {e + 1} epochs is {loss}");
                            BestLoss = loss;
                        }
                        else
                        {
                            Console.WriteLine($"Loss increased after epoch {e + 1}, final loss was {BestLoss}, using the model from epoch {e + 1 - evalEvery}");
                            Net = lastModel;
                            Optim.Net = Net;
                            break;
                        }
                    }
                    else
                    {
                        Console.WriteLine($"Validation loss after {e + 1} epochs is {loss}");
                    }
                }
                Optim.DecayLR();
            }
        }

        public Trainer(NeuralNetwork net, Optimizer optim)
        {
            // Requires neural network and optimzer for training to occur.
            // Assign neural network as instance variable to the optimizer.

            Net = net;
            Optim = optim;
            Optim.Net = net;
            BestLoss = 1e9;
        }
    }

    public class Optimizer
    {
        // Base class for neural network optimizer

        protected double LR { get; set; }
        protected double FinalLR { get; set; }
        protected string? DecayType { get; set; }
        protected bool First { get; set; }
        protected double DecayPerEpoch { get; set; }
        public double MaxEpochs { get; set; }
        public NeuralNetwork? Net { get; set; }

        public void SetupDecay()
        {
            switch (DecayType)
            {
                case "exponential":
                    DecayPerEpoch = np.power(FinalLR / LR, 1.0 / (MaxEpochs - 1));
                    break;
                case "linear":
                    DecayPerEpoch = (LR - FinalLR) / (MaxEpochs - 1);
                    break;
                default:
                    return;
            }
        }

        public void DecayLR()
        {
            switch (DecayType)
            {
                case "exponential":
                    LR *= DecayPerEpoch;
                    break;
                case "linear":
                    LR -= DecayPerEpoch;
                    break;
                default:
                    return;
            }
        }

        public virtual void Step()
        {
            var newParams = new List<NDArray>();
            foreach (var (param, paramGrad) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads()))
            {
                var inputDict = new Dictionary<string, NDArray>() { { "param", param }, { "grad", paramGrad } };
                var newParam = UpdateRule(inputDict);
                newParams.Add(newParam);
            }
            Net.SetParams(newParams);
        }

        protected virtual NDArray UpdateRule(Dictionary<string, NDArray> args)
        {
            throw new NotImplementedException();
        }

        public Optimizer(double lr = 0.01, double finalLR = 0.0, string? decayType = null)
        {
            // Every optimizer must have initial learning rate

            LR = lr;
            FinalLR = finalLR;
            DecayType = decayType;
            First = true;
        }
    }

    public class SGD : Optimizer
    {
        // Stochastic gradient descent optimizer

        protected override NDArray UpdateRule(Dictionary<string, NDArray> args)
        {
            var update = LR * args["grad"];
            return args["param"] - update;
        }

        public SGD(double lr = 0.01, double finalLR = 0.0, string? decayType = null) : base(lr, finalLR, decayType) { }
    }

    public class SGDMomentum : Optimizer
    {
        protected double Momentum { get; set; }
        protected List<NDArray> Velocities { get; set; }

        public override void Step()
        {
            if (First)
            {
                Velocities = new List<NDArray>();
                foreach (var param in Net.CalcParams())
                {
                    Velocities.Add(np.zeros_like(param));
                }
                First = false;
            }

            var newParams = new List<NDArray>();
            foreach (var (param, paramGrad, velocity) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads(), Velocities))
            {
                var inputDict = new Dictionary<string, NDArray>() { { "param", param }, { "grad", paramGrad }, { "velocity", velocity } };
                var newParam = UpdateRule(inputDict);
                newParams.Add(newParam);
            }
            Net.SetParams(newParams);
        }

        protected override NDArray UpdateRule(Dictionary<string, NDArray> args)
        {
            // Update velocity

            var velocity = args["velocity"] * Momentum;
            velocity += LR * args["grad"];

            // Use velocity to update parameters

            args["param"] -= args["velocity"];
            var param = args["param"] - velocity;
            return param;
        }

        public SGDMomentum(double lr = 0.01, double finalLR = 0.0, string? decayType = null, double momentum = 0.9) : base(lr, finalLR, decayType)
        {
            Momentum = momentum;
            Velocities = new List<NDArray>();
        }
    }

    public class AdaGrad : Optimizer
    {
        protected double Eps { get; set; }
        protected List<NDArray> SumSquares { get; set; }

        public override void Step()
        {
            if (First)
            {
                SumSquares = new List<NDArray>();
                foreach (var param in Net.CalcParams())
                {
                    SumSquares.Add(np.zeros_like(param));
                }
                First = false;
            }

            var newParams = new List<NDArray>();
            foreach (var (param, paramGrad, sumSquare) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads(), SumSquares))
            {
                var inputDict = new Dictionary<string, NDArray>() { { "param", param }, { "grad", paramGrad }, { "sumSquare", sumSquare } };
                var newParam = UpdateRule(inputDict);
                newParams.Add(newParam);
            }
            Net.SetParams(newParams);
        }

        protected override NDArray UpdateRule(Dictionary<string, NDArray> args)
        {
            // Update sum of squares

            var sumSquare = args["sumSquare"] + Eps + np.power(args["grad"], 2);

            // Scale learning rate by sum of squares

            LR = np.divide(LR, np.sqrt(sumSquare));

            // Use to update parameters

            var param = args["param"] - LR * args["grad"];

            return param;
        }

        public AdaGrad(double lr = 0.01, double finalLR = 0.0) : base(lr, finalLR)
        {
            Eps = 1e-7;
            SumSquares = new List<NDArray>();
        }
    }

    public class RegularizedSGD : Optimizer
    {
        protected double Alpha { get; set; }

        public override void Step()
        {
            var newParams = new List<NDArray>();
            foreach (var (param, paramGrad) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads()))
            {
                var inputDict = new Dictionary<string, NDArray>() { { "param", param }, { "grad", paramGrad } };
                var newParam = UpdateRule(inputDict);
                newParams.Add(newParam);
            }
            Net.SetParams(newParams);
        }

        protected override NDArray UpdateRule(Dictionary<string, NDArray> args)
        {
            var param = args["param"] - (LR * args["grad"] + Alpha * args["param"]);
            return param;
        }

        public RegularizedSGD(double lr = 0.01, double alpha = 0.1) : base(lr)
        {
            Alpha = alpha;
        }
    }

    public class NeuralNetwork
    {
        // A neural network consisting of multiple "layers"

        public List<Layer> Layers { get; set; }
        public Loss Loss { get; set; }
        public int Seed { get; set; }

        public NDArray Forward(NDArray xBatch, bool useGPU = false)
        {
            // Pass data forward through layers

            var xOut = xBatch.Clone();
            foreach (var layer in Layers)
            {
                xOut = layer.Forward(xOut, useGPU);
            }

            return xOut;
        }

        public void Backward(NDArray lossGrad, bool useGPU = false)
        {
            // Pass data backward through layers

            var grad = lossGrad;
            var revLayers = Layers.ToList();
            revLayers.Reverse();
            foreach (var layer in revLayers)
            {
                grad = layer.Backward(grad, useGPU);
            }
        }

        public double TrainBatch(NDArray xBatch, NDArray yBatch, bool useGPU = false)
        {
            // Pass data forward through layers
            // Compute loss
            // Pass data backward through layers

            xBatch = xBatch.Clone();
            yBatch = yBatch.Clone();

            var predictions = Forward(xBatch, useGPU);
            var loss = Loss.Forward(predictions, yBatch);
            var lossGrad = Loss.Backward();
            Backward(lossGrad, useGPU);

            return loss;
        }

        public List<NDArray> CalcParams()
        {
            // Get parameters for network

            List<NDArray> netParams = new List<NDArray>();

            foreach (var layer in Layers)
            {
                netParams.AddRange(layer.CalcParams());
            }
            return netParams.ToList();
        }

        public void SetParams(List<NDArray> newParams)
        {
            newParams = newParams.ToList();
            List<NDArray> layerParams = new List<NDArray>();
            for (int i = 0; i < Layers.Count; i++)
            {
                layerParams.Clear();
                layerParams.AddRange(newParams.GetRange(0, Layers[i].Params.Count));
                Layers[i].SetParams(layerParams);
                newParams.RemoveRange(0, Layers[i].Params.Count);
            }
        }

        public List<NDArray> CalcParamGrads()
        {
            // Get gradient of loss with respect to parameters for network

            List<NDArray> netParamGrads = new List<NDArray>();

            foreach (var layer in Layers)
            {
                netParamGrads.AddRange(layer.CalcParamGrads());
            }
            return netParamGrads.ToList();
        }

        public NeuralNetwork Copy()
        {
            List<Layer> newLayers = new List<Layer>();
            foreach (var layer in Layers)
            {
                newLayers.Add(layer.Copy());
            }
            Loss newLoss = Loss.Copy();
            return new NeuralNetwork(newLayers, newLoss, Seed);
        }

        public void SetupLayers(NDArray input)
        {
            input = input.Clone();
            foreach (var layer in Layers)
            {
                layer.SetupForInput(input);
                input = layer.Forward(input);
            }
        }

        public string CreateJsonString()
        {
            var layers = new List<LayerData>();
            foreach (var layer in Layers)
            {
                var data = layer.CreateLayerData();
                layers.Add(data);
            }
            double? eps = null;
            if (Loss is SoftmaxCrossEntropy)
            {
                var softmax = Loss as SoftmaxCrossEntropy;
                eps = softmax.Eps;
            }
            var lossData = new LossData(Loss.GetType().Name, eps);
            var networkData = new NNData(layers, lossData, Seed);
            string jsonString = JsonSerializer.Serialize(networkData);
            return jsonString;
        }

        public NeuralNetwork(List<Layer> layers, Loss loss, int seed = 1)
        {
            // Neural networks need layers and a loss

            Layers = layers.ToList();
            Loss = loss;
            Seed = seed;
            if (seed != default)
            {
                foreach (var layer in Layers) { layer.Seed = Seed; }
            }
        }

        public NeuralNetwork(string jsonString)
        {
            // Build neural network from json string

            if (jsonString == null) { throw new ArgumentNullException(); }
            NNData data = JsonSerializer.Deserialize<NNData>(jsonString);
            Seed = data.Seed;
            Type lossType = Type.GetType(data.Loss.LossType);
            dynamic loss = Activator.CreateInstance(lossType);
            loss.BuildFromLossData(data.Loss);
            Loss = loss;
            Layers = new List<Layer>();
            foreach (var layerData in data.Layers)
            {
                Type layerType = Type.GetType(layerData.LayerType);
                dynamic layer = Activator.CreateInstance(layerType);
                layer.BuildFromLayerData(layerData);
                Layers.Add(layer);
            }
        }
    }

    public class Layer
    {
        // A "layer" of neurons for a neural network

        public int Neurons { get; set; } = 0;
        public bool First { get; set; } = true;
        public List<NDArray>? Params { get; set; }
        public List<NDArray>? ParamGrads { get; set; }
        public List<Operation>? Operations { get; set; }
        public NDArray? Input { get; set; }
        public NDArray? Output { get; set; }
        public NDArray? InputGrad { get; set; }
        public int Seed { get; set; } = 1;

        public NDArray Forward(NDArray input, bool useGPU = false)
        {
            // Pass input forward through operations

            input = input.Clone();
            Input = input.Clone();
            if (First)
            {
                SetupLayer(input);
                First = false;
            }

            foreach (var operation in Operations)
            {
                input = operation.Forward(input, useGPU);
            }
            Output = input;

            return Output;
        }

        public NDArray Backward(NDArray outputGrad, bool useGPU = false)
        {
            // Send output gradient backward through operations

            Helpers.AssertSameShape(Output, outputGrad);

            outputGrad = outputGrad.Clone();
            var revOps = Operations.ToList();
            revOps.Reverse();
            foreach (var operation in revOps)
            {
                outputGrad = operation.Backward(outputGrad, useGPU);
            }

            InputGrad = outputGrad.Clone();
            CalcParamGrads();
            return InputGrad;
        }

        public virtual void SetupForInput(NDArray input)
        {
            input = input.Clone();
            Input = input.Clone();
            SetupLayer(input);
        }

        protected virtual void SetupLayer(NDArray numIn)
        {
            // SetupLayer() must be defined for each layer

            throw new NotImplementedException();
        }

        public List<NDArray> CalcParamGrads()
        {
            // Extracts ParamGrads from layer's operations

            ParamGrads = new List<NDArray>();
            foreach (var operation in Operations)
            {
                if (operation is ParamOperation)
                {
                    var paramOp = operation as ParamOperation;
                    ParamGrads.Add(paramOp.ParamGrad);
                }
            }
            return ParamGrads.ToList();
        }

        public List<NDArray> CalcParams()
        {
            // Extracts Params from layer's operations

            Params = new List<NDArray>();
            foreach (var operation in Operations)
            {
                if (operation is ParamOperation)
                {
                    var paramOp = operation as ParamOperation;
                    Params.Add(paramOp.Param);
                }
            }
            return Params.ToList();
        }

        public void SetParams(List<NDArray> newParams)
        {
            int index = 0;
            foreach (var operation in Operations)
            {
                if (operation is ParamOperation)
                {
                    operation.SetParams(newParams[index]);
                    index++;
                }
            }
        }

        public virtual Layer Copy()
        {
            List<Operation> newOps = new List<Operation>();
            foreach (var operation in Operations)
            {
                newOps.Add(operation.Copy());
            }
            List<NDArray> newParamGrads = new List<NDArray>();
            foreach (var paramGrad in ParamGrads)
            {
                newParamGrads.Add(paramGrad.Clone());
            }
            List<NDArray> newParams = new List<NDArray>();
            foreach (var param in Params)
            {
                newParams.Add(param.Clone());
            }
            return new Layer
            {
                First = this.First,
                Input = this.Input,
                InputGrad = this.InputGrad,
                Neurons = this.Neurons,
                Operations = newOps.ToList(),
                Output = this.Output,
                ParamGrads = newParamGrads.ToList(),
                Params = newParams.ToList(),
                Seed = this.Seed
            };
        }

        public virtual LayerData CreateLayerData()
        {
            // CreateLayerData() must be defined for each subclass

            var ops = new List<string>();
            foreach (var op in Operations)
            {
                ops.Add(op.GetType().Name);
            }
            var layerParamsNDArray = CalcParams();
            var layerParams = Helpers.NDArrayList2JagArrayList(layerParamsNDArray);
            var data = new LayerData(GetType().Name, ops, layerParams, First, Neurons, Seed);
            return data;
        }

        public virtual void BuildFromLayerData(LayerData data)
        {
            // BuildFromLayerData() must be defined for each subclass

            Neurons = data.Neurons;
            First = data.First;
            Seed = data.Seed;
            Params = new List<NDArray>();
            Operations = new List<Operation>();
            foreach (var param in data.Params)
            {
                Params.Add(np.array(param, np.float64));
            }
            foreach (var opName in data.Operations)
            {
                Type type = Type.GetType(opName);
                dynamic op = Activator.CreateInstance(type);
                Operations.Add(op);
            }
            SetParams(Params);
        }

        public Layer()
        {
            First = true;
            Params = new List<NDArray>();
            ParamGrads = new List<NDArray>();
            Operations = new List<Operation>();
        }

        public Layer(int neurons) : this()
        {
            // Number of "neurons" roughly corresponds to "breadth" of layer

            Neurons = neurons;
        }
    }

    public class Dense : Layer
    {
        public Operation? Activation { get; set; }

        protected override void SetupLayer(NDArray numIn)
        {
            // Defines operations of fully connected layer

            if (Seed != default) { np.random.seed(Seed); }

            Params = new List<NDArray>();

            // weights
            Params.Add(np.random.randn(Input.shape[1], Neurons));

            // bias
            Params.Add(np.random.randn(1, Neurons));

            Operations = [new WeightMultiply(Params[0]), new BiasAdd(Params[1]), Activation];

            First = false;
        }

        public override Dense Copy()
        {
            var newLayer = base.Copy();
            var newActiv = Activation.Copy();
            return new Dense
            {
                Activation = newActiv,
                Input = newLayer.Input,
                First = newLayer.First,
                InputGrad = newLayer.InputGrad,
                Neurons = newLayer.Neurons,
                Operations = newLayer.Operations,
                Output = newLayer.Output,
                ParamGrads = newLayer.ParamGrads,
                Params = newLayer.Params,
                Seed = newLayer.Seed
            };
        }

        public override LayerData CreateLayerData()
        {
            var data = base.CreateLayerData();
            data.Activation += Activation.GetType().Name;
            return data;
        }

        public override void BuildFromLayerData(LayerData data)
        {
            base.BuildFromLayerData(data);
            Type activType = Type.GetType(data.Activation);
            dynamic activ = Activator.CreateInstance(activType);
            Activation = activ;
        }

        public Dense() { }

        public Dense(int neurons, Operation? activation = null) : base(neurons)
        {
            Activation = activation ?? new Sigmoid();
        }
    }

    public class Operation
    {
        // Base class for operations

        public NDArray? Input { get; set; }
        public NDArray? Output { get; set; }
        public NDArray? InputGrad { get; set; }

        public NDArray Forward(NDArray input, bool useGPU = false)
        {
            // Send input forward through operation

            Input = input;
            Output = CalcOutput(useGPU);
            return Output;
        }

        public virtual NDArray Backward(NDArray outputGrad, bool useGPU = false)
        {
            // Calculate input gradient from output gradient

            Helpers.AssertSameShape(Output, outputGrad);

            InputGrad = CalcInputGrad(outputGrad, useGPU);

            Helpers.AssertSameShape(Input, InputGrad);

            return InputGrad;
        }

        public virtual void SetParams(NDArray newParams)
        {
            // SetParams() must be defined only for the ParamOperation subclass

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcOutput(bool useGPU = false)
        {
            // CalcOutput() must be defined for each operation

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            // CalcInputGrad() must be defined for each operation

            throw new NotImplementedException();
        }

        public virtual Operation Copy()
        {
            // Copy() must be separately defined for every subclass
            return new Operation { Input = this.Input, InputGrad = this.InputGrad, Output = this.Output };
        }
    }

    public class Sigmoid : Operation
    {
        // Sigmoid activation function

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            // Compute output

            return 1.0 / (1.0 + np.exp(-1.0 * Input));
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Compute input gradient

            NDArray sigmoidBackward;
            if (useGPU)
            {
                sigmoidBackward = GPUNDArrayOps.PointwiseMultiply(Output, 1.0 - Output);
                InputGrad = GPUNDArrayOps.PointwiseMultiply(sigmoidBackward, outputGrad);
            }
            else
            {
                sigmoidBackward = Output * (1.0 - Output);
                InputGrad = sigmoidBackward * outputGrad;
            }
            return InputGrad;
        }

        public override Sigmoid Copy()
        {
            var newOp = base.Copy();
            return new Sigmoid
            {
                Input = newOp.Input,
                InputGrad = newOp.InputGrad,
                Output = newOp.Output
            };
        }
    }

    public class Linear : Operation
    {
        // "Identity" activation function

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            // Pass through

            return Input;
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Pass through

            return outputGrad;
        }

        public override Linear Copy()
        {
            var newOp = base.Copy();
            return new Linear
            {
                Input = newOp.Input,
                InputGrad = newOp.InputGrad,
                Output = newOp.Output
            };
        }
    }

    public class Tanh : Operation
    {
        // Hyperbolic tangent activation function

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            return np.tanh(Input);
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            if (useGPU)
            {
                return GPUNDArrayOps.PointwiseMultiply(outputGrad, 1 - GPUNDArrayOps.PointwiseMultiply(Output, Output));
            }
            else
            {
                return outputGrad * (1 - Output * Output);
            }
        }

        public override Tanh Copy()
        {
            var newOp = base.Copy();
            return new Tanh
            {
                Input = newOp.Input,
                InputGrad = newOp.InputGrad,
                Output = newOp.Output
            };
        }
    }

    public class ReLU : Operation
    {
        // Rectified linear unit activation function

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            return np.clip(Input, 0, null);
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            var mask = Output >= 0;
            return outputGrad * mask;
        }

        public override ReLU Copy()
        {
            var newOp = base.Copy();
            return new ReLU
            {
                Input = newOp.Input,
                InputGrad = newOp.InputGrad,
                Output = newOp.Output
            };
        }
    }

    public class ParamOperation : Operation
    {
        // An Operation with parameters

        public NDArray Param { get; protected set; }
        public NDArray? ParamGrad { get; protected set; }

        public override NDArray Backward(NDArray outputGrad, bool useGPU = false)
        {
            // Calculate input gradient and parameter gradient from output gradient

            Helpers.AssertSameShape(Output, outputGrad);

            InputGrad = CalcInputGrad(outputGrad, useGPU);
            ParamGrad = CalcParamGrad(outputGrad, useGPU);

            Helpers.AssertSameShape(Input, InputGrad);
            Helpers.AssertSameShape(Param, ParamGrad);

            return InputGrad;
        }

        public override void SetParams(NDArray newParams)
        {
            Param = newParams.Clone();
        }

        protected virtual NDArray CalcParamGrad(NDArray outputGrad, bool useGPU = false)
        {
            // CalcParamGrad() must be defined for each subclass

            throw new NotImplementedException();
        }

        public override ParamOperation Copy()
        {
            var newOp = base.Copy();
            NDArray newParam = Param.Clone();
            return new ParamOperation(newParam)
            {
                Input = newOp.Input,
                InputGrad = newOp.InputGrad,
                Output = newOp.Output,
                ParamGrad = this.ParamGrad
            };
        }

        public ParamOperation() { }

        public ParamOperation(NDArray param)
        {
            Param = param.Clone();
        }
    }

    public class WeightMultiply : ParamOperation
    {
        // Weight multiplication operation for a neural network

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            // Compute output
            if (useGPU)
            {
                return GPUNDArrayOps.MatrixMultiply(Input, Param);
            }
            else
            {
                return np.dot(Input, Param);
            }
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Compute input gradient

            if (useGPU)
            {
                return GPUNDArrayOps.MatrixMultiply(outputGrad, np.transpose(Param, [1, 0]));
            }
            else
            {
                return np.dot(outputGrad, np.transpose(Param, [1, 0]));
            }
        }

        protected override NDArray CalcParamGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Compute parameter gradient

            if (useGPU)
            {
                return GPUNDArrayOps.MatrixMultiply(np.transpose(Input, [1, 0]), outputGrad);
            }
            else
            {
                return np.dot(np.transpose(Input, [1, 0]), outputGrad);
            }
        }

        public override WeightMultiply Copy()
        {
            var newParamOp = base.Copy();
            return new WeightMultiply(newParamOp.Param)
            {
                Input = newParamOp.Input,
                InputGrad = newParamOp.InputGrad,
                Output = newParamOp.Output,
                ParamGrad = newParamOp.ParamGrad
            };
        }

        public WeightMultiply() { }

        public WeightMultiply(NDArray param)
        {
            Param = param.Clone();
        }
    }

    public class BiasAdd : ParamOperation
    {
        // Perform bias addition

        protected override NDArray CalcOutput(bool useGPU = false)
        {
            // Compute output

            return Input + Param;
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Compute input gradient

            if (useGPU)
            {
                return GPUNDArrayOps.PointwiseMultiply(np.ones_like(Input), outputGrad);
            }
            else
            {
                return np.ones_like(Input) * outputGrad;
            }
        }

        protected override NDArray CalcParamGrad(NDArray outputGrad, bool useGPU = false)
        {
            // Compute parameter gradient

            if (useGPU)
            {
                ParamGrad = GPUNDArrayOps.PointwiseMultiply(np.ones_like(Param), outputGrad);
            }
            else
            {
                ParamGrad = np.ones_like(Param) * outputGrad;
            }
            return Helpers.SumAlongX0(ParamGrad).reshape(1, ParamGrad.shape[1]);
        }

        public override BiasAdd Copy()
        {
            var newParamOp = base.Copy();
            return new BiasAdd(newParamOp.Param)
            {
                Input = newParamOp.Input,
                InputGrad = newParamOp.InputGrad,
                Output = newParamOp.Output,
                ParamGrad = newParamOp.ParamGrad
            };
        }

        public BiasAdd() { }

        public BiasAdd(NDArray param) : base(param)
        {
            Helpers.AssertEqualInt(param.shape[0], 1);
        }
    }

    public class Loss
    {
        // "Loss" of a neural network

        public NDArray? Prediction { get; set; }
        public NDArray? Target { get; set; }
        public NDArray? InputGrad { get; set; }

        public double Forward(NDArray prediction, NDArray target)
        {
            // Compute actual loss value

            Helpers.AssertSameShape(prediction, target);

            Prediction = prediction.Clone();
            Target = target.Clone();

            var lossValue = CalcOutput();
            return lossValue;
        }

        public NDArray Backward()
        {
            // Computes gradient loss value with respect to input to loss function

            InputGrad = CalcInputGrad();

            Helpers.AssertSameShape(Prediction, InputGrad);

            return InputGrad;
        }

        protected virtual double CalcOutput()
        {
            // CalcOutput() must be defined for each subclass

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcInputGrad()
        {
            // CalcInputGrad() must be defined for each subclass

            throw new NotImplementedException();
        }

        public virtual Loss Copy()
        {
            return new Loss { Prediction = this.Prediction, Target = this.Target, InputGrad = this.InputGrad };
        }

        public virtual void BuildFromLossData(LossData data) { }

        public Loss() { }
    }

    public class MeanSquaredError : Loss
    {
        protected override double CalcOutput()
        {
            // Compute per-observation squared error loss
            var diff = Prediction - Target;
            var pow = np.power(diff, 2);
            var sum = pow.Data<double>().Sum();
            var denom = (double)Prediction.shape[0];
            var loss = sum / denom;
            return loss;
        }

        protected override NDArray CalcInputGrad()
        {
            // Compute loss gradient with respect to input

            return 2.0 * (Prediction - Target) / Prediction.shape[0];
        }

        public override MeanSquaredError Copy()
        {
            var newLoss = base.Copy();
            return new MeanSquaredError
            {
                InputGrad = newLoss.InputGrad,
                Prediction = newLoss.Prediction,
                Target = newLoss.Target,
            };
        }
    }

    public class SoftmaxCrossEntropy : Loss
    {
        public double Eps { get; set; }
        public bool SingleClass { get; set; }
        public NDArray SoftmaxPreds { get; set; }

        protected override double CalcOutput()
        {
            if (Target.shape[1] == 0) { SingleClass = true; }

            if (SingleClass)
            {
                (Prediction, Target) = (Helpers.Normalize(Prediction), Helpers.Normalize(Target));
            }

            SoftmaxPreds = Helpers.Softmax(Prediction, axis: 1);
            SoftmaxPreds = np.clip(SoftmaxPreds, Eps, 1 - Eps);

            var loss = -1.0 * Target * np.log(SoftmaxPreds) - (1.0 - Target) * np.log(1.0 - SoftmaxPreds);

            return np.sum(loss) / Prediction.shape[0];
        }

        protected override NDArray CalcInputGrad()
        {
            if (SingleClass)
            {
                return Helpers.Unnormalize(SoftmaxPreds - Target);
            }
            else
            {
                return (SoftmaxPreds - Target) / Prediction.shape[0];
            }
        }

        public override void BuildFromLossData(LossData data)
        {
            base.BuildFromLossData(data);
            Eps = data.Eps.Value;
        }

        public SoftmaxCrossEntropy() { }

        public SoftmaxCrossEntropy(double eps = 1e-9)
        {
            Eps = eps;
            SingleClass = false;
        }
    }

    public static class Helpers
    {
        public static void AssertSameShape(NDArray arr1, NDArray arr2)
        {
            if (arr1.Shape != arr2.Shape) { throw new IncorrectShapeException(); }
        }

        public static void AssertEqualInt(int num1, int num2)
        {
            if (num1 != num2) { throw new IncorrectShapeException(); }
        }

        public static IEnumerable<(T1, T2)> Zip<T1, T2>(this IEnumerable<T1> t1, IEnumerable<T2> t2)
        {
            using var t1e = t1.GetEnumerator();
            using var t2e = t2.GetEnumerator();
            while (t1e.MoveNext() && t2e.MoveNext())
            {
                yield return (t1e.Current, t2e.Current);
            }
        }

        public static IEnumerable<(T1, T2, T3)> Zip<T1, T2, T3>(this IEnumerable<T1> t1, IEnumerable<T2> t2, IEnumerable<T3> t3)
        {
            using var t1e = t1.GetEnumerator();
            using var t2e = t2.GetEnumerator();
            using var t3e = t3.GetEnumerator();
            while (t1e.MoveNext() && t2e.MoveNext() && t3e.MoveNext())
            {
                yield return (t1e.Current, t2e.Current, t3e.Current);
            }
        }

        public static (NDArray x, NDArray y) PermuteData(NDArray x, NDArray y)
        {
            var perm = np.random.permutation(x.shape[0]);
            return (x[perm], y[perm]);
        }

        public static NDArray SumAlongX0(NDArray input)
        {
            var inputData = input.Data<double>().ToArray();
            int nRows = input.shape[0];
            int nCols = input.shape[1];
            var result = new double[nCols];

            for (int r = 0; r < nRows; r++)
            {
                for (int c = 0; c < nCols; c++)
                {
                    result[c] += inputData[r * nCols + c];
                }
            }

            return np.array(result);
        }

        public static NDArray ToNDArray(double[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            double[] flat = new double[rows * cols];

            Buffer.BlockCopy(input, 0, flat, 0, sizeof(double) * rows * cols);

            return np.array(flat, dtype: np.float64).reshape(rows, cols);
        }

        public static List<double[][]> NDArrayList2JagArrayList(List<NDArray> input)
        {
            var output = new List<double[][]>();
            foreach (var array in input)
            {
                output.Add(NDArray2JagArray(array));
            }
            return output;
        }

        public static double[][] NDArray2JagArray(NDArray input)
        {
            int rows = input.shape[0];
            int cols = input.shape[1];
            double[][] output = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                output[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    output[i][j] = input[i, j];
                }
            }
            return output;
        }

        public static double[][][] NDArray2JagArray3D(NDArray input)
        {
            int dim1 = input.shape[0];
            int dim2 = input.shape[1];
            int dim3 = input.shape[2];
            double[][][] output = new double[dim1][][];
            for (int i = 0; i < dim1; i++)
            {
                output[i] = new double[dim2][];
                for (int j = 0; j < dim2; j++)
                {
                    output[i][j] = new double[dim3];
                    for (int k = 0; k < dim3; k++)
                    {
                        output[i][j][k] = input[i, j, k];
                    }
                }
            }
            return output;
        }

        public static NDArray To2D(NDArray a, string type = "col")
        {
            if (a.ndim != 1) { throw new ArgumentException("Input tensor must be 1D"); }
            return type == "col" ? a.reshape(-1, 1) : a.reshape(1, -1);
        }

        public static NDArray Normalize(NDArray a)
        {
            var other = 1 - a;
            return np.concatenate([a, other], axis: 1);
        }

        public static NDArray Unnormalize(NDArray a)
        {
            return a[np.newaxis, 0];
        }

        public static IEnumerable<(NDArray xBatch, NDArray yBatch)> GenerateBatches(
    NDArray x, NDArray y, int size = 32)
        {
            if (x.shape[0] != y.shape[0]) { throw new IncorrectShapeException(); }

            int n = x.shape[0];

            for (int i = 0; i < n; i += size)
            {
                int end = Math.Min(i + size, n);

                NDArray xBatch = x[$"{i}:{end}"];
                NDArray yBatch = y[$"{i}:{end}"];

                yield return (xBatch, yBatch);
            }
        }

        public static (NDArray xBatch, NDArray yBatch) GenerateSetupBatch(
            NDArray x, NDArray y, int size = 32)
        {
            if (x.shape[0] != y.shape[0]) { throw new IncorrectShapeException(); }

            int n = x.shape[0];
            int end = Math.Min(size, n);
            NDArray xBatch = x[$"{0}:{end}"];
            NDArray yBatch = y[$"{0}:{end}"];

            return (xBatch, yBatch);
        }

        public static NDArray Softmax(NDArray x, int axis = 0)
        {
            return np.exp(x - LogSumExp(x, axis: axis, keepDims: true));
        }

        public static NDArray LogSumExp(NDArray x, int axis = 0, bool keepDims = false)
        {
            double max = x.max(axis: axis, keepdims: true);
            var shift = x - max;
            var expon = np.exp(shift);
            var sum = expon.sum(axis: axis, keepdims: true);
            var log = np.log(sum);
            var result = max + log;
            if (!keepDims)
            {
                result = np.squeeze(result, axis: axis);
            }
            return result;
        }

        public static NDArray StandardScale(NDArray data)
        {
            var mean = np.mean(data, axis: 0);
            var std = np.std(data, axis: 0);
            var stdData = std.Clone();
            for (int i = 0; i < std.size; i++)
            {
                if (stdData[i] == 0)
                {
                    stdData[i] = 1.0;
                }
            }
            std = stdData;
            var result = (data - mean) / std;
            return (data - mean) / std;
        }

        public static NDArray PyObjectToNDArray(PyObject pyObj)
        {
            using (Py.GIL())
            {
                dynamic np = Py.Import("numpy");
                dynamic npArr = np.array(pyObj, dtype: np.float64);
                int[] dims = ((PyObject)npArr.shape).As<int[]>();
                double[] flatData = npArr.ravel().As<double[]>();
                var nd = new NDArray(flatData, dims);

                return nd;
            }
        }
    }

    public static class GPUNDArrayOps
    {
        public static NDArray PointwiseAdd(NDArray aArray, NDArray bArray)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            var shapeA = aArray.shape.ToArray();
            var shapeB = bArray.shape.ToArray();
            var resultShape = GetBroadcastedShape(shapeA, shapeB);
            int nDims = resultShape.Length;
            int resultLength = resultShape.Aggregate(1, (acc, val) => acc * val);

            using var aBuffer = accelerator.Allocate1D(aArray.Data<double>().ToArray());
            using var bBuffer = accelerator.Allocate1D(bArray.Data<double>().ToArray());
            using var cBuffer = accelerator.Allocate1D<double>(resultLength);

            using var shapeABuffer = accelerator.Allocate1D(shapeA);
            using var shapeBBuffer = accelerator.Allocate1D(shapeB);
            using var resultShapeBuffer = accelerator.Allocate1D(resultShape);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GPUKernels.PointwiseAddKernel);

            kernel(resultLength, aBuffer.View, bBuffer.View, cBuffer.View, shapeABuffer.View, shapeBBuffer.View,
                resultShapeBuffer.View, nDims);

            accelerator.Synchronize();
            var cData = cBuffer.GetAsArray1D();

            var cArray = np.array(cData).reshape(resultShape);
            return cArray;
        }

        public static NDArray PointwiseSubtract(NDArray aArray, NDArray bArray)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            var shapeA = aArray.shape.ToArray();
            var shapeB = bArray.shape.ToArray();
            var resultShape = GetBroadcastedShape(shapeA, shapeB);
            int nDims = resultShape.Length;
            int resultLength = resultShape.Aggregate(1, (acc, val) => acc * val);

            using var aBuffer = accelerator.Allocate1D(aArray.Data<double>().ToArray());
            using var bBuffer = accelerator.Allocate1D(bArray.Data<double>().ToArray());
            using var cBuffer = accelerator.Allocate1D<double>(resultLength);

            using var shapeABuffer = accelerator.Allocate1D(shapeA);
            using var shapeBBuffer = accelerator.Allocate1D(shapeB);
            using var resultShapeBuffer = accelerator.Allocate1D(resultShape);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GPUKernels.PointwiseSubtractKernel);

            kernel(resultLength, aBuffer.View, bBuffer.View, cBuffer.View, shapeABuffer.View, shapeBBuffer.View,
                resultShapeBuffer.View, nDims);

            accelerator.Synchronize();
            var cData = cBuffer.GetAsArray1D();

            var cArray = np.array(cData).reshape(resultShape);
            return cArray;
        }

        public static NDArray PointwiseMultiply(NDArray aArray, NDArray bArray)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            var shapeA = aArray.shape.ToArray();
            var shapeB = bArray.shape.ToArray();
            var resultShape = GetBroadcastedShape(shapeA, shapeB);
            int nDims = resultShape.Length;
            int resultLength = resultShape.Aggregate(1, (acc, val) => acc * val);

            using var aBuffer = accelerator.Allocate1D(aArray.Data<double>().ToArray());
            using var bBuffer = accelerator.Allocate1D(bArray.Data<double>().ToArray());
            using var cBuffer = accelerator.Allocate1D<double>(resultLength);

            using var shapeABuffer = accelerator.Allocate1D(shapeA);
            using var shapeBBuffer = accelerator.Allocate1D(shapeB);
            using var resultShapeBuffer = accelerator.Allocate1D(resultShape);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GPUKernels.PointwiseMultiplyKernel);

            kernel(resultLength, aBuffer.View, bBuffer.View, cBuffer.View, shapeABuffer.View, shapeBBuffer.View,
                resultShapeBuffer.View, nDims);

            accelerator.Synchronize();
            var cData = cBuffer.GetAsArray1D();

            var cArray = np.array(cData).reshape(resultShape);
            return cArray;
        }

        public static NDArray PointwiseDivide(NDArray aArray, NDArray bArray)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            var shapeA = aArray.shape.ToArray();
            var shapeB = bArray.shape.ToArray();
            var resultShape = GetBroadcastedShape(shapeA, shapeB);
            int nDims = resultShape.Length;
            int resultLength = resultShape.Aggregate(1, (acc, val) => acc * val);

            using var aBuffer = accelerator.Allocate1D(aArray.Data<double>().ToArray());
            using var bBuffer = accelerator.Allocate1D(bArray.Data<double>().ToArray());
            using var cBuffer = accelerator.Allocate1D<double>(resultLength);

            using var shapeABuffer = accelerator.Allocate1D(shapeA);
            using var shapeBBuffer = accelerator.Allocate1D(shapeB);
            using var resultShapeBuffer = accelerator.Allocate1D(resultShape);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GPUKernels.PointwiseDivideKernel);

            kernel(resultLength, aBuffer.View, bBuffer.View, cBuffer.View, shapeABuffer.View, shapeBBuffer.View,
                resultShapeBuffer.View, nDims);

            accelerator.Synchronize();
            var cData = cBuffer.GetAsArray1D();

            var cArray = np.array(cData).reshape(resultShape);
            return cArray;
        }

        public static NDArray MatrixMultiply(NDArray aArray, NDArray bArray)
        {
            using var context = Context.CreateDefault();
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            var shapeA = aArray.shape.ToArray();
            var shapeB = bArray.shape.ToArray();
            var resultShape = GetMatMultBroadcastedShape(shapeA, shapeB);
            int nDims = resultShape.Length;
            int resultLength = resultShape.Aggregate(1, (acc, val) => acc * val);
            int kDim = shapeA[nDims - 1];

            using var aBuffer = accelerator.Allocate1D(aArray.Data<double>().ToArray());
            using var bBuffer = accelerator.Allocate1D(bArray.Data<double>().ToArray());
            using var cBuffer = accelerator.Allocate1D<double>(resultLength);

            using var shapeABuffer = accelerator.Allocate1D(shapeA);
            using var shapeBBuffer = accelerator.Allocate1D(shapeB);
            using var resultShapeBuffer = accelerator.Allocate1D(resultShape);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(GPUKernels.MatrixMultiplyKernel);

            kernel(resultLength, aBuffer.View, bBuffer.View, cBuffer.View,
                shapeABuffer.View, shapeBBuffer.View, resultShapeBuffer.View, nDims, kDim);

            accelerator.Synchronize();
            var cData = cBuffer.GetAsArray1D();

            var cArray = np.array(cData).reshape(resultShape);
            return cArray;
        }

        static int[] GetBroadcastedShape(int[] shapeA, int[] shapeB)
        {
            int nDimsA = shapeA.Length;
            int nDimsB = shapeB.Length;
            int maxNDims = Math.Max(nDimsA, nDimsB);
            var resultShape = new int[maxNDims];

            for (int i = 1; i <= maxNDims; i++)
            {
                int dimA = (nDimsA - i >= 0) ? shapeA[nDimsA - i] : 1;
                int dimB = (nDimsB - i >= 0) ? shapeB[nDimsB - i] : 1;

                if (dimA != dimB && dimA != 1 && dimB != 1)
                {
                    throw new InvalidOperationException("Incompatible shapes for broadcasting.");
                }
                resultShape[maxNDims - i] = Math.Max(dimA, dimB);
            }
            return resultShape;
        }

        static int[] GetMatMultBroadcastedShape(int[] shapeA, int[] shapeB)
        {
            int nDimsA = shapeA.Length;
            int nDimsB = shapeB.Length;

            int innerDimA = shapeA[nDimsA - 1];
            int outerDimA = shapeA[nDimsA - 2];
            int innerDimB = shapeB[nDimsB - 2];
            int outerDimB = shapeB[nDimsB - 1];

            if (innerDimA != innerDimB)
            {
                throw new InvalidOperationException(
                    "Incompatible shapes for matrix multiplication. Inner dimensions must match.");
            }

            int maxNDims = Math.Max(nDimsA, nDimsB);
            var resultShape = new int[maxNDims];

            resultShape[maxNDims - 2] = outerDimA;
            resultShape[maxNDims - 1] = outerDimB;

            for (int i = 1; i <= maxNDims - 2; i++)
            {
                int dimA = (nDimsA - 2 - i >= 0) ? shapeA[nDimsA - 2 - i] : 1;
                int dimB = (nDimsB - 2 - i >= 0) ? shapeB[nDimsB - 2 - i] : 1;

                if (dimA != dimB && dimA != 1 && dimB != 1)
                {
                    throw new InvalidOperationException("Incompatible batch shapes for broadcasting.");
                }
                resultShape[maxNDims - 2 - i] = Math.Max(dimA, dimB);
            }

            return resultShape;
        }
    }

    public static class GPUKernels
    {
        public static void PointwiseAddKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            ArrayView<double> c,
            ArrayView<int> shapeA,
            ArrayView<int> shapeB,
            ArrayView<int> resultShape,
            int nDims)
        {
            int linearIndex = index.X;

            int indexA = 0;
            int nDimsA = (int)shapeA.Length;
            int remainingIndexA = linearIndex;
            for (int i = nDims - 1; i >= 0; i--)
            {
                int coordA = remainingIndexA % resultShape[i];

                int dimA = (nDims - 1 - i < nDimsA) ? shapeA[nDims - 1 - (nDims - i)] : 1;
                if (dimA == 1) { coordA = 0; }

                remainingIndexA /= resultShape[i];

                int strideA = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideA *= resultShape[j];
                }
                indexA += coordA + strideA;
            }

            int indexB = 0;
            int nDimsB = (int)shapeB.Length;
            int remainingIndexB = linearIndex;
            for (int i = 0; i < nDims; i++)
            {
                int coordB = remainingIndexB % resultShape[i];

                int dimB = (nDims - 1 - i < nDimsB) ? shapeB[nDims - 1 - (nDims - i)] : 1;
                if (dimB == 1) { coordB = 0; }

                remainingIndexB /= resultShape[i];

                int strideB = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideB *= resultShape[j];
                }
                indexB += coordB + strideB;
            }

            c[linearIndex] = a[indexA] + b[indexB];
        }

        public static void PointwiseSubtractKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            ArrayView<double> c,
            ArrayView<int> shapeA,
            ArrayView<int> shapeB,
            ArrayView<int> resultShape,
            int nDims)
        {
            int linearIndex = index.X;

            int indexA = 0;
            int nDimsA = (int)shapeA.Length;
            int remainingIndexA = linearIndex;
            for (int i = nDims - 1; i >= 0; i--)
            {
                int coordA = remainingIndexA % resultShape[i];

                int dimA = (nDims - 1 - i < nDimsA) ? shapeA[nDims - 1 - (nDims - i)] : 1;
                if (dimA == 1) { coordA = 0; }

                remainingIndexA /= resultShape[i];

                int strideA = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideA *= resultShape[j];
                }
                indexA += coordA + strideA;
            }

            int indexB = 0;
            int nDimsB = (int)shapeB.Length;
            int remainingIndexB = linearIndex;
            for (int i = 0; i < nDims; i++)
            {
                int coordB = remainingIndexB % resultShape[i];

                int dimB = (nDims - 1 - i < nDimsB) ? shapeB[nDims - 1 - (nDims - i)] : 1;
                if (dimB == 1) { coordB = 0; }

                remainingIndexB /= resultShape[i];

                int strideB = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideB *= resultShape[j];
                }
                indexB += coordB + strideB;
            }

            c[linearIndex] = a[indexA] - b[indexB];
        }

        public static void PointwiseMultiplyKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            ArrayView<double> c,
            ArrayView<int> shapeA,
            ArrayView<int> shapeB,
            ArrayView<int> resultShape,
            int nDims)
        {
            int linearIndex = index.X;

            int indexA = 0;
            int nDimsA = (int)shapeA.Length;
            int remainingIndexA = linearIndex;
            for (int i = nDims - 1; i >= 0; i--)
            {
                int coordA = remainingIndexA % resultShape[i];

                int dimA = (nDims - 1 - i < nDimsA) ? shapeA[nDims - 1 - (nDims - i)] : 1;
                if (dimA == 1) { coordA = 0; }

                remainingIndexA /= resultShape[i];

                int strideA = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideA *= resultShape[j];
                }
                indexA += coordA + strideA;
            }

            int indexB = 0;
            int nDimsB = (int)shapeB.Length;
            int remainingIndexB = linearIndex;
            for (int i = 0; i < nDims; i++)
            {
                int coordB = remainingIndexB % resultShape[i];

                int dimB = (nDims - 1 - i < nDimsB) ? shapeB[nDims - 1 - (nDims - i)] : 1;
                if (dimB == 1) { coordB = 0; }

                remainingIndexB /= resultShape[i];

                int strideB = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideB *= resultShape[j];
                }
                indexB += coordB + strideB;
            }

            c[linearIndex] = a[indexA] * b[indexB];
        }

        public static void PointwiseDivideKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            ArrayView<double> c,
            ArrayView<int> shapeA,
            ArrayView<int> shapeB,
            ArrayView<int> resultShape,
            int nDims)
        {
            int linearIndex = index.X;

            int indexA = 0;
            int nDimsA = (int)shapeA.Length;
            int remainingIndexA = linearIndex;
            for (int i = nDims - 1; i >= 0; i--)
            {
                int coordA = remainingIndexA % resultShape[i];

                int dimA = (nDims - 1 - i < nDimsA) ? shapeA[nDims - 1 - (nDims - i)] : 1;
                if (dimA == 1) { coordA = 0; }

                remainingIndexA /= resultShape[i];

                int strideA = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideA *= resultShape[j];
                }
                indexA += coordA + strideA;
            }

            int indexB = 0;
            int nDimsB = (int)shapeB.Length;
            int remainingIndexB = linearIndex;
            for (int i = 0; i < nDims; i++)
            {
                int coordB = remainingIndexB % resultShape[i];

                int dimB = (nDims - 1 - i < nDimsB) ? shapeB[nDims - 1 - (nDims - i)] : 1;
                if (dimB == 1) { coordB = 0; }

                remainingIndexB /= resultShape[i];

                int strideB = 1;
                for (int j = i + 1; j < nDims; j++)
                {
                    strideB *= resultShape[j];
                }
                indexB += coordB + strideB;
            }

            c[linearIndex] = a[indexA] / b[indexB];
        }

        public static void MatrixMultiplyKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            ArrayView<double> c,
            ArrayView<int> shapeA,
            ArrayView<int> shapeB,
            ArrayView<int> resultShape,
            int nDims,
            int kDim)
        {
            int linearIndex = index.X;

            double sum = 0.0;

            for (int k = 0; k < kDim; k++)
            {
                int indexA = 0;
                int remainingIndexA = linearIndex;
                int ndimsA = (int)shapeA.Length;
                for (int i = nDims - 1; i >= 0; i--)
                {
                    int coord;
                    if (i == nDims - 1)
                    {
                        coord = k;
                    }
                    else if (i == nDims - 2)
                    {
                        coord = remainingIndexA % resultShape[i];
                    }
                    else
                    {
                        int dimA = (nDims - 1 - i < ndimsA) ? shapeA[nDims - 1 - (nDims - i)] : 1;
                        coord = (dimA == 1) ? 0 : remainingIndexA % resultShape[i];
                    }

                    int strideA = 1;
                    for (int j = i + 1; j < nDims; j++)
                    {
                        strideA *= resultShape[j];
                    }
                    indexA += coord * strideA;

                    remainingIndexA /= resultShape[i];
                }

                int indexB = 0;
                int remainingIndexB = linearIndex;
                int ndimsB = (int)shapeB.Length;
                for (int i = nDims - 1; i >= 0; i--)
                {
                    int coord;
                    if (i == nDims - 1)
                    {
                        coord = remainingIndexB % resultShape[i];
                    }
                    else if (i == nDims - 2)
                    {
                        coord = k;
                    }
                    else
                    {
                        int dimB = (nDims - 1 - i < ndimsB) ? shapeB[nDims - 1 - (nDims - i)] : 1;
                        coord = (dimB == 1) ? 0 : remainingIndexB % resultShape[i];
                    }

                    int strideB = 1;
                    for (int j = i + 1; j < nDims; j++)
                    {
                        strideB *= resultShape[j];
                    }
                    indexB += coord * strideB;

                    remainingIndexB /= resultShape[i];
                }

                sum += a[indexA] * b[indexB];
            }

            c[linearIndex] = sum;
        }
    }

    public class NNData
    {
        public List<LayerData> Layers { get; set; }
        public LossData Loss { get; set; }
        public int Seed { get; set; }

        public NNData() { }

        public NNData(List<LayerData> layers, LossData loss, int seed)
        {
            Layers = layers.ToList();
            Loss = loss;
            Seed = seed;
        }
    }

    public class LayerData
    {
        public string LayerType { get; set; }
        public List<string> Operations { get; set; }
        public List<double[][]> Params { get; set; }
        public bool First { get; set; }
        public int Neurons { get; set; }
        public int Seed { get; set; }
        public string Activation { get; set; } = "NNN.";

        public LayerData() { }

        public LayerData(string layerType, List<string> operations, List<double[][]> layerParams, bool first, int neurons, int seed, string activation = "")
        {
            LayerType = $"NNN.{layerType}";
            Operations = new List<string>();
            foreach (var op in operations)
            {
                Operations.Add($"NNN.{op}");
            }
            Params = layerParams.ToList();
            First = first;
            Neurons = neurons;
            Seed = seed;
            Activation = $"NNN.{activation}";
        }
    }

    public class LossData
    {
        public string LossType { get; set; }
        public double? Eps { get; set; }

        public LossData() { }

        public LossData(string lossType, double? eps)
        {
            LossType = $"NNN.{lossType}";
            Eps = eps;
        }
    }

    public class BostonDataset
    {
        public double[][]? Data { get; set; }
        public double[]? Target { get; set; }
        public string[]? FeatureNames { get; set; }
    }

    public class BostonLoader
    {
        static readonly string Url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";

        public static async Task<BostonDataset> LoadAsync()
        {
            string csvFile = "boston.csv";

            if (!File.Exists(csvFile))
            {
                using var client = new HttpClient();
                var csvData = await client.GetStringAsync(Url);
                await File.WriteAllTextAsync(csvFile, csvData);
            }

            using var reader = new StreamReader(csvFile);
            using var csv = new CsvReader(reader, new CsvHelper.Configuration.CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLowerInvariant()
            });
            var records = csv.GetRecords<BostonRow>().ToList();

            var data = records.Select(r => new double[]
            {
                r.CRIM, r.ZN, r.INDUS, r.CHAS, r.NOX, r.RM,
                r.AGE, r.DIS, r.RAD, r.TAX, r.PTRATIO, r.B, r.LSTAT
            }).ToArray();

            var target = records.Select(r => r.MEDV).ToArray();

            var featureNames = new string[]
            {
                "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
            };

            return new BostonDataset
            {
                Data = data,
                Target = target,
                FeatureNames = featureNames
            };
        }
    }

    public class BostonRow
    {
        public double CRIM { get; set; }
        public double ZN { get; set; }
        public double INDUS { get; set; }
        public double CHAS { get; set; }
        public double NOX { get; set; }
        public double RM { get; set; }
        public double AGE { get; set; }
        public double DIS { get; set; }
        public double RAD { get; set; }
        public double TAX { get; set; }
        public double PTRATIO { get; set; }
        public double B { get; set; }
        public double LSTAT { get; set; }
        public double MEDV { get; set; }
    }

    public class MNISTLoader
    {
        public static async Task<MNISTData> LoadMNIST()
        {
            string jsonFile = "MNIST.json";
            MNISTData mnistData;
            if (!File.Exists(jsonFile))
            {
                using (Py.GIL())
                {
                    string pythonDLL = @"C:\Users\paren\AppData\Local\Programs\Python\Python38\python38.dll";
                    Runtime.PythonDLL = pythonDLL;
                    PythonEngine.PythonPath += @";C:\Users\paren\AppData\Local\Programs\Python\Python38\Lib\site-packages";
                    PythonEngine.Initialize();

                    Console.WriteLine("Downloading MNIST dataset...");
                    var (xTrain, yTrain, xTest, yTest) = (np.zeros(), np.zeros(), np.zeros(), np.zeros());
                    dynamic mnist = Py.Import("keras.datasets.mnist");
                    dynamic data = mnist.load_data();

                    dynamic train = data[0];
                    dynamic test = data[1];

                    Console.WriteLine("Download finished, converting MNIST data...");
                    // Convert Python NumPy arrays to NumSharp NDArrays
                    xTrain = Helpers.PyObjectToNDArray(train[0]);
                    yTrain = Helpers.PyObjectToNDArray(train[1]);
                    xTest = Helpers.PyObjectToNDArray(test[0]);
                    yTest = Helpers.PyObjectToNDArray(test[1]);
                    Console.WriteLine("Conversion finished, saving JSON file...");
                    var (jagXTrain, jagYTrain, jagXTest, jagYTest) = (Helpers.NDArray2JagArray3D(xTrain), yTrain.ToArray<double>(),
                        Helpers.NDArray2JagArray3D(xTest), yTest.ToArray<double>());
                    MNISTData saveData = new MNISTData(jagXTrain, jagYTrain, jagXTest, jagYTest);
                    string jsonString = JsonSerializer.Serialize(saveData);
                    File.WriteAllText(jsonFile, jsonString);
                    Console.WriteLine("JSON file saved...");
                }
            }
            Console.WriteLine("Reading JSON file...");
            string json = File.ReadAllText(jsonFile);
            mnistData = JsonSerializer.Deserialize<MNISTData>(json);
            Console.WriteLine("Reading finished...");
            return mnistData;
        }
    }

    public class MNISTData
    {
        public double[][][] XTrain { get; set; }
        public double[] YTrain { get; set; }
        public double[][][] XTest { get; set; }
        public double[] YTest { get; set; }

        public MNISTData() { }

        public MNISTData(double[][][] xTrain, double[] yTrain, double[][][] xTest, double[] yTest)
        {
            XTrain = xTrain.ToArray();
            YTrain = yTrain.ToArray();
            XTest = xTest.ToArray();
            YTest = yTest.ToArray();
        }
    }
}