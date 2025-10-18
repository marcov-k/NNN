using NNN;
using NumSharp;
using System.Globalization;
using CsvHelper;

NeuralNetwork lr = new NeuralNetwork(layers: [new Dense(neurons: 1, activation: new Linear())],
    loss: new MeanSquaredError(), seed: 20190501);

NeuralNetwork nn = new NeuralNetwork(layers: [new Dense(neurons: 13, activation: new Sigmoid()),
    new Dense(neurons: 1, activation: new Linear())], loss: new MeanSquaredError(), seed: 20190501);

NeuralNetwork dl = new NeuralNetwork(layers: [new Dense(neurons: 13, activation: new Sigmoid()),
    new Dense(neurons: 13, activation: new Sigmoid()), new Dense(neurons: 1, activation: new Linear())],
    loss: new MeanSquaredError(), seed: 20190501);

var boston = await BostonLoader.LoadAsync();
var data = Helpers.ToNDArray(boston.Data);
var target = Helpers.ToNDArray(boston.Target);
var features = boston.FeatureNames;

data = StandardScale(data);

var (xTrain, xTest, yTrain, yTest) = TrainTestSplit(data, target, testSize: 0.3, seed: 80718);
yTrain = To2D(yTrain);
yTest = To2D(yTest);

var trainer = new Trainer(lr, new SGD(lr: 0.01));

Console.WriteLine("Training linear regression model...");
trainer.Fit(xTrain, yTrain, xTest, yTest, epochs: 50, evalEvery: 10, seed: 20190501);
Console.WriteLine();
EvalRegressionModel(lr, xTest, yTest);
Console.WriteLine();

trainer = new Trainer(nn, new SGD(lr: 0.01));

Console.WriteLine("Training neural network model...");
trainer.Fit(xTrain, yTrain, xTest, yTest, epochs: 50, evalEvery: 10, seed: 20190501);
Console.WriteLine();
EvalRegressionModel(nn, xTest, yTest);
Console.WriteLine();

trainer = new Trainer(dl, new SGD(lr: 0.01));

Console.WriteLine("Training deep learning model...");
trainer.Fit(xTrain, yTrain, xTest, yTest, epochs: 50, evalEvery: 10, seed: 20190501);
Console.WriteLine();
EvalRegressionModel(dl, xTest, yTest);
Console.WriteLine();
Console.WriteLine("Press any key to close...");
Console.ReadKey();

static NDArray To2D(NDArray a, string type = "col")
{
    if (a.ndim != 1) { throw new ArgumentException("Input tesnor must be 1D"); }
    return type == "col" ? a.reshape(-1, 1) : a.reshape(1, -1);
}

static NDArray StandardScale(NDArray data)
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

static (NDArray xTrain, NDArray xTest, NDArray yTrain, NDArray yTest)
    TrainTestSplit(NDArray x, NDArray y, double testSize = 0.3, int seed = 80718)
{
    var rnd = new Random(seed);
    int n = x.shape[0];
    var indexes = np.arange(n);
    indexes = np.random.permutation(indexes);
    int testCount = (int)(n * testSize);
    var testIdx = indexes[$":{testCount}"];
    var trainIdx = indexes[$"{testCount}:"];
    var xTrain = x[trainIdx];
    var xTest = x[testIdx];
    var yTrain = y[trainIdx];
    var yTest = y[testIdx];
    return (xTrain, xTest, yTrain, yTest);
}

static void EvalRegressionModel(NeuralNetwork model, NDArray xTest, NDArray yTest)
{
    // Compute MAE and RMSE for neural network

    var preds = model.Forward(xTest);
    preds = preds.reshape(-1, 1);
    var mae = MAE(yTest, preds);
    var rmse = RMSE(yTest, preds);
    Console.WriteLine($"Mean absolute error: {mae}");
    Console.WriteLine();
    Console.WriteLine($"Root mean squared error: {rmse}");
}

static double MAE(NDArray yTrue, NDArray yPred)
{
    // Compute mean absolute error for neural network

    return np.abs(yTrue - yPred).mean();
}

static double RMSE(NDArray yTrue, NDArray yPred)
{
    // Compute root mean squared error for neural network

    return np.sqrt(np.mean(np.power(yTrue - yPred, 2)));
}

namespace NNN
{
    public class Trainer
    {
        // Trains a neural network

        protected NeuralNetwork Net { get; set; }
        protected Optimizer Optim { get; set; }
        protected double BestLoss { get; set; }

        public void Fit(NDArray xTrain, NDArray yTrain, NDArray xTest, NDArray yTest, int epochs = 100, int evalEvery = 10,
            int batchSize = 32, int seed = 1, bool restart = true)
        {
            // Fits neural network on training data for certain number of epochs
            // Every "evalEvery" epochs, evaluates neural network on testing data

            np.random.seed(seed);

            if (restart)
            {
                foreach (var layer in Net.Layers)
                {
                    layer.First = true;
                }
            }

            var setupBatch = GenerateSetupBatch(xTrain, yTrain).xBatch;
            Net.SetupLayers(setupBatch);

            var lastModel = Net.Copy();
            for (int e = 0; e < epochs; e++)
            {
                if ((e + 1) % evalEvery == 0)
                {
                    lastModel = Net.Copy();
                }

                (xTrain, yTrain) = Helpers.PermuteData(xTrain, yTrain);

                var batchGenerator = GenerateBatches(xTrain, yTrain, batchSize);

                foreach (var (xBatch, yBatch) in batchGenerator)
                {
                    Net.TrainBatch(xBatch, yBatch);
                    Optim.Step();
                }

                if ((e + 1) % evalEvery == 0)
                {
                    var testPreds = Net.Forward(xTest);
                    var loss = Net.Loss.Forward(testPreds, yTest);

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
            }
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
        public NeuralNetwork? Net { get; set; }

        public virtual void Step()
        {
            // Step() must be defined for every optimizer
        }

        public Optimizer(double lr = 0.01)
        {
            // Every optimizer must have initial learning rate

            LR = lr;
        }
    }

    public class SGD(double lr = 0.01) : Optimizer(lr)
    {
        // Stochastic gradient descent optimizer

        public override void Step()
        {
            // For each parameter, adjust in appropriate direction,
            // with magnitude of adjustment based on learning rate

            List<NDArray> newParams = new List<NDArray>();

            foreach (var (param, paramGrad) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads()))
            {
                newParams.Add(param - LR * paramGrad);
            }
            Net.SetParams(newParams);
        }
    }

    public class NeuralNetwork
    {
        // A neural network consisting of multiple "layers"

        public List<Layer> Layers { get; set; }
        public Loss Loss { get; set; }
        public int Seed { get; set; }

        public NDArray Forward(NDArray xBatch)
        {
            // Pass data forward through layers

            var xOut = xBatch.Clone();
            foreach (var layer in Layers)
            {
                xOut = layer.Forward(xOut);
            }

            return xOut;
        }

        public void Backward(NDArray lossGrad)
        {
            // Pass data backward through layers

            var grad = lossGrad;
            var revLayers = Layers.ToList();
            revLayers.Reverse();
            foreach (var layer in revLayers)
            {
                grad = layer.Backward(grad);
            }
        }

        public double TrainBatch(NDArray xBatch, NDArray yBatch)
        {
            // Pass data forward through layers
            // Compute loss
            // Pass data backward through layers

            xBatch = xBatch.Clone();
            yBatch = yBatch.Clone();

            var predictions = Forward(xBatch);
            var loss = Loss.Forward(predictions, yBatch);
            var lossGrad = Loss.Backward();
            Backward(lossGrad);

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
            foreach (var layer in Layers)
            {
                layer.SetupForInput(input);
            }
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

        public NDArray Forward(NDArray input)
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
                input = operation.Forward(input);
            }
            Output = input;

            return Output;
        }

        public NDArray Backward(NDArray outputGrad)
        {
            // Send output gradient backward through operations

            Helpers.AssertSameShape(Output, outputGrad);

            outputGrad = outputGrad.Clone();
            var revOps = Operations.ToList();
            revOps.Reverse();
            foreach (var operation in revOps)
            {
                outputGrad = operation.Backward(outputGrad);
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

        public Layer() { }

        public Layer(int neurons)
        {
            // Number of "neurons" roughly corresponds to "breadth" of layer

            Neurons = neurons;
            First = true;
            Params = new List<NDArray>();
            ParamGrads = new List<NDArray>();
            Operations = new List<Operation>();
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

        public NDArray Forward(NDArray input)
        {
            // Send input forward through operation

            Input = input;
            Output = CalcOutput();
            return Output;
        }

        public virtual NDArray Backward(NDArray outputGrad)
        {
            // Calculate input gradient from output gradient

            Helpers.AssertSameShape(Output, outputGrad);

            InputGrad = CalcInputGrad(outputGrad);

            Helpers.AssertSameShape(Input, InputGrad);

            return InputGrad;
        }

        public virtual void SetParams(NDArray newParams)
        {
            // SetParams() must be defined only for the ParamOperation subclass

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcOutput()
        {
            // CalcOutput() must be defined for each operation

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcInputGrad(NDArray outputGrad)
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

        protected override NDArray CalcOutput()
        {
            // Compute output

            return 1.0 / (1.0 + np.exp(-1.0 * Input));
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad)
        {
            // Compute input gradient

            var sigmoidBackward = Output * (1.0 - Output);
            InputGrad = sigmoidBackward * outputGrad;
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

        protected override NDArray CalcOutput()
        {
            // Pass through

            return Input;
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad)
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

    public class ParamOperation : Operation
    {
        // An Operation with parameters

        public NDArray Param { get; protected set; }
        public NDArray? ParamGrad { get; protected set; }

        public override NDArray Backward(NDArray outputGrad)
        {
            // Calculate input gradient and parameter gradient from output gradient

            Helpers.AssertSameShape(Output, outputGrad);

            InputGrad = CalcInputGrad(outputGrad);
            ParamGrad = CalcParamGrad(outputGrad);

            Helpers.AssertSameShape(Input, InputGrad);
            Helpers.AssertSameShape(Param, ParamGrad);

            return InputGrad;
        }

        public override void SetParams(NDArray newParams)
        {
            Param = newParams.Clone();
        }

        protected virtual NDArray CalcParamGrad(NDArray outputGrad)
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

        public ParamOperation(NDArray param)
        {
            Param = param.Clone();
        }
    }

    public class WeightMultiply(NDArray param) : ParamOperation(param)
    {
        // Weight multiplication operation for a neural network

        protected override NDArray CalcOutput()
        {
            // Compute output

            return np.dot(Input, Param);
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad)
        {
            // Compute input gradient

            return np.dot(outputGrad, np.transpose(Param, [1, 0]));
        }

        protected override NDArray CalcParamGrad(NDArray outputGrad)
        {
            // Compute parameter gradient

            return np.dot(np.transpose(Input, [1, 0]), outputGrad);
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
    }

    public class BiasAdd : ParamOperation
    {
        // Perform bias addition

        protected override NDArray CalcOutput()
        {
            // Compute output

            return Input + Param;
        }

        protected override NDArray CalcInputGrad(NDArray outputGrad)
        {
            // Compute input gradient

            return np.ones_like(Input) * outputGrad;
        }

        protected override NDArray CalcParamGrad(NDArray outputGrad)
        {
            // Compute parameter gradient

            ParamGrad = np.ones_like(Param) * outputGrad;
            return Helpers.SumAlongX0(ParamGrad).reshape(1, ParamGrad.shape[1]);
        }

        public BiasAdd(NDArray param) : base(param)
        {
            Helpers.AssertEqualInt(param.shape[0], 1);
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
    }

    public class MeanSquaredError : Loss
    {
        protected override double CalcOutput()
        {
            // Compute per-observation squared error loss
            var diff = Prediction - Target;
            var pow = np.power(diff, 2);
            var sum = pow.Data<double>().Sum();
            var denom = Prediction.shape[0];
            var doubleDenom = (double)denom;
            var loss = sum / doubleDenom;
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
                Target = newLoss.Target
            };
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

        public static NDArray ToNDArray(double[][] input)
        {
            return np.array(input, np.float64);
        }

        public static NDArray ToNDArray(double[] input)
        {
            int cols = input.Length;
            var result = np.array(input, np.float64);
            return result;
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
}