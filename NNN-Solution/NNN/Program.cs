using NumSharp;
using NumSharp.Extensions;
using System.Linq;

namespace NNN
{
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
            List<Layer> layersCopy = new List<Layer>();
            foreach (var layer in Layers)
            {
                layersCopy.Add(layer.Copy());
            }
            return new NeuralNetwork(layersCopy, Loss, Seed);
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

    public class Trainer
    {
        // Trains a neural network

        protected NeuralNetwork Net { get; set; }
        protected Optimizer Optim { get; set; }
        protected double BestLoss { get; set; }

        public void Fit(NDArray xTrain, NDArray yTrain, NDArray xTest, NDArray yTest, int epochs = 100, int evalEvery = 10,
            int batchSize = 32, int seed = 1, bool restart = true)
        {
            // Fits neural network on training data for certain number of epochs. Every "evalEvery" epochs, evaluates
            // neural network on testing data.

            np.random.seed(seed);

            if (restart)
            {
                foreach (var layer in Net.Layers)
                {
                    layer.First = true;
                }
            }

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
                        Console.WriteLine($"Validation loss after {e + 1} epochs is {loss:.3f}");
                        BestLoss = loss;
                    }
                    else
                    {
                        Console.WriteLine($"Loss increased after epoch {e + 1}, final loss was {BestLoss:.3f}, using the model from epoch {e + 1 - evalEvery}");
                        Net = lastModel;
                        Optim.Net = Net;
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

            foreach (var(param, paramGrad) in Helpers.Zip(Net.CalcParams(), Net.CalcParamGrads()))
            {
                newParams.Add(param - LR * paramGrad);
            }
            Net.SetParams(newParams);
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
            if (First)
            {
                SetupLayer(input);
                First = false;
            }

            Input = input.Clone();

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

        protected virtual void SetupLayer(int numIn)
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
            return new Layer
            {
                First = this.First,
                Input = this.Input,
                InputGrad = this.InputGrad,
                Neurons = this.Neurons,
                Operations = this.Operations.ToList(),
                Output = this.Output,
                ParamGrads = this.ParamGrads.ToList(),
                Params = this.Params.ToList()
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
        public Operation Activation { get; set; }

        protected override void SetupLayer(int numIn)
        {
            // Defines operations of fully connected layer

            if (Seed != default) { np.random.seed(Seed); }

            Params = new List<NDArray>();

            // weights
            Params.Add(np.random.randn(Input.shape[1], Neurons));

            // bias
            Params.Add(np.random.randn(1, Neurons));

            Operations = [new WeightMultiply(Params[0]), new BiasAdd(Params[1]), Activation];
        }

        public override Dense Copy()
        {
            return new Dense
            {
                Activation = this.Activation,
                Input = this.Input,
                First = this.First,
                InputGrad = this.InputGrad,
                Neurons = this.Neurons,
                Operations = this.Operations.ToList(),
                Output = this.Output,
                ParamGrads = this.ParamGrads.ToList(),
                Params = this.Params.ToList(),
                Seed = this.Seed
            };
        }

        public Dense() { }

        public Dense(int neurons, Operation activation) : base(neurons)
        {
            Activation = activation;
        }
    }

    public class Operation
    {
        // Base class for operations

        protected NDArray? Input { get; set; }
        protected NDArray? Output { get; set; }
        protected NDArray? InputGrad { get; set; }

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
    }

    public class ParamOperation : Operation
    {
        // An Operation with parameters

        public NDArray Param { get; protected set; }
        public NDArray? ParamGrad { get; protected set; }

        public override NDArray Backward(NDArray outputGrad)
        {
            // Calculate input gradient and parameter gradient from output gradient

            Helpers.AssertSameShape(Input, outputGrad);

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
            return np.sum(ParamGrad, axis: 0).reshape(1, ParamGrad.shape[1]);
        }

        public BiasAdd(NDArray param) : base(param)
        {
            Helpers.AssertEqualInt(param.shape[0], 1);
        }
    }

    public class Loss
    {
        // "Loss" of a neural network

        protected NDArray? Prediction { get; set; }
        protected NDArray? Target { get; set; }
        protected NDArray? InputGrad { get; set; }

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

        protected virtual float CalcOutput()
        {
            // CalcOutput() must be defined for each subclass

            throw new NotImplementedException();
        }

        protected virtual NDArray CalcInputGrad()
        {
            // CalcInputGrad() must be defined for each subclass

            throw new NotImplementedException();
        }
    }

    public class MeanSquaredError : Loss
    {
        protected override float CalcOutput()
        {
            // Compute per-observation squared error loss

            var loss = np.sum(np.power(Prediction - Target, 2)) / Prediction.shape[0];
            return loss;
        }

        protected override NDArray CalcInputGrad()
        {
            // Compute loss gradient with respect to input

            return 2.0 * (Prediction - Target) / Prediction.shape[0];
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
    }
}