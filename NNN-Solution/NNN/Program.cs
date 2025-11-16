using NNN;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.optim;
using static TorchSharp.torchvision;

var train = datasets.MNIST(root: "mnistdata", train: true, download: true);
var test = datasets.MNIST(root: "mnistdata", train: false, download: true);

var trainLoader = torch.utils.data.DataLoader(train, batchSize: 32, shuffle: true);
var testLoader = torch.utils.data.DataLoader(test, batchSize: 32, shuffle: false);

var model = new MNISTConvNet();
var criterion = new CrossEntropyLoss();
var optimizer = SGD(model.parameters(), learningRate: 0.001f, momentum: 0.9f);

var paramList = model.parameters().ToList();
Console.WriteLine($"model.parameters() count: {paramList.Count}");
long paramCount = paramList.Sum(p => (long)p.numel());
Console.WriteLine($"Total model parameters: {paramCount}");
foreach (var g in optimizer.ParamGroups)
    Console.WriteLine($"Param group LR: {g.LearningRate}");

var trainer = new TorchSharpTrainer(model, optimizer, criterion);
trainer.Fit(trainDataloader: trainLoader, testDataloader: testLoader, epochs: 1, evalEvery: 1);

Console.WriteLine($"Final accuracy: {Math.Round(TestAccuracy(model, testLoader), 4) * 100.0f}%");

static float TestAccuracy(TorchSharpModel model, DataLoader testLoader)
{
    using (torch.no_grad())
    {
        model.eval();
        var accuracies = new List<float>();
        foreach (var item in testLoader)
        {
            var xBatch = item["data"];
            var yBatch = item["label"];
            xBatch = TorchSharpTrainer.NormalizeBatch(xBatch);

            var output = model.Forward(xBatch);
            var predictions = output.argmax(1);
            var correct = predictions.eq(yBatch);
            var accuracy = correct.to_type(torch.ScalarType.Float32).mean().ToSingle();
            accuracies.Add(accuracy);
        }
        return torch.from_array(accuracies.ToArray()).mean().item<float>();
    }
}

namespace NNN
{
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;
    using static TorchSharp.torch.optim;
    using static TorchSharp.torch.optim.lr_scheduler;

    public class TorchSharpTrainer
    {
        TorchSharpModel Model;
        Optimizer Optim;
        Loss<Tensor, Tensor, Tensor> Loss;

        public void Fit(Tensor? xTrain = null, Tensor? yTrain = null, Tensor? xTest = null, Tensor? yTest = null, DataLoader? trainDataloader = null,
            DataLoader? testDataloader = null, int epochs = 100, int evalEvery = 10, int batchSize = 32, int? finalLRExp = null)
        {
            double initLR = Optim.ParamGroups.First().LearningRate;
            double decay;
            LRScheduler? scheduler = null;
            if (finalLRExp != null)
            {
                decay = Math.Pow(finalLRExp.Value / initLR, 1.0 / (epochs + 1));
                scheduler = ExponentialLR(Optim, gamma: decay);
            }

            for (int e = 0; e < epochs; e++)
            {
                scheduler?.step();

                if (trainDataloader == null)
                {
                    (xTrain, yTrain) = Utils.PermuteData(xTrain, yTrain);

                    var batchGenerator = GenerateBatches(xTrain, yTrain, batchSize);

                    Model.train();

                    foreach (var (xBatch, yBatch) in batchGenerator)
                    {
                        var normXBatch = NormalizeBatch(xBatch);
                        Optim.zero_grad();

                        var output = Model.Forward(normXBatch);

                        var loss = Loss.call(output, yBatch);
                        loss.backward();
                        Optim.step();
                    }

                    if (e % evalEvery == 0)
                    {
                        using (no_grad())
                        {
                            Model.eval();
                            var output = Model.Forward(xTest);
                            var loss = Loss.call(output, yTest);
                            Console.WriteLine($"The loss after {e + 1} epochs was {loss.item<float>()}");
                        }
                    }
                }
                else
                {
                    Model.train();
                    int batchIdx = 0;
                    foreach (var item in trainDataloader)
                    {
                        var xBatch = item["data"];
                        var yBatch = item["label"];
                        xBatch = NormalizeBatch(xBatch);

                        Optim.zero_grad();

                        var output = Model.Forward(xBatch);

                        var loss = Loss.call(output, yBatch);
                        if (batchIdx % 50 == 0) Console.WriteLine($"Epoch {e} Batch {batchIdx} loss: {loss.item<float>()}");
                        loss.backward();
                        Optim.step();
                        batchIdx++;
                    }

                    if (e % evalEvery == 0)
                    {
                        using (no_grad())
                        {
                            Model.eval();
                            var losses = new List<Tensor>();
                            foreach (var item in testDataloader)
                            {
                                var xBatch = item["data"];
                                var yBatch = item["label"];
                                xBatch = NormalizeBatch(xBatch);

                                var output = Model.Forward(xBatch);
                                var loss = Loss.call(output, yBatch);
                                losses.Add(loss);
                            }
                            var lossVals = losses.Select((x) => x.item<float>()).ToArray();
                            var avgLoss = lossVals.Average();
                            Console.WriteLine($"The loss after {e + 1} epochs was {Math.Round(avgLoss, 4)}");
                        }
                    }
                }
            }
        }

        public static Tensor NormalizeBatch(Tensor batch)
        {
            batch = batch.to_type(ScalarType.Float32).div(255.0f);
            double mean = 0.1305;
            double std = 0.3081;
            batch = batch.sub(mean).div(std);
            return batch;
        }

        IEnumerable<(Tensor x, Tensor y)> GenerateBatches(Tensor x, Tensor y, int size = 32)
        {
            long n = x.shape[0];

            for (long i = 0; i < n; i += size)
            {
                long end = Math.Min(i + size, n);

                var xBatch = x.index(TensorIndex.Slice(i, end));
                var yBatch = y.index(TensorIndex.Slice(i, end));

                yield return (xBatch, yBatch);
            }
        }

        public TorchSharpTrainer(TorchSharpModel model, Optimizer optim, Loss<Tensor, Tensor, Tensor> criterion)
        {
            Model = model;
            Optim = optim;
            Loss = criterion;
        }
    }

    public class TorchSharpModel : Module
    {
        public Tensor Forward(Tensor x, bool inference = false)
        {
            return ForwardImpl(x);
        }

        protected virtual Tensor ForwardImpl(Tensor x)
        {
            throw new NotImplementedException();
        }

        public TorchSharpModel(string name) : base(name) { }
    }

    public class MNISTConvNet : TorchSharpModel
    {
        ConvLayer Conv1;
        ConvLayer Conv2;
        DenseLayer Dense1;
        DenseLayer Dense2;

        protected override Tensor ForwardImpl(Tensor x)
        {
            x = Conv1.Forward(x);
            x = Conv2.Forward(x);

            x = Dense1.Forward(x);
            x = Dense2.Forward(x);

            return x;
        }

        public MNISTConvNet() : base("MNISTConvNet")
        {
            Conv1 = new ConvLayer(inChannels: 1, outChannels: 14, filterSize: 5, activation: ReLU());
            Conv2 = new ConvLayer(inChannels: 14, outChannels: 7, filterSize: 5, activation: ReLU(), flatten: true);
            Dense1 = new DenseLayer(inputSize: 7 * 28 * 28, neurons: 32, activation: ReLU());
            Dense2 = new DenseLayer(inputSize: 32, neurons: 10);

            RegisterComponents();
        }
    }

    public class TorchSharpLayer : Module
    {
        public Tensor Forward(Tensor x)
        {
            return ForwardImpl(x);
        }

        protected virtual Tensor ForwardImpl(Tensor x)
        {
            throw new NotImplementedException();
        }

        public TorchSharpLayer(string name) : base(name) { }
    }

    public class DenseLayer : TorchSharpLayer
    {
        readonly Linear Linear;
        readonly IModule<Tensor, Tensor>? Activation;
        readonly IModule<Tensor, Tensor>? Dropout;

        protected override Tensor ForwardImpl(Tensor x)
        {
            x = Linear.call(x);

            if (Activation != null)
            {
                x = Activation.call(x);
            }
            if (Dropout != null)
            {
                x = Dropout.call(x);
            }

            return x;
        }

        public DenseLayer(int inputSize, int neurons, IModule<Tensor, Tensor>? activation = null, float? dropout = null) : base("DenseLayer")
        {
            Linear = Linear(inputSize, neurons);
            Activation = activation;
            if (dropout.HasValue)
            {
                Dropout = Dropout(dropout.Value);
            }

            RegisterComponents();
        }
    }

    public class ConvLayer : TorchSharpLayer
    {
        readonly IModule<Tensor, Tensor> Conv;
        readonly IModule<Tensor, Tensor>? Activation;
        readonly bool Flatten;
        readonly IModule<Tensor, Tensor>? Dropout;

        protected override Tensor ForwardImpl(Tensor x)
        {
            x = Conv.call(x);

            if (Activation != null)
            {
                x = Activation.call(x);
            }
            if (Flatten)
            {
                x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]);
            }
            if (Dropout != null)
            {
                x = Dropout.call(x);
            }

            return x;
        }

        public ConvLayer(int inChannels, int outChannels, int filterSize, IModule<Tensor, Tensor>? activation = null, float? dropout = null, bool flatten = false) : base("ConvLayer")
        {
            Conv = Conv2d(inChannels, outChannels, filterSize, padding: (int)MathF.Floor(filterSize / 2f));
            Activation = activation;
            Flatten = flatten;
            if (dropout.HasValue)
            {
                Dropout = Dropout(dropout.Value);
            }

            RegisterComponents();
        }
    }

    public static class Utils
    {
        public static void InferenceMode(nn.Module m)
        {
            m.eval();
        }

        public static (Tensor x, Tensor y) PermuteData(Tensor x, Tensor y, int seed = 1)
        {
            var perm = randperm(x.shape[0]);
            return (x[perm], y[perm]);
        }
    }
}