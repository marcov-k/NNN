using NNN;

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
                        Optim.zero_grad();

                        var output = Model.Forward(xBatch);

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
                            Console.WriteLine($"The loss after {e + 1} epochs was {loss.item<double>()}");
                        }
                    }
                }
                else
                {
                    Model.train();

                    foreach (var item in trainDataloader)
                    {
                        var xBatch = item["data"];
                        var yBatch = item["label"];

                        Optim.zero_grad();

                        var output = Model.Forward(xBatch);

                        var loss = Loss.call(output, yBatch);
                        loss.backward();
                        Optim.step();
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

                                var output = Model.Forward(xBatch);
                                var loss = Loss.call(output, yBatch);
                                losses.Add(loss);
                            }
                            Console.WriteLine($"The loss after {e + 1} epochs was {Math.Round(from_array(losses.ToArray()).mean().item<double>(), 4)}");
                        }
                    }
                }
            }
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
        public virtual Tensor Forward(Tensor x, bool inference = false)
        {
            throw new NotImplementedException();
        }

        public TorchSharpModel(string name) : base(name) { }
    }

    public class TorchSharpLayer : Module
    {
        public Tensor Forward(Tensor x, bool inference = false)
        {
            if (inference)
            {
                eval();
                using var _ = no_grad();

                return ForwardImpl(x);
            }
            else
            {
                train();
                return ForwardImpl(x);
            }
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

        public DenseLayer(string name, int inputSize, int neurons, IModule<Tensor, Tensor>? activation = null, float? dropout = null) : base(name)
        {
            Linear = Linear(inputSize, neurons);
            Activation = activation;
            if (dropout.HasValue)
            {
                Dropout = Dropout(1f - dropout.Value);
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