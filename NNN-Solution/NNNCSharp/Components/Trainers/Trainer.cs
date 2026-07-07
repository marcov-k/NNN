using NNNCSharp.Components.Buffers;
using NNNCSharp.Components.Costs;
using NNNCSharp.Components.Models;
using NNNCSharp.Components.Optimizers;
using NNNCSharp.Components.Utilities;
using System.Diagnostics;
using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Trainers;

/// <summary>
/// Basic non-experiencial supervised neural network trainer.
/// </summary>
/// <param name="model">Model to be trained.</param>
/// <param name="optimizer">Optimizer to use for parameter updates.</param>
/// <param name="cost">Cost function to use for loss calculation.</param>
public class Trainer(Model model, Optimizer optimizer, Cost cost, double maxGradNorm = 1.0)
{
    /// <summary>
    /// Model being trained.
    /// </summary>
    public Model Model { get; private set; } = model;
    /// <summary>
    /// Optimizer being used for parameter updates.
    /// </summary>
    readonly Optimizer Optimizer = optimizer;
    /// <summary>
    /// Cost function being used for loss calculation.
    /// </summary>
    readonly Cost Cost = cost;
    /// <summary>
    /// Maximum magnitude of gradients without normalization.
    /// </summary>
    readonly double MaxGradNorm = maxGradNorm;

    /// <summary>
    /// Trains the model for the given number of epochs using the given dataset.
    /// </summary>
    /// <param name="batchBuffer">Buffer containing the training data.</param>
    /// <param name="batchSize">Number of training inputs to include in each batch.</param>
    /// <param name="epochs">Number of epochs to train for.</param>
    /// <param name="batchAllInputs">Whether to train on all training inputs during each epoch.</param>
    /// <param name="testFunc">Function to use to evaluate model success rate.</param>
    /// <param name="decayLR">Whether to decay the learning rate over time.</param>
    /// <param name="minLRFraction">Minimum allowed fraction of the original learning rate.</param>
    /// <param name="testEvery">How many epochs to run between performance tests.</param>
    /// <param name="testLength">How many iterations to run per performance test.</param>
    public void Train(BatchBuffer batchBuffer, int batchSize, int epochs, bool batchAllInputs = true,
        Func<Model, int, bool>? testFunc = null, bool decayLR = true, double minLRFraction = 0.1, int testEvery = 100,
        int testLength = 1000)
    {
        Stopwatch totalTimer = new();
        Stopwatch epochTimer = new();
        TimeSpan avgElapsed = new(0);

        var bestModel = Model.Copy();
        double bestAccuracy = double.MinValue;

        double baseLR = Optimizer.LR;

        Tensor[] inputs = new Tensor[1];
        Tensor[] targets = new Tensor[1];
        Tensor predictions;
        Tensor loss;

        double totalLoss;

        totalTimer.Start();
        epochTimer.Start();

        if (testFunc is not null)
        {
            Console.WriteLine($"\nEvaluating initial model performance...");
            int successes = 0;
            for (int i = 0; i < testLength; i++)
            {
                if (testFunc(Model, i)) successes++;
            }
            double successPercent = ((double)successes / testLength) * 100.0;
            Console.WriteLine($"Model success percentage: {successPercent:F2}%");
        }

        // Train for the given number of epochs
        int optimizerStep = 0;
        for (int e = 0; e < epochs; e++)
        {
            totalLoss = 0.0;

            if (decayLR)
            {
                double progress = epochs > 1 ? (double)e / (epochs - 1) : 0.0;
                double cosFactor = 0.5 * (1.0 + Math.Cos(Math.PI * progress));
                double decayRange = 1.0 - minLRFraction;
                Optimizer.LR = baseLR * (minLRFraction + decayRange * cosFactor);
            }

            if (batchAllInputs)
            {
                (inputs, targets) = batchBuffer.GetBatches(batchSize);
            }
            else
            {
                (inputs[0], targets[0]) = batchBuffer.GetBatch(batchSize);
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                predictions = Model.Forward(inputs[i]);
                loss = Cost.CalculateCost(predictions, targets[i]);
                totalLoss += loss[0];
                loss.Backward();
                Model.ClipGradients(MaxGradNorm);

                foreach (var param in Model.Parameters)
                {
                    Optimizer.Step(param, optimizerStep);
                    optimizerStep++;
                }
            }

            var elapsed = epochTimer.Elapsed;
            avgElapsed += (elapsed - avgElapsed) / (e + 1);
            epochTimer.Restart();

            // Log diagnostic data to the console
            if ((e + 1) % testEvery == 0 || (e + 1) == epochs)
            {
                var eta = avgElapsed * (epochs - e - 1);
                Console.WriteLine($"\n\nEpochs completed: {e + 1}/{epochs}");
                Console.WriteLine($"Average loss for last epoch: {(totalLoss / inputs.Length):F3}");
                Console.WriteLine($"Epoch duration: {MathUtils.RoundToMS(elapsed)}");
                Console.WriteLine($"\nAverage time per epoch: {MathUtils.RoundToMS(avgElapsed)}");
                Console.WriteLine($"Estimated time remaining: {MathUtils.RoundToMS(eta)}");
                if (testFunc is not null)
                {
                    Console.WriteLine($"\nEvaluating model performance...");
                    int successes = 0;
                    for (int i = 0; i < testLength; i++)
                    {
                        if (testFunc(Model, i)) successes++;
                    }
                    double successPercent = ((double)successes / testLength) * 100.0;
                    if (successPercent > bestAccuracy)
                    {
                        bestModel.Dispose();
                        bestModel = Model.Copy();
                        bestAccuracy = successPercent;
                    }
                    Console.WriteLine($"Model success percentage: {successPercent:F2}%");
                }
            }
        }

        if (decayLR) Optimizer.LR = baseLR;

        Model.Dispose();
        Model = bestModel;
        totalTimer.Stop();
        Console.WriteLine($"Total training time: {MathUtils.RoundToMS(totalTimer.Elapsed)}");
    }
}
