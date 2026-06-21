using NNN.Components.Autodiff;
using NNN.Components.Buffers;
using NNN.Components.Costs;
using NNN.Components.Models;
using NNN.Components.Optimizers;
using NNN.Components.Utilities;
using System.Diagnostics;

namespace NNN.Components.Trainers;

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
    readonly Model Model = model;
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
    /// <param name="inputs">Tensor representing the input values of the dataset.</param>
    /// <param name="targets">Tensor representing the target outputs of the dataset.</param>
    /// <param name="epochs">Number of epochs to train for.</param>
    public void Train(BatchBuffer batchBuffer, int batchSize, int epochs, bool batchAllInputs = true,
        Func<Model, int, bool>? testFunc = null, int testEvery = 100, int testLength = 1000)
    {
        Stopwatch totalTimer = new();
        Stopwatch epochTimer = new();
        TimeSpan avgElapsed = new(0);

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
                    Console.WriteLine($"Parameter before test: value = {param.Data[0]}, {param.Data[1]}, {param.Data[2]}, grad = {param.Grad[0]}, {param.Grad[1]}, {param.Grad[2]}");
                }
                Console.WriteLine();

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
                    Console.WriteLine($"Model success percentage: {successPercent:F2}%");
                }
                foreach (var param in Model.Parameters)
                {
                    Console.WriteLine($"Parameter after test: value = {param.Data[0]}, grad = {param.Grad[0]}");
                }
            }
        }

        totalTimer.Stop();
        Console.WriteLine($"Total training time: {MathUtils.RoundToMS(totalTimer.Elapsed)}");
    }
}
