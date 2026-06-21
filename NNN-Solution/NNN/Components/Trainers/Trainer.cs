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
public class Trainer(Model model, Optimizer optimizer, Cost cost)
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
    /// Trains the model for the given number of epochs using the given dataset.
    /// </summary>
    /// <param name="inputs">Tensor representing the input values of the dataset.</param>
    /// <param name="targets">Tensor representing the target outputs of the dataset.</param>
    /// <param name="epochs">Number of epochs to train for.</param>
    public void Train(BatchBuffer batchBuffer, int batchSize, int epochs, Func<Model, bool>? testFunc = null,
        int testEvery = 100, int testLength = 1000)
    {
        Stopwatch totalTimer = new();
        Stopwatch epochTimer = new();
        TimeSpan avgElapsed = new(0);

        Tensor inputs;
        Tensor targets;
        Tensor predictions;
        Tensor loss;

        // Train for the given number of epochs
        totalTimer.Start();
        epochTimer.Start();
        for (int e = 0; e < epochs; e++)
        {
            (inputs, targets) = batchBuffer.GetBatch(batchSize);

            predictions = Model.Forward(inputs);
            loss = Cost.CalculateCost(predictions, targets);
            loss.Backward();

            foreach (var param in Model.Parameters)
            {
                Optimizer.Step(param, epochs);
            }

            var elapsed = epochTimer.Elapsed;
            avgElapsed += (elapsed - avgElapsed) / (e + 1);

            // Log diagnostic data to the console
            if ((e + 1) % testEvery == 0 || (e + 1) == epochs)
            {
                var eta = avgElapsed * (epochs - e - 1);
                Console.WriteLine($"\n\nEpochs completed: {e + 1}/{epochs}");
                Console.WriteLine($"Loss for last epoch: {loss[0]:F3}");
                Console.WriteLine($"Epoch duration: {MathUtils.RoundToMS(elapsed)}");
                Console.WriteLine($"\nAverage time per epoch: {MathUtils.RoundToMS(avgElapsed)}");
                Console.WriteLine($"Estimated time remaining: {MathUtils.RoundToMS(eta)}");
                if (testFunc is not null)
                {
                    Console.WriteLine($"\nEvaluating model performance...");
                    int successes = 0;
                    for (int i = 0; i < testLength; i++)
                    {
                        if (testFunc(Model)) successes++;
                    }
                    double successPercent = ((double)successes / testLength) * 100.0;
                    Console.WriteLine($"Model success percentage: {successPercent:F2}%");
                }
            }

            epochTimer.Restart();
        }

        totalTimer.Stop();
        Console.WriteLine($"Total training time: {MathUtils.RoundToMS(totalTimer.Elapsed)}");
    }
}
