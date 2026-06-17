using NNN.Components.Autodiff;
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
    public void Train(Tensor inputs, Tensor targets, int epochs)
    {
        Stopwatch timer = new();

        int logEvery = Math.Max(100, MathUtils.RoundToInterval(epochs / 500f, 100));
        Tensor predictions;
        Tensor loss;

        // Train for the given number of epochs
        timer.Start();
        for (int e = 0; e < epochs; e++)
        {
            predictions = Model.Forward(inputs);
            loss = Cost.CalculateCost(predictions, targets);
            loss.Backward();

            foreach (var param in Model.Parameters)
            {
                Optimizer.Step(param, epochs);
            }

            // Log diagnostic data to the console
            if (e % logEvery == 0 || e == epochs - 1)
            {
                Console.WriteLine($"Epoch {e} : Loss = {loss[0]} : Time elapsed = {timer.ElapsedMilliseconds}ms : Time per epoch = {((float)timer.ElapsedMilliseconds / logEvery):F2}ms");
                timer.Restart();
            }
        }
    }
}
