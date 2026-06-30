using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Costs;

/// <summary>
/// Base class for cost functions used for calculating model losses.
/// </summary>
public abstract class Cost
{
    /// <summary>
    /// Calculates the mean loss of model predictions.
    /// </summary>
    /// <param name="predictions">Predicted values from the model.</param>
    /// <param name="target">Target outputs.</param>
    /// <returns>Tensor storing the mean loss of the predictions.</returns>
    public abstract Tensor CalculateCost(Tensor predictions, Tensor target);

    /// <summary>
    /// Calculates the loss of each model prediction.
    /// </summary>
    /// <param name="predictions">Predicted values from the model.</param>
    /// <param name="target">Target outputs.</param>
    /// <returns>Tensor storing the loss of each prediction.</returns>
    public abstract Tensor CalculatePerSampleCost(Tensor predictions, Tensor target);

    /// <summary>
    /// Calculates the loss of each model prediction from a PER sampled batch.
    /// </summary>
    /// <param name="predictions">Predicted values from the model.</param>
    /// <param name="target">Target values.</param>
    /// <param name="weights">Sampling weights of the predictions' associated experiences.</param>
    /// <returns>CostResult storing Tensor of per-prediction losses and corresponding PER sampling priorites.</returns>
    public virtual CostResult CalculateCostWithPriority(Tensor predictions, Tensor target, double[]? weights = null)
    {
        var losses = CalculatePerSampleCost(predictions, target); // calculate loss of each prediction

        // Update PER sampling priorities and scale losses based on sampling bias
        var priorities = new double[losses.ElementCount];
        for (int i = 0; i < losses.ElementCount; i++)
        {
            priorities[i] = Math.Abs(losses[i]) + 1e-8;

            if (weights is not null) losses[i] *= weights[i];
        }

        return new(losses, priorities);
    }
}

/// <summary>
/// Record storing PER loss and priority pairs.
/// </summary>
/// <param name="Losses">Tensor storing per-prediction losses.</param>
/// <param name="Priorities">Array storing per-prediction PER sampling priorities.</param>
public record CostResult(Tensor Losses, double[] Priorities);
