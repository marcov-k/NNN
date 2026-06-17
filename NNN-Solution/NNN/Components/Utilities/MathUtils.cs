using NNN.Components.Autodiff;

namespace NNN.Components.Utilities;

/// <summary>
/// Static class containing various math utility functions.
/// </summary>
public static class MathUtils
{
    /// <summary>
    /// Current Random instance.
    /// </summary>
    static readonly Random Random = new();

    /// <summary>
    /// Generates a random number from a probability distribution with a mean of 0 and standard deviation of 1.
    /// </summary>
    /// <returns>Random number generated from a probability distribution with a mean of 0 and standard deviation of 1.</returns>
    public static double NextGaussian()
    {
        // Generate 2 random values from uniform distribution
        double u1 = 1.0 - Random.NextDouble();
        double u2 = 1.0 - Random.NextDouble();

        // Apply Box-Muller Transform -> Z_1 = sqrt(-2 * ln(u_1)) * sin(2π * u_2)
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return randStdNormal;
    }

    /// <summary>
    /// Generates a random number from a probability distribution with the given mean and standard deviation.
    /// </summary>
    /// <param name="mean">Mean of the probability distribution to generate from.</param>
    /// <param name="stdDev">Standard deviation of the probability distribution to generate from.</param>
    /// <returns>Random number generated from a probability distribution with the given mean and standard deviation.</returns>
    public static double NextGaussian(double mean, double stdDev)
    {
        double randStdNormal = NextGaussian(); // generate random number from standard distribution

        // Convert to target distribution -> X = (Z * σ) + μ
        double randNormal = randStdNormal * stdDev + mean;
        return randNormal;
    }

    /// <summary>
    /// Rounds a value to the given interval digit position.
    /// </summary>
    /// <param name="value">Value to round.</param>
    /// <param name="interval">Interval to round to.</param>
    /// <returns>Input value rounded to the given interval.</returns>
    public static int RoundToInterval(double value, int interval)
    {
        return (int)Math.Round(value / interval, MidpointRounding.AwayFromZero) * interval;
    }

    /// <summary>
    /// Samples the sigmoid function at the given input value.
    /// </summary>
    /// <param name="value">Input value to sample at.</param>
    /// <returns>Value of the sigmoid function at the given input value.</returns>
    public static double Sigmoid(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }

    /// <summary>
    /// Samples the hyperbolic tangent function at the given input value.
    /// </summary>
    /// <param name="value">Input value to sample at.</param>
    /// <returns>Value of the hyperbolic tangent function at the given input value.</returns>
    public static double Tanh(double value)
    {
        return Math.Tanh(value);
    }

    /// <summary>
    /// Rounds a TimeSpan to the nearest whole millisecond.
    /// </summary>
    /// <param name="input">TimeSpan to round.</param>
    /// <returns>Given TimeSpan rounded to the nearest whole millisecond.</returns>
    public static TimeSpan RoundToMS(TimeSpan input)
    {
        return TimeSpan.FromMilliseconds(Math.Round(input.TotalMilliseconds));
    }

    /// <summary>
    /// Tests the accuracy of the autodiff engine's gradient estimate compared to numerical analysis.
    /// </summary>
    /// <param name="inputs">Inputs to the test function.</param>
    /// <param name="testOp">Test function for autodiff engine.</param>
    /// <param name="loss">Corresponding function for numerical analysis.</param>
    public static void GradientTest(Tensor[] inputs, Func<Tensor[], Tensor> testOp, Func<Tensor[], double> loss)
    {
        var result = testOp(inputs);
        var mean = Tensor.Mean(result);
        mean.Backward();

        // Calculate relative error for every gradient of each input
        for (int input = 0; input < inputs.Length; input++)
        {
            for (int e = 0; e < inputs[input].ElementCount; e++)
            {
                var numerical = NumericalGradient(inputs, input, e, loss);
                double analytical = inputs[input].Grad[e];
                double relError = Math.Abs(numerical - analytical) / (Math.Abs(numerical) + 1e-8);
                Console.WriteLine($"inputs[{input}][{e}]: numerical = {numerical}, analytical = {analytical}, relError = {relError}");
            }
        }
    }

    /// <summary>
    /// Calculates gradient using numerical analysis.
    /// </summary>
    /// <param name="inputs">Inputs to the test function.</param>
    /// <param name="inputIndex">Index of the input to test.</param>
    /// <param name="e">Index of the element in the input to test.</param>
    /// <param name="loss">Test function for numerical analysis.</param>
    /// <returns></returns>
    static double NumericalGradient(Tensor[] inputs, int inputIndex, int e, Func<Tensor[], double> loss)
    {
        // Estimate gradient via finite difference
        double eps = 1e-8;
        inputs[inputIndex][e] += eps;
        double lossPlus = loss(inputs);
        inputs[inputIndex][e] -= 2 * eps;
        double lossMinus = loss(inputs);
        inputs[inputIndex][e] += eps;
        return (lossPlus - lossMinus) / (2 * eps);
    }
}
