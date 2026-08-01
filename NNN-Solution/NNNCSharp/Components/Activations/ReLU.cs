using NNNCSharp.Components.Autodiff;

namespace NNNCSharp.Components.Activations
{
    /// <summary>
    /// Rectified Linear Unit activation function.
    /// </summary>
    public class ReLU : Activation
    {
        /// <summary>
        /// Applies the ReLU activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the ReLU activation function to.</param>
        /// <returns>Tensor containing the result of applying the ReLU activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.ReLU(input);

        /// <summary>
        /// Creates a new ReLU activation function instance.
        /// </summary>
        /// <returns>New ReLU activation function instance.</returns>
        public override Activation Copy() => new ReLU();
    }
}
