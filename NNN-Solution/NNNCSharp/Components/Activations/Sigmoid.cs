using NNNCSharp.Components.Autodiff;
using System.IO;

namespace NNNCSharp.Components.Activations
{
    /// <summary>
    /// Sigmoid activation function.
    /// </summary>
    public class Sigmoid : Activation
    {
        /// <summary>
        /// Applies the sigmoid activation function to each element in the input tensor.
        /// </summary>
        /// <param name="input">Tensor to apply the sigmoid activation function to.</param>
        /// <returns>Tensor containing the result of applying the sigmoid activation function to the input tensor.</returns>
        public override Tensor Forward(Tensor input) => Tensor.Sigmoid(input);

        /// <summary>
        /// Creates a new sigmoid activation function instance.
        /// </summary>
        /// <returns>New sigmoid activation function instance.</returns>
        public override Activation Copy() => new Sigmoid();

        public override void WriteUniqueData(FileStream stream) { }

        public override void BuildFromData(FileStream stream) { }

        public override string PrintActivation(FileStream stream) => string.Empty;
    }
}
