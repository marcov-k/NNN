using NNN.Components.Autodiff;

namespace NNN.Components.Activations;

public class Softmax : Activation
{
    public override Tensor Forward(Tensor input) => Tensor.Softmax(input);

    public override Activation Copy() => new Softmax();
}
