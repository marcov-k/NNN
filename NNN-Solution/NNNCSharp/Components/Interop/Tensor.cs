using NNNCSharp.Components.Episodes;

namespace NNNCSharp.Components.Interop;

public sealed class Tensor : IDisposable
{
    internal IntPtr Handle => _handle.DangerousGetHandle();
    readonly TensorSafeHandle _handle;

    internal Tensor(IntPtr handle)
    {
        _handle = new TensorSafeHandle(handle);
    }

    public Tensor(int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create(dims, dims.Length, requiresGrad);
        _handle = new TensorSafeHandle(rawHandle);
    }

    public Tensor(double value, int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create_scalar(value, dims, dims.Length, requiresGrad);
        _handle = new TensorSafeHandle(rawHandle);
    }

    public static Tensor Scalar(double value, int[] dims, bool requiresGrad = false)
    {
        return new(value, dims, requiresGrad);
    }

    public static Tensor InitWeights(int inputCount, int neuronCount)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_weights(inputCount, neuronCount);
        return new(rawHandle);
    }

    public static Tensor InitBiases(int neuronCount)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_biases(neuronCount);
        return new(rawHandle);
    }

    public static Tensor InitKernels(int filterCount, int[] kernelDims, int inputChannels)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_kernels(filterCount, kernelDims, kernelDims.Length, inputChannels);
        return new(rawHandle);
    }

    public Tensor Copy()
    {
        IntPtr rawHandle = NativeMethods.tensor_copy(Handle);
        return new(rawHandle);
    }

    public void Dispose()
    {
        _handle.Dispose();
    }

    public int Rank => NativeMethods.tensor_rank(Handle);

    public unsafe Span<int> Dimensions
    {
        get
        {
            IntPtr dimsPtr = NativeMethods.tensor_dims_ptr(Handle);
            return new Span<int>((void*)dimsPtr, Rank);
        }
    }

    public unsafe Span<int> Strides
    {
        get
        {
            IntPtr stridesPtr = NativeMethods.tensor_strides_ptr(Handle);
            return new Span<int>((void*)stridesPtr, Rank);
        }
    }

    public int ElementCount => NativeMethods.tensor_element_count(Handle);

    public int GradCount => NativeMethods.tensor_grad_count(Handle);

    public unsafe Span<double> Data
    {
        get
        {
            IntPtr dataPtr = NativeMethods.tensor_data_ptr(Handle);
            return new Span<double>((void*)dataPtr, ElementCount);
        }
    }

    public unsafe Span<double> Grad
    {
        get
        {
            IntPtr gradPtr = NativeMethods.tensor_grad_ptr(Handle);
            return new Span<double>((void*)gradPtr, GradCount);
        }
    }

    public double this[int index]
    {
        get => NativeMethods.tensor_get_at(Handle, index);
        set => NativeMethods.tensor_set_at(Handle, value, index);
    }

    public double this[params int[] indices]
    {
        get => NativeMethods.tensor_get_at_spatial(Handle, indices, indices.Length);
        set => NativeMethods.tensor_set_at_spatial(Handle, value, indices, indices.Length);
    }

    public bool RequiresGrad
    {
        get => NativeMethods.tensor_get_requires_grad(Handle);
        set => NativeMethods.tensor_set_requires_grad(Handle, value);
    }

    public static bool Inference
    {
        get => NativeMethods.tensor_get_inference();
        set => NativeMethods.tensor_set_inference(value);
    }

    public static void BeginForward() => NativeMethods.tensor_begin_forward();

    public void ClearGraph() => NativeMethods.tensor_clear_graph(Handle);

    public void Backward() => NativeMethods.tensor_backward(Handle);

    public static Tensor operator +(Tensor a, Tensor b)
    {
        return new(NativeMethods.tensor_add(a.Handle, b.Handle));
    }

    public static Tensor operator +(Tensor a, double b)
    {
        return new(NativeMethods.tensor_add_scalar(a.Handle, b));
    }

    public static Tensor operator +(double a, Tensor b) => b + a;

    public static Tensor operator -(Tensor a, Tensor b)
    {
        return new(NativeMethods.tensor_sub(a.Handle, b.Handle));
    }

    public static Tensor operator -(Tensor a, double b)
    {
        return new(NativeMethods.tensor_sub_scalar(a.Handle, b));
    }

    public static Tensor operator -(double a, Tensor b)
    {
        return new(NativeMethods.tensor_sub_scalar_left(a, b.Handle));
    }

    public static Tensor operator *(Tensor a, Tensor b)
    {
        return new(NativeMethods.tensor_mul(a.Handle, b.Handle));
    }

    public static Tensor operator *(Tensor a, double b)
    {
        return new(NativeMethods.tensor_mul_scalar(a.Handle, b));
    }

    public static Tensor operator *(double a, Tensor b) => b * a;

    public static Tensor operator /(Tensor a, Tensor b)
    {
        return new(NativeMethods.tensor_div(a.Handle, b.Handle));
    }

    public static Tensor operator /(Tensor a, double b)
    {
        return new(NativeMethods.tensor_div_scalar(a.Handle, b));
    }

    public static Tensor operator /(double a, Tensor b)
    {
        return new(NativeMethods.tensor_div_scalar_left(a, b.Handle));
    }

    public static Tensor Pow(Tensor a, Tensor exp)
    {
        return new(NativeMethods.tensor_pow(a.Handle, exp.Handle));
    }

    public static Tensor Pow(Tensor a, double exp)
    {
        return new(NativeMethods.tensor_pow_scalar(a.Handle, exp));
    }

    public static Tensor Pow(double a, Tensor exp)
    {
        return new(NativeMethods.tensor_pow_scalar_left(a, exp.Handle));
    }

    public static Tensor Exp(Tensor t)
    {
        return new(NativeMethods.tensor_exp(t.Handle));
    }

    public static Tensor Log(Tensor arg, Tensor logBase)
    {
        return new(NativeMethods.tensor_log(arg.Handle, logBase.Handle));
    }

    public static Tensor Log(Tensor arg, double logBase)
    {
        return new(NativeMethods.tensor_log_scalar(arg.Handle, logBase));
    }

    public static Tensor Log(double arg, Tensor logBase)
    {
        return new(NativeMethods.tensor_log_scalar_left(arg, logBase.Handle));
    }

    public static Tensor Ln(Tensor t)
    {
        return new(NativeMethods.tensor_ln(t.Handle));
    }

    public static Tensor operator ^(Tensor a, Tensor b)
    {
        return new(NativeMethods.tensor_matmul(a.Handle, b.Handle));
    }

    public static Tensor Convolve(Tensor input, Tensor kernels, Tensor biases)
    {
        return new(NativeMethods.tensor_convolve(input.Handle, kernels.Handle, biases.Handle));
    }

    public static Tensor MaskActions(Tensor qValues, int[] actions)
    {
        return new(NativeMethods.tensor_mask_actions(qValues.Handle, actions, actions.Length));
    }

    public static Tensor MaskActions(Tensor qValues, List<Experience> batch)
    {
        int actionCount = batch.Count;
        var actions = new int[actionCount];
        for (int i = 0; i < actionCount; i++)
        {
            actions[i] = batch[i].Action;
        }
        return MaskActions(qValues, actions);
    }

    public static int ArgMax(Tensor t)
    {
        return NativeMethods.tensor_arg_max(t.Handle);
    }

    public static Tensor Sum(Tensor t)
    {
        return new(NativeMethods.tensor_sum(t.Handle));
    }

    public static Tensor Mean(Tensor t)
    {
        return new(NativeMethods.tensor_mean(t.Handle));
    }

    public static Tensor Transpose(Tensor t, int[] axes)
    {
        return new(NativeMethods.tensor_transpose(t.Handle, axes, axes.Length));
    }

    public static Tensor Transpose(Tensor t)
    {
        return new(NativeMethods.tensor_transpose_default(t.Handle));
    }

    public static Tensor Broadcast(Tensor t, int[] targetDims)
    {
        return new(NativeMethods.tensor_broadcast(t.Handle, targetDims, targetDims.Length));
    }

    public static Tensor Reshape(Tensor t, int[] newDims)
    {
        return new(NativeMethods.tensor_reshape(t.Handle, newDims, newDims.Length));
    }

    public static Tensor Flatten(Tensor t, int startAxis = 0)
    {
        return new(NativeMethods.tensor_flatten(t.Handle, startAxis));
    }

    public static Tensor WrapBatch(Tensor t)
    {
        return new(NativeMethods.tensor_wrap_batch(t.Handle));
    }

    public static Tensor Clip(Tensor t, double min, double max)
    {
        return new(NativeMethods.tensor_clip(t.Handle, min, max));
    }

    public static Tensor GetDenseDropoutMask(Tensor t, double dropout)
    {
        return new(NativeMethods.tensor_get_dense_dropout_mask(t.Dimensions.ToArray(), t.Rank, dropout));
    }

    public static Tensor GetSpatialDropoutMask(Tensor t, double dropout)
    {
        return new(NativeMethods.tensor_get_spatial_dropout_mask(t.Dimensions.ToArray(), t.Rank, dropout));
    }

    public static Tensor ReLU(Tensor t)
    {
        return new(NativeMethods.tensor_relu(t.Handle));
    }

    public static Tensor LeakyReLU(Tensor t, double tau)
    {
        return new(NativeMethods.tensor_leaky_relu(t.Handle, tau));
    }

    public static Tensor Sigmoid(Tensor t)
    {
        return new(NativeMethods.tensor_sigmoid(t.Handle));
    }

    public static Tensor Tanh(Tensor t)
    {
        return new(NativeMethods.tensor_tanh(t.Handle));
    }

    public static Tensor Softmax(Tensor t)
    {
        return new(NativeMethods.tensor_softmax(t.Handle));
    }

    public static Tensor MSE(Tensor t, Tensor target)
    {
        return new(NativeMethods.tensor_mse(t.Handle, target.Handle));
    }

    public static Tensor Huber(Tensor t, Tensor target, double delta)
    {
        return new(NativeMethods.tensor_huber(t.Handle, target.Handle, delta));
    }

    public static Tensor SoftmaxCrossEntropy(Tensor t, Tensor target)
    {
        return new(NativeMethods.tensor_softmax_cross_entropy(t.Handle, target.Handle));
    }
}
