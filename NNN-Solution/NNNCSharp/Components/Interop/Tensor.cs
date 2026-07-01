using NNNCSharp.Components.Episodes;

namespace NNNCSharp.Components.Interop;

public sealed class Tensor : IDisposable
{
    internal IntPtr Handle => _handle.DangerousGetHandle();
    readonly TensorSafeHandle _handle;

    internal Tensor(IntPtr handle)
    {
        _handle = new(handle);
    }

    public Tensor(int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create(dims, dims.Length, requiresGrad);
        _handle = new(rawHandle);
    }

    public Tensor()
    {
        IntPtr rawHandle = NativeMethods.tensor_create_empty();
        _handle = new(rawHandle);
    }

    public Tensor(double value, int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create_scalar(value, dims, dims.Length, requiresGrad);
        _handle = new(rawHandle);
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

    public int LinearIndex(params int[] indices)
    {
        return NativeMethods.tensor_linear_index(Handle, indices, indices.Length);
    }

    public int[] GetFullIndices(int index)
    {
        var indices = new int[Rank];
        NativeMethods.tensor_get_full_indices(Handle, index, indices);
        return indices;
    }

    public void GetFullIndices(int index, Span<int> indices)
    {
        var indices_arr = new int[Rank];
        NativeMethods.tensor_get_full_indices(Handle, index, indices_arr);
        indices_arr.AsSpan().CopyTo(indices);
    }

    public bool RequiresGrad
    {
        get => NativeMethods.tensor_get_requires_grad(Handle);
        set => NativeMethods.tensor_set_requires_grad(Handle, value);
    }

    public static bool LogDebug
    {
        set => NativeMethods.tensor_set_log_debug(value);
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
        IntPtr h = NativeMethods.tensor_add(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator +(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_add_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    public static Tensor operator +(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_add_scalar(b.Handle, a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator -(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_sub(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator -(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_sub_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    public static Tensor operator -(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_sub_scalar_left(a, b.Handle);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator *(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_mul(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator *(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_mul_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    public static Tensor operator *(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_mul_scalar(b.Handle, a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator /(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_div(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor operator /(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_div_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    public static Tensor operator /(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_div_scalar_left(a, b.Handle);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor Pow(Tensor a, Tensor exp)
    {
        IntPtr h = NativeMethods.tensor_pow(a.Handle, exp.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(exp);
        return new(h);
    }

    public static Tensor Pow(Tensor a, double exp)
    {
        IntPtr h = NativeMethods.tensor_pow_scalar(a.Handle, exp);
        GC.KeepAlive(a);
        return new(h);
    }

    public static Tensor Pow(double a, Tensor exp)
    {
        IntPtr h = NativeMethods.tensor_pow_scalar_left(a, exp.Handle);
        GC.KeepAlive(exp);
        return new(h);
    }

    public static Tensor Exp(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_exp(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Log(Tensor arg, Tensor logBase)
    {
        IntPtr h = NativeMethods.tensor_log(arg.Handle, logBase.Handle);
        GC.KeepAlive(arg);
        GC.KeepAlive(logBase);
        return new(h);
    }

    public static Tensor Log(Tensor arg, double logBase)
    {
        IntPtr h = NativeMethods.tensor_log_scalar(arg.Handle, logBase);
        GC.KeepAlive(arg);
        return new(h);
    }

    public static Tensor Log(double arg, Tensor logBase)
    {
        IntPtr h = NativeMethods.tensor_log_scalar_left(arg, logBase.Handle);
        GC.KeepAlive(logBase);
        return new(h);
    }

    public static Tensor Ln(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_ln(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor operator ^(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_matmul(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    public static Tensor Convolve(Tensor input, Tensor kernels, Tensor biases)
    {
        IntPtr h = NativeMethods.tensor_convolve(input.Handle, kernels.Handle, biases.Handle);
        GC.KeepAlive(input);
        GC.KeepAlive(kernels);
        GC.KeepAlive(biases);
        return new(h);
    }

    public static Tensor MaskActions(Tensor qValues, int[] actions)
    {
        IntPtr h = NativeMethods.tensor_mask_actions(qValues.Handle, actions, actions.Length);
        GC.KeepAlive(qValues);
        GC.KeepAlive(actions);
        return new(h);
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
        int argMax = NativeMethods.tensor_arg_max(t.Handle);
        GC.KeepAlive(t);
        return argMax;
    }

    public static Tensor Sum(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_sum(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Mean(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_mean(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Transpose(Tensor t, int[] axes)
    {
        IntPtr h = NativeMethods.tensor_transpose(t.Handle, axes, axes.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(axes);
        return new(h);
    }

    public static Tensor Transpose(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_transpose_default(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Broadcast(Tensor t, int[] targetDims)
    {
        IntPtr h = NativeMethods.tensor_broadcast(t.Handle, targetDims, targetDims.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(targetDims);
        return new(h);
    }

    public static Tensor Reshape(Tensor t, int[] newDims)
    {
        IntPtr h = NativeMethods.tensor_reshape(t.Handle, newDims, newDims.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(newDims);
        return new(h);
    }

    public static Tensor Flatten(Tensor t, int startAxis = 0)
    {
        IntPtr h = NativeMethods.tensor_flatten(t.Handle, startAxis);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor WrapBatch(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_wrap_batch(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Clip(Tensor t, double min, double max)
    {
        IntPtr h = NativeMethods.tensor_clip(t.Handle, min, max);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor GetDenseDropoutMask(Tensor t, double dropout)
    {
        var dims = t.Dimensions.ToArray();
        IntPtr h = NativeMethods.tensor_get_dense_dropout_mask(dims, dims.Length, dropout);
        GC.KeepAlive(dims);
        return new(h);
    }

    public static Tensor GetSpatialDropoutMask(Tensor t, double dropout)
    {
        var dims = t.Dimensions.ToArray();
        IntPtr h = NativeMethods.tensor_get_spatial_dropout_mask(dims, dims.Length, dropout);
        GC.KeepAlive(dims);
        return new(h);
    }

    public static Tensor ReLU(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_relu(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor LeakyReLU(Tensor t, double tau)
    {
        IntPtr h = NativeMethods.tensor_leaky_relu(t.Handle, tau);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Sigmoid(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_sigmoid(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Tanh(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_tanh(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor Softmax(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_softmax(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    public static Tensor MSE(Tensor t, Tensor target)
    {
        IntPtr h = NativeMethods.tensor_mse(t.Handle, target.Handle);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }

    public static Tensor Huber(Tensor t, Tensor target, double delta)
    {
        IntPtr h = NativeMethods.tensor_huber(t.Handle, target.Handle, delta);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }

    public static Tensor SoftmaxCrossEntropy(Tensor t, Tensor target)
    {
        IntPtr h = NativeMethods.tensor_softmax_cross_entropy(t.Handle, target.Handle);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }
}
