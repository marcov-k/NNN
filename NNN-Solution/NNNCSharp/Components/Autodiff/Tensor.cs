using NNNCSharp.Components.Episodes;
using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Autodiff;

/// <summary>
/// C# wrapper class for C++ tensors.
/// </summary>
public sealed class Tensor : IDisposable
{
    // Handle wrapping

    /// <summary>
    /// void* handle of the wrapped C++ tensor.
    /// </summary>
    internal IntPtr Handle => _handle.DangerousGetHandle();
    /// <summary>
    /// SafeHandle instance for the wrapped C++ tensor.
    /// </summary>
    readonly TensorSafeHandle _handle;

    /// <summary>
    /// Creates a new C# tensor wrapper around the given void* handle of a C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    internal Tensor(IntPtr handle)
    {
        _handle = new(handle);
    }

    // Initialization and disposal

    /// <summary>
    /// Creates a new tensor instance with the given dimensions and requiresGrad flag.
    /// </summary>
    /// <param name="dims">Dimensions of the new tensor.</param>
    /// <param name="requiresGrad">requiresGrad flag of the new tensor.</param>
    public Tensor(int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create(dims, dims.Length, requiresGrad);
        _handle = new(rawHandle);
    }

    /// <summary>
    /// Creates a new empty tensor instance.
    /// </summary>
    public Tensor()
    {
        IntPtr rawHandle = NativeMethods.tensor_create_empty();
        _handle = new(rawHandle);
    }

    /// <summary>
    /// Creates a new tensor instance with the given dimensions and requiresGrad flag and fills it with a single scalar value.
    /// </summary>
    /// <param name="value">Scalar value to fill the new tensor with.</param>
    /// <param name="dims">Dimensions of the new tensor.</param>
    /// <param name="requiresGrad">requiresGrad flag of the new tensor.</param>
    public Tensor(double value, int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create_scalar(value, dims, dims.Length, requiresGrad);
        _handle = new(rawHandle);
    }

    /// <summary>
    /// Creates a new tensor instance with the given dimensions and requiresGrad flag and fills it with a single scalar value.
    /// </summary>
    /// <param name="value">Scalar value to fill the new tensor with.</param>
    /// <param name="dims">Dimensions of the new tensor.</param>
    /// <param name="requiresGrad">requiresGrad flag of the new tensor.</param>
    /// <returns>New scalar tensor instance.</returns>
    public static Tensor Scalar(double value, int[] dims, bool requiresGrad = false)
    {
        return new(value, dims, requiresGrad);
    }

    /// <summary>
    /// Initializes a new weights tensor instance for a layer with the given input and neuron counts.
    /// </summary>
    /// <param name="inputCount">Input count of the layer.</param>
    /// <param name="neuronCount">Neuron count of the layer.</param>
    /// <returns>New weights tensor instance.</returns>
    public static Tensor InitWeights(int inputCount, int neuronCount)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_weights(inputCount, neuronCount);
        return new(rawHandle);
    }

    /// <summary>
    /// Initializes a new bias tensor instance for a layer with the given neuron count.
    /// </summary>
    /// <param name="neuronCount">Neuron count of the layer.</param>
    /// <returns>New bias tensor instance.</returns>
    public static Tensor InitBiases(int neuronCount)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_biases(neuronCount);
        return new(rawHandle);
    }

    /// <summary>
    /// Initializes a new kernels tensor instance for a layer with the given filter count, kernel dimensions, and input channel count.
    /// </summary>
    /// <param name="filterCount">Filter count of the layer.</param>
    /// <param name="kernelDims">Kernel dimensions to use.</param>
    /// <param name="inputChannels">Number of input channels of the layer.</param>
    /// <returns>New kernels tensor instance.</returns>
    public static Tensor InitKernels(int filterCount, int[] kernelDims, int inputChannels)
    {
        IntPtr rawHandle = NativeMethods.tensor_init_kernels(filterCount, kernelDims, kernelDims.Length, inputChannels);
        return new(rawHandle);
    }

    /// <summary>
    /// Creates a copy of the given tensor detached from the existing autograd graph.
    /// </summary>
    /// <returns>Copied tensor.</returns>
    public Tensor Copy()
    {
        IntPtr rawHandle = NativeMethods.tensor_copy(Handle);
        return new(rawHandle);
    }

    /// <summary>
    /// Safely disposes the tensor instance.
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
    }

    // Data access

    /// <summary>
    /// Rank of the tensor.
    /// </summary>
    public int Rank => NativeMethods.tensor_rank(Handle);

    /// <summary>
    /// Dimensions of the tensor.
    /// </summary>
    public unsafe Span<int> Dimensions
    {
        get
        {
            IntPtr dimsPtr = NativeMethods.tensor_dims_ptr(Handle);
            return new Span<int>((void*)dimsPtr, Rank);
        }
    }

    /// <summary>
    /// Strides of the tensor.
    /// </summary>
    public unsafe Span<int> Strides
    {
        get
        {
            IntPtr stridesPtr = NativeMethods.tensor_strides_ptr(Handle);
            return new Span<int>((void*)stridesPtr, Rank);
        }
    }

    /// <summary>
    /// Number of elements contained in the tensor.
    /// </summary>
    public int ElementCount => NativeMethods.tensor_element_count(Handle);

    /// <summary>
    /// Number of gradient values contained in the tensor.
    /// </summary>
    public int GradCount => NativeMethods.tensor_grad_count(Handle);

    /// <summary>
    /// Data contained in the tensor.
    /// </summary>
    public unsafe Span<double> Data
    {
        get
        {
            IntPtr dataPtr = NativeMethods.tensor_data_ptr(Handle);
            return new Span<double>((void*)dataPtr, ElementCount);
        }
    }

    /// <summary>
    /// Gradients contained in the tensor.
    /// </summary>
    public unsafe Span<double> Grad
    {
        get
        {
            IntPtr gradPtr = NativeMethods.tensor_grad_ptr(Handle);
            return new Span<double>((void*)gradPtr, GradCount);
        }
    }

    /// <summary>
    /// Accesses the value at the given linear index.
    /// </summary>
    /// <param name="index">Linear index to access.</param>
    /// <returns>Value at the given linear index.</returns>
    public double this[int index]
    {
        get => NativeMethods.tensor_get_at(Handle, index);
        set => NativeMethods.tensor_set_at(Handle, value, index);
    }

    /// <summary>
    /// Access the value at the given indices.
    /// </summary>
    /// <param name="indices">Indices to access.</param>
    /// <returns>Value at the given indices.</returns>
    public double this[params int[] indices]
    {
        get => NativeMethods.tensor_get_at_spatial(Handle, indices, indices.Length);
        set => NativeMethods.tensor_set_at_spatial(Handle, value, indices, indices.Length);
    }

    /// <summary>
    /// Converts the given indices into a linear index in the tensor.
    /// </summary>
    /// <param name="indices">Indices to convert.</param>
    /// <returns>Corresponding linear index in the tensor.</returns>
    public int LinearIndex(params int[] indices)
    {
        return NativeMethods.tensor_linear_index(Handle, indices, indices.Length);
    }

    /// <summary>
    /// Converts the given linear index into indices in the tensor.
    /// </summary>
    /// <param name="index">Linear index to convert.</param>
    /// <returns>Corresponding indices in the tensor.</returns>
    public int[] GetFullIndices(int index)
    {
        var indices = new int[Rank];
        NativeMethods.tensor_get_full_indices(Handle, index, indices);
        return indices;
    }

    /// <summary>
    /// Converts the given linear index into indices in the tensor and writes the result to the provided span.
    /// </summary>
    /// <param name="index">Linear index to convert.</param>
    /// <param name="indices">Span to write the corresponding indices to.</param>
    public void GetFullIndices(int index, Span<int> indices)
    {
        var indices_arr = new int[Rank];
        NativeMethods.tensor_get_full_indices(Handle, index, indices_arr);
        indices_arr.AsSpan().CopyTo(indices);
    }

    /// <summary>
    /// RequiresGrad flag of the tensor.
    /// </summary>
    public bool RequiresGrad
    {
        get => NativeMethods.tensor_get_requires_grad(Handle);
        set => NativeMethods.tensor_set_requires_grad(Handle, value);
    }

    // Debug flags

    /// <summary>
    /// LogDebug flag of the C++ DLL.
    /// </summary>
    public static bool LogDebug
    {
        set => NativeMethods.tensor_set_log_debug(value);
    }

    // Autograd graph

    /// <summary>
    /// Inference flag of the C++ autograd engine.
    /// </summary>
    public static bool Inference
    {
        get => NativeMethods.tensor_get_inference();
        set => NativeMethods.tensor_set_inference(value);
    }

    /// <summary>
    /// Begins a new forward pass in the C++ autograd engine.
    /// </summary>
    public static void BeginForward() => NativeMethods.tensor_begin_forward();

    /// <summary>
    /// Clears the autograd graph connections of the given tensor.
    /// </summary>
    public void ClearGraph() => NativeMethods.tensor_clear_graph(Handle);

    /// <summary>
    /// Triggers the backward gradient calculation for the autograd graph starting at the given tensor.
    /// </summary>
    public void Backward() => NativeMethods.tensor_backward(Handle);

    // Tensor operations

    /// <summary>
    /// Adds the given tensors -> (a (T) + b (T))
    /// </summary>
    /// <param name="a">First tensor to add.</param>
    /// <param name="b">Second tensor to add.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator +(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_add(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Adds the given tensor and scalar -> (a (T) + b (S))
    /// </summary>
    /// <param name="a">Tensor to add.</param>
    /// <param name="b">Scalar to add.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator +(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_add_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    /// <summary>
    /// Adds the given scalar and tensor -> (a (S) + b (T))
    /// </summary>
    /// <param name="a">Scalar to add.</param>
    /// <param name="b">Tensor to add.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator +(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_add_scalar(b.Handle, a); // commutative operation - a + b = b + a
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Subtracts the given tensors -> (a (T) - b (T))
    /// </summary>
    /// <param name="a">Tensor to subtract from.</param>
    /// <param name="b">Tensor to subtract.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator -(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_sub(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Subtracts the given tensor and scalar -> (a (T) - b (S))
    /// </summary>
    /// <param name="a">Tensor to subtract from.</param>
    /// <param name="b">Scalar to subtract.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator -(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_sub_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    /// <summary>
    /// Subtracts the given scalar and tensor -> (a (S) - b (T))
    /// </summary>
    /// <param name="a">Scalar to subtract from.</param>
    /// <param name="b">Tensor to subtract.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator -(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_sub_scalar_left(a, b.Handle);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Multiplies the given tensors -> (a (T) * b (T))
    /// </summary>
    /// <param name="a">First tensor to multiply.</param>
    /// <param name="b">Second tensor to multiply.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator *(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_mul(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Multiplies the given tensor and scalar -> (a (T) * b (S))
    /// </summary>
    /// <param name="a">Tensor to multiply.</param>
    /// <param name="b">Scalar to multiply.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator *(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_mul_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    /// <summary>
    /// Multiplies the given scalar and tensor -> (a (S) * b (T))
    /// </summary>
    /// <param name="a">Scalar to multiply.</param>
    /// <param name="b">Tensor to multiply.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator *(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_mul_scalar(b.Handle, a); // commutative operation - a * b = b * a
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Divides the given tensors -> (a (T) / b (T))
    /// </summary>
    /// <param name="a">Tensor to divide.</param>
    /// <param name="b">Tensor to divide by.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator /(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_div(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Divides the given tensor and scalar -> (a (T) / b (S))
    /// </summary>
    /// <param name="a">Tensor to divide.</param>
    /// <param name="b">Scalar to divide by.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator /(Tensor a, double b)
    {
        IntPtr h = NativeMethods.tensor_div_scalar(a.Handle, b);
        GC.KeepAlive(a);
        return new(h);
    }

    /// <summary>
    /// Divides the given scalar and tensor -> (a (S) / b (T))
    /// </summary>
    /// <param name="a">Scalar to divide.</param>
    /// <param name="b">Tensor to divide by.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator /(double a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_div_scalar_left(a, b.Handle);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Raises the given tensor to the given exponent tensor -> (a (T) ^ exp (T))
    /// </summary>
    /// <param name="a">Tensor to raise.</param>
    /// <param name="exp">Tensor exponent to raise to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Pow(Tensor a, Tensor exp)
    {
        IntPtr h = NativeMethods.tensor_pow(a.Handle, exp.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(exp);
        return new(h);
    }

    /// <summary>
    /// Raises the given tensor to the given exponent scalar -> (a (T) ^ exp (S))
    /// </summary>
    /// <param name="a">Tensor to raise.</param>
    /// <param name="exp">Scalar exponent to raise to.</param>
    /// <returns></returns>
    public static Tensor Pow(Tensor a, double exp)
    {
        IntPtr h = NativeMethods.tensor_pow_scalar(a.Handle, exp);
        GC.KeepAlive(a);
        return new(h);
    }

    /// <summary>
    /// Raises the given scalar to the given exponent tensor -> (a (S) ^ exp (T))
    /// </summary>
    /// <param name="a">Scalar to raise.</param>
    /// <param name="exp">Tensor exponent to raise to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Pow(double a, Tensor exp)
    {
        IntPtr h = NativeMethods.tensor_pow_scalar_left(a, exp.Handle);
        GC.KeepAlive(exp);
        return new(h);
    }

    /// <summary>
    /// Raises e to the power of the given tensor -> (e ^ t)
    /// </summary>
    /// <param name="t">Tensor to raise to the power of.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Exp(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_exp(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Computes the logarithm with the given tensor base of the given tensor -> (log_baseT(arg (T))
    /// </summary>
    /// <param name="arg">Tensor argument.</param>
    /// <param name="logBase">Tensor base to use.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Log(Tensor arg, Tensor logBase)
    {
        IntPtr h = NativeMethods.tensor_log(arg.Handle, logBase.Handle);
        GC.KeepAlive(arg);
        GC.KeepAlive(logBase);
        return new(h);
    }

    /// <summary>
    /// Computes the logarithm with the given scalar base of the given tensor -> (log_base(arg (T))
    /// </summary>
    /// <param name="arg">Tensor argument.</param>
    /// <param name="logBase">Scalar base to use.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Log(Tensor arg, double logBase)
    {
        IntPtr h = NativeMethods.tensor_log_scalar(arg.Handle, logBase);
        GC.KeepAlive(arg);
        return new(h);
    }

    /// <summary>
    /// Computes the logarithm with the given tensor base of the given scalar -> (log_baseT(arg (S))
    /// </summary>
    /// <param name="arg">Scalar argument.</param>
    /// <param name="logBase">Tensor base to use.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Log(double arg, Tensor logBase)
    {
        IntPtr h = NativeMethods.tensor_log_scalar_left(arg, logBase.Handle);
        GC.KeepAlive(logBase);
        return new(h);
    }

    /// <summary>
    /// Computes the natural logarithm of the given tensor -> (ln(t))
    /// </summary>
    /// <param name="t">Tensor argument.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Ln(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_ln(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Computes the matrix multiplication between the given tensors -> (a @ b)
    /// </summary>
    /// <param name="a">First tensor to matrix multiply.</param>
    /// <param name="b">Second tensor to matrix multiply.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor operator ^(Tensor a, Tensor b)
    {
        IntPtr h = NativeMethods.tensor_matmul(a.Handle, b.Handle);
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return new(h);
    }

    /// <summary>
    /// Performs a convolution of the given input and kernels tensors -> convolve(input, kernels)
    /// </summary>
    /// <param name="input">Tensor input to convolve.</param>
    /// <param name="kernels">Tensor kernels to convolve with.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Convolve(Tensor input, Tensor kernels)
    {
        IntPtr h = NativeMethods.tensor_convolve(input.Handle, kernels.Handle);
        GC.KeepAlive(input);
        GC.KeepAlive(kernels);
        return new(h);
    }

    // Tensor utilities

    /// <summary>
    /// Masks the given Q Values tensor based on the given action indices.
    /// </summary>
    /// <param name="qValues">Tensor Q Values to mask.</param>
    /// <param name="actions">Action indices to mask with.</param>
    /// <returns>Masked tensor.</returns>
    public static Tensor MaskActions(Tensor qValues, int[] actions)
    {
        IntPtr h = NativeMethods.tensor_mask_actions(qValues.Handle, actions, actions.Length);
        GC.KeepAlive(qValues);
        GC.KeepAlive(actions);
        return new(h);
    }

    /// <summary>
    /// Masks the given Q Values tensor based on the given batch experiences.
    /// </summary>
    /// <param name="qValues">Tensor Q Values to mask.</param>
    /// <param name="batch">Batch experiences to mask with.</param>
    /// <returns>Masked tensor.</returns>
    public static Tensor MaskActions(Tensor qValues, List<Experience> batch)
    {
        // Extract action indices
        int actionCount = batch.Count;
        var actions = new int[actionCount];
        for (int i = 0; i < actionCount; i++)
        {
            actions[i] = batch[i].Action;
        }
        return MaskActions(qValues, actions);
    }

    /// <summary>
    /// Returns the linear index of the highest value in the given tensor.
    /// </summary>
    /// <param name="t">Tensor to find highest value of.</param>
    /// <returns>Linear index of the highest value in the given tensor.</returns>
    public static int ArgMax(Tensor t)
    {
        int argMax = NativeMethods.tensor_arg_max(t.Handle);
        GC.KeepAlive(t);
        return argMax;
    }

    /// <summary>
    /// Computes the sum of the given tensor.
    /// </summary>
    /// <param name="t">Tensor to compute sum of.</param>
    /// <returns>Result tensor -> dimensions: [1]</returns>
    public static Tensor Sum(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_sum(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Computes the mean of the given tensor.
    /// </summary>
    /// <param name="t">Tensor to compute mean of.</param>
    /// <returns>Result tensor -> dimensions: [1]</returns>
    public static Tensor Mean(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_mean(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Transposes the given tensor based on the given axis indices.
    /// </summary>
    /// <param name="t">Tensor to transpose.</param>
    /// <param name="axes">Dimension permutation order.</param>
    /// <returns>Transposed tensor.</returns>
    public static Tensor Transpose(Tensor t, int[] axes)
    {
        IntPtr h = NativeMethods.tensor_transpose(t.Handle, axes, axes.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(axes);
        return new(h);
    }

    /// <summary>
    /// Transposes the given tensor using the default permutation order.
    /// </summary>
    /// <param name="t">Tensor to transpose.</param>
    /// <returns>Transposed tensor.</returns>
    public static Tensor Transpose(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_transpose_default(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Broadcasts the given tensor into the given dimensions.
    /// </summary>
    /// <param name="t">Tensor to broadcast.</param>
    /// <param name="targetDims">Dimensions to broadcast to.</param>
    /// <returns>Broadcasted tensor.</returns>
    public static Tensor Broadcast(Tensor t, int[] targetDims)
    {
        IntPtr h = NativeMethods.tensor_broadcast(t.Handle, targetDims, targetDims.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(targetDims);
        return new(h);
    }

    /// <summary>
    /// Reshapes the given tensor into the given dimensions.
    /// </summary>
    /// <param name="t">Tensor to reshape.</param>
    /// <param name="newDims">Dimensions to reshape into.</param>
    /// <returns>Reshaped tensor.</returns>
    public static Tensor Reshape(Tensor t, int[] newDims)
    {
        IntPtr h = NativeMethods.tensor_reshape(t.Handle, newDims, newDims.Length);
        GC.KeepAlive(t);
        GC.KeepAlive(newDims);
        return new(h);
    }

    /// <summary>
    /// Flattens the dimensions of the given tensor starting at the given axis.
    /// </summary>
    /// <param name="t">Tensor to flatten.</param>
    /// <param name="startAxis">Axis to flatten from.</param>
    /// <returns>Flattened tensor.</returns>
    public static Tensor Flatten(Tensor t, int startAxis = 0)
    {
        IntPtr h = NativeMethods.tensor_flatten(t.Handle, startAxis);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Wraps the given tensor into a batch of a single input.
    /// </summary>
    /// <param name="t">Tensor to wrap.</param>
    /// <returns>Batch-wrapped tensor.</returns>
    public static Tensor WrapBatch(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_wrap_batch(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Clips the values of the given tensor to within the given min and max bounds.
    /// </summary>
    /// <param name="t">Tensor to clip.</param>
    /// <param name="min">Minimum value to clip below.</param>
    /// <param name="max">Maximum value to clip above.</param>
    /// <returns>Clipped tensor.</returns>
    public static Tensor Clip(Tensor t, double min, double max)
    {
        IntPtr h = NativeMethods.tensor_clip(t.Handle, min, max);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Initializes a new mask applying a dense layer dropout to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply dropout to.</param>
    /// <param name="dropout">Dropout rate to apply.</param>
    /// <returns>Dropout mask tensor.</returns>
    public static Tensor GetDenseDropoutMask(Tensor t, double dropout)
    {
        var dims = t.Dimensions.ToArray();
        IntPtr h = NativeMethods.tensor_get_dense_dropout_mask(dims, dims.Length, dropout);
        GC.KeepAlive(dims);
        return new(h);
    }

    /// <summary>
    /// Initializes a new mask applying spatial dropout to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply spatial dropout to.</param>
    /// <param name="dropout">Dropout rate to use.</param>
    /// <returns>Spatial dropout mask tensor.</returns>
    public static Tensor GetSpatialDropoutMask(Tensor t, double dropout)
    {
        var dims = t.Dimensions.ToArray();
        IntPtr h = NativeMethods.tensor_get_spatial_dropout_mask(dims, dims.Length, dropout);
        GC.KeepAlive(dims);
        return new(h);
    }

    // Activation functions

    /// <summary>
    /// Applies the ReLU activation function to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply ReLU to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor ReLU(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_relu(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Applies the Leaky ReLU activation function to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Leaky ReLU to.</param>
    /// <param name="tau">Tau value to use.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor LeakyReLU(Tensor t, double tau)
    {
        IntPtr h = NativeMethods.tensor_leaky_relu(t.Handle, tau);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Applies the Sigmoid activation function to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Sigmoid to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Sigmoid(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_sigmoid(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Applies the Tanh activation function to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Tanh to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Tanh(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_tanh(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    /// <summary>
    /// Applies the Softmax activation function to the given tensor.
    /// </summary>
    /// <param name="t">Tensor to apply Softmax to.</param>
    /// <returns>Result tensor.</returns>
    public static Tensor Softmax(Tensor t)
    {
        IntPtr h = NativeMethods.tensor_softmax(t.Handle);
        GC.KeepAlive(t);
        return new(h);
    }

    // Cost functions

    /// <summary>
    /// Computes the per-element Mean Squared Error loss of the given tensor using the given target tensor.
    /// </summary>
    /// <param name="t">Prediction tensor to compute loss of.</param>
    /// <param name="target">Target tensor to use.</param>
    /// <returns>Loss tensor.</returns>
    public static Tensor MSE(Tensor t, Tensor target)
    {
        IntPtr h = NativeMethods.tensor_mse(t.Handle, target.Handle);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }

    /// <summary>
    /// Computes the per-element pseudo-Huber loss of the given tensor using the given target tensor.
    /// </summary>
    /// <param name="t">Prediction tensor to compute loss of.</param>
    /// <param name="target">Target tensor to use.</param>
    /// <param name="delta">Delta value to use.</param>
    /// <returns>Loss tensor.</returns>
    public static Tensor Huber(Tensor t, Tensor target, double delta)
    {
        IntPtr h = NativeMethods.tensor_huber(t.Handle, target.Handle, delta);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }

    /// <summary>
    /// Computes the per-element Softmax Cross Entropy loss of the given tensor using the given target tensor.
    /// </summary>
    /// <param name="t">Prediction tensor to compute loss of.</param>
    /// <param name="target">Target tensor to use.</param>
    /// <returns>Loss tensor.</returns>
    public static Tensor SoftmaxCrossEntropy(Tensor t, Tensor target)
    {
        IntPtr h = NativeMethods.tensor_softmax_cross_entropy(t.Handle, target.Handle);
        GC.KeepAlive(t);
        GC.KeepAlive(target);
        return new(h);
    }
}
