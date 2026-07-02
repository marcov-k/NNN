using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Interop;

/// <summary>
/// Static class for C++ interop function definition imports.
/// </summary>
internal static class NativeMethods
{
    /// <summary>
    /// Name of the C++ DLL file to import from.
    /// </summary>
    const string DllName = "NNN.dll";

    // Initialization and disposal

    /// <summary>
    /// Creates a new C++ tensor instance with the given dimensions and requires_grad flag.
    /// </summary>
    /// <param name="dims">Dimensions of the new C++ tensor.</param>
    /// <param name="rank">Rank of the new C++ tensor.</param>
    /// <param name="requires_grad">requires_grad flag of the new C++ tensor.</param>
    /// <returns>void* handle of the new C++ tensor instance.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_create(int[] dims, int rank, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    /// <summary>
    /// Creates a new empty C++ tensor instance.
    /// </summary>
    /// <returns>void* handle of the new C++ tensor instance.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_create_empty();

    /// <summary>
    /// Creates a new C++ tensor instance with the given dimensions and requires_grad flag and fills it with a single scalar value.
    /// </summary>
    /// <param name="value">Scalar value to fill the new C++ tensor with.</param>
    /// <param name="dims">Dimensions of the new C++ tensor.</param>
    /// <param name="rank">Rank of the new C++ tensor.</param>
    /// <param name="requires_grad">requires_grad flag of the new C++ tensor.</param>
    /// <returns>void* handle of the new C++ tensor instance.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_create_scalar(double value, int[] dims, int rank, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    /// <summary>
    /// Initializes a new C++ weights tensor instance for a layer with the given input and neuron counts.
    /// </summary>
    /// <param name="input_count">Input count of the layer.</param>
    /// <param name="neuron_count">Neuron count of the layer.</param>
    /// <returns>void* handle of the new C++ weights tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_weights(int input_count, int neuron_count);

    /// <summary>
    /// Initializes a new C++ bias tensor instance for a layer with the given neuron count.
    /// </summary>
    /// <param name="neuron_count">Neuron count of the layer.</param>
    /// <returns>void* handle of the new C++ bias tensor instance.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_biases(int neuron_count);

    /// <summary>
    /// Initializes a new C++ kernels tensor instance for a layer with the given filter count, kernel dimensions, and input channel count.
    /// </summary>
    /// <param name="filter_count">Filter count of the layer.</param>
    /// <param name="kernel_dims">Kernel dimensions to use.</param>
    /// <param name="kernel_rank">Length of the kernel dimensions array.</param>
    /// <param name="input_channels">Number of input channels of the layer.</param>
    /// <returns>void* handle of the new C++ kernels tensor instance.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_kernels(int filter_count, int[] kernel_dims, int kernel_rank, int input_channels);

    /// <summary>
    /// Creates a copy of the given C++ tensor detached from the existing autograd graph.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor to copy.</param>
    /// <returns>void* handle of the copied C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_copy(IntPtr handle);

    /// <summary>
    /// Releases the given C++ tensor - deletes its export handle.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor to release.</param>
    [DllImport(DllName)]
    internal static extern void tensor_release(IntPtr handle);

    // Data access

    /// <summary>
    /// Returns the rank of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>Rank of the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern int tensor_rank(IntPtr handle);

    /// <summary>
    /// Returns a pointer to the start of the dimensions array of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>int* to the start of the dimensions array of the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_dims_ptr(IntPtr handle);

    /// <summary>
    /// Returns a pointer to the start of the strides array of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>int* to the start of the strides array of the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_strides_ptr(IntPtr handle);

    /// <summary>
    /// Returns the number of elements contained in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>Number of elements in the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern int tensor_element_count(IntPtr handle);

    /// <summary>
    /// Returns the number of gradient values contained in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>Number of gradient values in the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern int tensor_grad_count(IntPtr handle);

    /// <summary>
    /// Returns a pointer to the start of the data array of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>double* to the start of the data array of the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_data_ptr(IntPtr handle);

    /// <summary>
    /// Returns the value at the given linear index in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="index">Linear index to read from.</param>
    /// <returns>Value at the given linear index.</returns>
    [DllImport(DllName)]
    internal static extern double tensor_get_at(IntPtr handle, int index);

    /// <summary>
    /// Sets the value at the given linear index in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="value">Value to write.</param>
    /// <param name="index">Linear index to write to.</param>
    [DllImport(DllName)]
    internal static extern void tensor_set_at(IntPtr handle, double value, int index);

    /// <summary>
    /// Returns the value at the given indices in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="indices">Indices to read from.</param>
    /// <param name="rank">Length of the indices array.</param>
    /// <returns>Values at the given indices.</returns>
    [DllImport(DllName)]
    internal static extern double tensor_get_at_spatial(IntPtr handle, int[] indices, int rank);

    /// <summary>
    /// Sets the value at the given indices in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="value">Value to write.</param>
    /// <param name="indices">Indices to write to.</param>
    /// <param name="rank">Length of the indices array.</param>
    [DllImport(DllName)]
    internal static extern void tensor_set_at_spatial(IntPtr handle, double value, int[] indices, int rank);

    /// <summary>
    /// Returns a pointer to the start of gradient array of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>double* to the start of the gradient array of the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_grad_ptr(IntPtr handle);

    /// <summary>
    /// Converts the given indices into a linear index in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="indices">Indices to linearize.</param>
    /// <param name="rank">Length of the indices array.</param>
    /// <returns>Corresponding linear index in the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern int tensor_linear_index(IntPtr handle, int[] indices, int rank);

    /// <summary>
    /// Converts the given linear index into indices in the given C++ tensor and writes the result to the provided array.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="index">Linear index to convert.</param>
    /// <param name="out_indices">Array to write the corresponding indices to.</param>
    [DllImport(DllName)]
    internal static extern void tensor_get_full_indices(IntPtr handle, int index, int[] out_indices);

    /// <summary>
    /// Returns the requires_grad flag of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>requires_grad flag of the given C++ tensor.</returns>
    [DllImport(DllName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool tensor_get_requires_grad(IntPtr handle);

    /// <summary>
    /// Sets the requires_grad flag of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="requires_grad">Value to set.</param>
    [DllImport(DllName)]
    internal static extern void tensor_set_requires_grad(IntPtr handle, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    // Debug flags

    /// <summary>
    /// Sets the log_debug flag of the C++ DLL.
    /// </summary>
    /// <param name="log_debug">Value to set.</param>
    [DllImport(DllName)]
    internal static extern void tensor_set_log_debug([MarshalAs(UnmanagedType.I1)] bool log_debug);

    // Autograd graph

    /// <summary>
    /// Returns the inference flag of the C++ autograd engine.
    /// </summary>
    /// <returns>Current inference flag value.</returns>
    [DllImport(DllName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool tensor_get_inference();

    /// <summary>
    /// Sets the inference flag of the C++ autograd engine.
    /// </summary>
    /// <param name="inference">Value to set.</param>
    [DllImport(DllName)]
    internal static extern void tensor_set_inference([MarshalAs(UnmanagedType.I1)] bool inference);

    /// <summary>
    /// Begins a new forward pass in the C++ autograd engine.
    /// </summary>
    [DllImport(DllName)]
    internal static extern void tensor_begin_forward();

    /// <summary>
    /// Clears the autograd graph connections of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    [DllImport(DllName)]
    internal static extern void tensor_clear_graph(IntPtr handle);

    /// <summary>
    /// Triggers the backward gradient calculation for the autograd graph starting at the given C++ tensor.
    /// </summary>
    /// <param name="handle">viod* handle of the C++ tensor.</param>
    [DllImport(DllName)]
    internal static extern void tensor_backward(IntPtr handle);

    // Tensor operations

    /// <summary>
    /// Adds the given C++ tensors -> (a (T) + b (T))
    /// </summary>
    /// <param name="handle_a">void* handle of the first C++ tensor.</param>
    /// <param name="handle_b">void* handle of the second C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_add(IntPtr handle_a, IntPtr handle_b);

    /// <summary>
    /// Adds the given C++ tensor and scalar -> (a (T) + b (S))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="b">Scalar to add.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_add_scalar(IntPtr handle_a, double b);

    /// <summary>
    /// Subtracts the given C++ tensors -> (a (T) - b (T))
    /// </summary>
    /// <param name="handle_a">void* handle of the first C++ tensor.</param>
    /// <param name="handle_b">vooid* handle of the second C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub(IntPtr handle_a, IntPtr handle_b);

    /// <summary>
    /// Subtracts the given C++ tensor and scalar -> (a (T) - b (S))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="b">Scalar to subtract.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub_scalar(IntPtr handle_a, double b);

    /// <summary>
    /// Subtracts the given scalar and C++ tensor -> (a (S) - b (T))
    /// </summary>
    /// <param name="a">Scalar to subtract from.</param>
    /// <param name="handle_b">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub_scalar_left(double a, IntPtr handle_b);

    /// <summary>
    /// Multiplies the given C++ tensors -> (a (T) * b (T))
    /// </summary>
    /// <param name="handle_a">void* handle of the first C++ tensor.</param>
    /// <param name="handle_b">void* handle of the second C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_mul(IntPtr handle_a, IntPtr handle_b);

    /// <summary>
    /// Multiplies the given C++ tensor and scalar -> (a (T) * b (S))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="b">Scalar to multiply by.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_mul_scalar(IntPtr handle_a, double b);

    /// <summary>
    /// Divides the given C++ tensors -> (a (T) / b (T))
    /// </summary>
    /// <param name="handle_a">void* handle of the first C++ tensor.</param>
    /// <param name="handle_b">void* handle of the second C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_div(IntPtr handle_a, IntPtr handle_b);

    /// <summary>
    /// Divides the given C++ tensor and scalar -> (a (T) / b (S))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="b">Scalar to divide by.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_div_scalar(IntPtr handle_a, double b);

    /// <summary>
    /// Divides the given scalar and C++ tensor -> (a (S) / b (T))
    /// </summary>
    /// <param name="a">Scalar to divide.</param>
    /// <param name="handle_b">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_div_scalar_left(double a, IntPtr handle_b);

    /// <summary>
    /// Raises the given C++ tensor to the given exponent C++ tensor -> (a (T) ^ exp (T))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="handle_exp">void* handle of the exponent C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow(IntPtr handle_a, IntPtr handle_exp);

    /// <summary>
    /// Raises the given C++ tensor to the given exponent scalar -> (a (T) ^ exp (S))
    /// </summary>
    /// <param name="handle_a">void* handle of the C++ tensor.</param>
    /// <param name="exp">Exponent scalar to raise to.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow_scalar(IntPtr handle_a, double exp);

    /// <summary>
    /// Raises the given scalar to the given exponent C++ tensor -> (a (S) ^ exp (T))
    /// </summary>
    /// <param name="a">Scalar to raise.</param>
    /// <param name="handle_exp">void* handle of the exponent C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow_scalar_left(double a, IntPtr handle_exp);

    /// <summary>
    /// Raises e to the power of the given C++ tensor -> (e ^ t)
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_exp(IntPtr handle);

    /// <summary>
    /// Computes the logarithm with the given C++ tensor base of the given C++ tensor -> (log_baseT(arg (T))
    /// </summary>
    /// <param name="handle_arg">void* handle of the argument C++ tensor.</param>
    /// <param name="handle_log_base">void* handle of the base C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_log(IntPtr handle_arg, IntPtr handle_log_base);

    /// <summary>
    /// Computes the logarithm with the given scalar base of the given C++ tensor -> (log_base(arg (T))
    /// </summary>
    /// <param name="handle_arg">void* handle of the argument C++ tensor.</param>
    /// <param name="log_base">Scalar to use as base.</param>
    /// <returns>viod* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_log_scalar(IntPtr handle_arg, double log_base);

    /// <summary>
    /// Computes the logarithm with the given C++ tensor base of the given scalar -> (log_baseT(arg (S))
    /// </summary>
    /// <param name="arg">Scalar argument.</param>
    /// <param name="handle_log_base">void* handle of the base C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_log_scalar_left(double arg, IntPtr handle_log_base);

    /// <summary>
    /// Computes the natural logarithm of the given C++ tensor -> (ln(t))
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_ln(IntPtr handle);

    /// <summary>
    /// Computes the matrix multiplication between the given C++ tensors -> (a @ b)
    /// </summary>
    /// <param name="handle_a">void* handle of the first C++ tensor.</param>
    /// <param name="handle_b">void* handle of the second C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_matmul(IntPtr handle_a, IntPtr handle_b);

    /// <summary>
    /// Performs a convolution of the given input, kernels, and bias tensors -> (convolve(input, kernels) + biases)
    /// </summary>
    /// <param name="handle_input">void* handle of the input C++ tensor.</param>
    /// <param name="handle_kernels">void* handle of the kernels C++ tensor.</param>
    /// <param name="handle_biases">void* handle of the bias C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_convolve(IntPtr handle_input, IntPtr handle_kernels, IntPtr handle_biases);

    // Tensor utilities

    /// <summary>
    /// Masks the given Q Values C++ tensor based on the given action indices.
    /// </summary>
    /// <param name="handle_q_values">void* handle of the Q Values C++ tensor.</param>
    /// <param name="actions">Action indices to mask with.</param>
    /// <param name="action_count">Length of the action array.</param>
    /// <returns>void* handle of the masked C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_mask_actions(IntPtr handle_q_values, int[] actions, int action_count);

    /// <summary>
    /// Returns the linear index of the highest value in the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>Linear index of the highest value in the given C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern int tensor_arg_max(IntPtr handle);

    /// <summary>
    /// Computes the sum of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_sum(IntPtr handle);

    /// <summary>
    /// Computes the mean of the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_mean(IntPtr handle);

    /// <summary>
    /// Transposes the given C++ tensor based on the given axis indices.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="axes">Dimension permutation order.</param>
    /// <param name="axes_length">Length of the axes array.</param>
    /// <returns>void* handle of the transposed C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_transpose(IntPtr handle, int[] axes, int axes_length);

    /// <summary>
    /// Transposes the given C++ tensor using the default transpose override.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the transposed C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_transpose_default(IntPtr handle);

    /// <summary>
    /// Broadcasts the given C++ tensor into the given dimensions.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="target_dims">Target dimensions to broadcast to.</param>
    /// <param name="target_dims_length">Length of the target dimensions array.</param>
    /// <returns>void* handle of the broadcasted C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_broadcast(IntPtr handle, int[] target_dims, int target_dims_length);

    /// <summary>
    /// Reshapes the given C++ tensor into the given dimensions.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="new_dims">New dimensions to reshape into.</param>
    /// <param name="new_dims_length">Length of the new dimensions array.</param>
    /// <returns>void* handle of the reshaped C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_reshape(IntPtr handle, int[] new_dims, int new_dims_length);

    /// <summary>
    /// Flattens the dimensions of the given C++ tensor starting at the given axis.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="start_axis">Axis to flatten from.</param>
    /// <returns>void* handle of the flattened C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_flatten(IntPtr handle, int start_axis);

    /// <summary>
    /// Wraps the given C++ tensor into a batch of a single input.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the wrapped C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_wrap_batch(IntPtr handle);

    /// <summary>
    /// Clips the values of the given C++ tensor to within the given min and max bounds.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="min">Minimum value to clip below.</param>
    /// <param name="max">Maximum value to clip above.</param>
    /// <returns>void* handle of the clipped C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_clip(IntPtr handle, double min, double max);

    /// <summary>
    /// Initializes a new mask applying a dense layer dropout to a C++ tensor with the given dimensions.
    /// </summary>
    /// <param name="dims">Dimensions of the C++ tensor to create the mask for.</param>
    /// <param name="dims_length">Length of the dimensions array.</param>
    /// <param name="dropout">Dropout rate parameter of the layer.</param>
    /// <returns>void* handle of the dropout mask C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_get_dense_dropout_mask(int[] dims, int dims_length, double dropout);

    /// <summary>
    /// Initializes a new mask applying spatial dropout to a C++ tensor with the given dimensions.
    /// </summary>
    /// <param name="dims">Dimensions of the C++ tensor to create the mask for.</param>
    /// <param name="dims_length">Length of the dimensions array.</param>
    /// <param name="dropout">Dropout rate parameter of the layer.</param>
    /// <returns>void* handle of the dropout mask C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_get_spatial_dropout_mask(int[] dims, int dims_length, double dropout);

    // Activation functions

    /// <summary>
    /// Applies the ReLU activation function to the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_relu(IntPtr handle);

    /// <summary>
    /// Applies the Leaky ReLU activation function to the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <param name="tau">Tau parameter to use.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_leaky_relu(IntPtr handle, double tau);

    /// <summary>
    /// Applies the Sigmoid activation function to the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_sigmoid(IntPtr handle);

    /// <summary>
    /// Applies the Tanh activation function to the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_tanh(IntPtr handle);

    /// <summary>
    /// Applies the Softmax activation function to the given C++ tensor.
    /// </summary>
    /// <param name="handle">void* handle of the C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_softmax(IntPtr handle);

    // Cost functions

    /// <summary>
    /// Computes the per-element Mean Squared Error loss of the given C++ tensor using the given target C++ tensor.
    /// </summary>
    /// <param name="handle_t">void* handle of the prediction C++ tensor.</param>
    /// <param name="handle_target">void* handle of the target C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_mse(IntPtr handle_t, IntPtr handle_target);

    /// <summary>
    /// Computes the per-element pseudo-Huber loss of the given C++ tensor using the given target C++ tensor.
    /// </summary>
    /// <param name="handle_t">void* handle of the prediction C++ tensor.</param>
    /// <param name="handle_target">void* handle of the target C++ tensor.</param>
    /// <param name="delta">Delta parameter to use.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_huber(IntPtr handle_t, IntPtr handle_target, double delta);

    /// <summary>
    /// Computes the per-element Softmax Cross Entropy loss of the given C++ tensor using the given target C++ tensor.
    /// </summary>
    /// <param name="handle_t">void* handle of the prediction C++ tensor.</param>
    /// <param name="handle_target">void* handle of the target C++ tensor.</param>
    /// <returns>void* handle of the result C++ tensor.</returns>
    [DllImport(DllName)]
    internal static extern IntPtr tensor_softmax_cross_entropy(IntPtr handle_t, IntPtr handle_target);

    // Optimizers

    /// <summary>
    /// Clips the gradients of the given parameter C++ tensors using the given max norm.
    /// </summary>
    /// <param name="handles">Array of void* handles of parameter C++ tensors.</param>
    /// <param name="para_count">Length of the parameter handles array.</param>
    /// <param name="max_norm">Max norm parameter to use.</param>
    [DllImport(DllName)]
    internal static extern void optimizers_clip_gradients(IntPtr[] handles, int para_count, double max_norm);

    /// <summary>
    /// Performs a Stochastic Gradient Descent optimizer step on the given parameter C++ tensor.
    /// </summary>
    /// <param name="handle_para">void* handle of the parameter C++ tensor.</param>
    /// <param name="lr">Learning rate parameter to use.</param>
    [DllImport(DllName)]
    internal static extern void optimizers_sgd(IntPtr handle_para, double lr);

    /// <summary>
    /// Performs an Adam optimizer step with the given parameters on the given parameter C++ tensor.
    /// </summary>
    /// <param name="handle_para">void* handle of the parameter C++ tensor.</param>
    /// <param name="lr">Learning rate parameter to use.</param>
    /// <param name="iter">Current optimizer iteration.</param>
    /// <param name="m">First moments to use.</param>
    /// <param name="v">Second moments to use.</param>
    /// <param name="moments_count">Length of the moments arrays.</param>
    /// <param name="beta1">Beta1 parameter to use.</param>
    /// <param name="one_minus_beta1">Precalculated 1 - beta1 parameter to use.</param>
    /// <param name="beta2">Beta2 parameter to use.</param>
    /// <param name="one_minus_beta2">Precalculated 1 - beta2 parameter to use.</param>
    /// <param name="epsilon">Epsilon value to use.</param>
    /// <param name="weight_decay">Weight decay parameter to use.</param>
    [DllImport(DllName)]
    internal static extern void optimizers_adam(IntPtr handle_para, double lr, int iter, double[] m, double[] v, int moments_count,
        double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay);

    // Models

    /// <summary>
    /// Applies a soft update to the given target model's parameters based on the given agent model's parameters.
    /// </summary>
    /// <param name="handles_agent">Array of void* handles of the agent's parameter C++ tensors.</param>
    /// <param name="handles_target">Array of void* handles of the target's parameter C++ tensors.</param>
    /// <param name="para_count">Length of the handles arrays.</param>
    /// <param name="tau">Tau parameter to use.</param>
    /// <param name="one_minus_tau">Precalculated 1 - tau parameter to use.</param>
    [DllImport(DllName)]
    internal static extern void models_soft_update(IntPtr[] handles_agent, IntPtr[] handles_target, int para_count, double tau,
        double one_minus_tau);
}
