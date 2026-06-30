using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Interop;

internal static class NativeMethods
{
    const string DllName = "NNN.dll";

    [DllImport(DllName)]
    internal static extern IntPtr tensor_create(int[] dims, int rank, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_create_empty();

    [DllImport(DllName)]
    internal static extern IntPtr tensor_create_scalar(double value, int[] dims, int rank, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_weights(int input_count, int neuron_count);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_biases(int neuron_count);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_init_kernels(int filter_count, int[] kernel_dims, int kernel_rank, int input_channels);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_copy(IntPtr handle);

    [DllImport(DllName)]
    internal static extern void tensor_release(IntPtr handle);

    [DllImport(DllName)]
    internal static extern int tensor_rank(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_dims_ptr(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_strides_ptr(IntPtr handle);

    [DllImport(DllName)]
    internal static extern int tensor_element_count(IntPtr handle);

    [DllImport(DllName)]
    internal static extern int tensor_grad_count(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_data_ptr(IntPtr handle);

    [DllImport(DllName)]
    internal static extern double tensor_get_at(IntPtr handle, int index);

    [DllImport(DllName)]
    internal static extern void tensor_set_at(IntPtr handle, double value, int index);

    [DllImport(DllName)]
    internal static extern double tensor_get_at_spatial(IntPtr handle, int[] indices, int rank);

    [DllImport(DllName)]
    internal static extern void tensor_set_at_spatial(IntPtr handle, double value, int[] indices, int rank);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_grad_ptr(IntPtr handle);

    [DllImport(DllName)]
    internal static extern int tensor_linear_index(IntPtr handle, int[] indices, int rank);

    [DllImport(DllName)]
    internal static extern void tensor_get_full_indices(IntPtr handle, int index, int[] out_indices);

    [DllImport(DllName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool tensor_get_requires_grad(IntPtr handle);

    [DllImport(DllName)]
    internal static extern void tensor_set_requires_grad(IntPtr handle, [MarshalAs(UnmanagedType.I1)] bool requires_grad);

    [DllImport(DllName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool tensor_get_inference();

    [DllImport(DllName)]
    internal static extern void tensor_set_inference([MarshalAs(UnmanagedType.I1)] bool inference);

    [DllImport(DllName)]
    internal static extern void tensor_begin_forward();

    [DllImport(DllName)]
    internal static extern void tensor_clear_graph(IntPtr handle);

    [DllImport(DllName)]
    internal static extern void tensor_backward(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_add(IntPtr handle_a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_add_scalar(IntPtr handle_a, double b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub(IntPtr handle_a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub_scalar(IntPtr handle_a, double b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_sub_scalar_left(double a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_mul(IntPtr handle_a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_mul_scalar(IntPtr handle_a, double b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_div(IntPtr handle_a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_div_scalar(IntPtr handle_a, double b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_div_scalar_left(double a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow(IntPtr handle_a, IntPtr handle_exp);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow_scalar(IntPtr handle_a, double exp);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_pow_scalar_left(double a, IntPtr handle_exp);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_exp(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_log(IntPtr handle_arg, IntPtr handle_log_base);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_log_scalar(IntPtr handle_arg, double log_base);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_log_scalar_left(double arg, IntPtr handle_log_base);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_ln(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_matmul(IntPtr handle_a, IntPtr handle_b);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_convolve(IntPtr handle_input, IntPtr handle_kernels, IntPtr handle_biases);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_mask_actions(IntPtr handle_q_values, int[] actions, int action_count);

    [DllImport(DllName)]
    internal static extern int tensor_arg_max(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_sum(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_mean(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_transpose(IntPtr handle, int[] axes, int axes_length);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_transpose_default(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_broadcast(IntPtr handle, int[] target_dims, int target_dims_length);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_reshape(IntPtr handle, int[] new_dims, int new_dims_length);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_flatten(IntPtr handle, int start_axis);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_wrap_batch(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_clip(IntPtr handle, double min, double max);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_get_dense_dropout_mask(int[] dims, int dims_length, double dropout);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_get_spatial_dropout_mask(int[] dims, int dims_length, double dropout);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_relu(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_leaky_relu(IntPtr handle, double tau);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_sigmoid(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_tanh(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_softmax(IntPtr handle);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_mse(IntPtr handle_t, IntPtr handle_target);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_huber(IntPtr handle_t, IntPtr handle_target, double delta);

    [DllImport(DllName)]
    internal static extern IntPtr tensor_softmax_cross_entropy(IntPtr handle_t, IntPtr handle_target);
}
