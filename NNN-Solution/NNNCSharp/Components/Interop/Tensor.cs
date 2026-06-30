namespace NNNCSharp.Components.Interop;

public sealed class Tensor : IDisposable
{
    readonly TensorSafeHandle _handle;

    public Tensor(int[] dims, bool requiresGrad = false)
    {
        IntPtr rawHandle = NativeMethods.tensor_create(dims, dims.Length, requiresGrad);
        _handle = new TensorSafeHandle(rawHandle);
    }

    public int ElementCount => NativeMethods.tensor_element_count(_handle.DangerousGetHandle());

    public unsafe Span<double> Data
    {
        get
        {
            IntPtr dataPtr = NativeMethods.tensor_data_ptr(_handle.DangerousGetHandle());
            return new Span<double>((void*)dataPtr, ElementCount);
        }
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}
