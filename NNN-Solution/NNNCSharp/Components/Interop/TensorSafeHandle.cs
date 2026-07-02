using Microsoft.Win32.SafeHandles;

namespace NNNCSharp.Components.Interop;

/// <summary>
/// SafeHandle subclass for C++ tensor instances.
/// </summary>
internal sealed class TensorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    /// <summary>
    /// Creates a new SafeHandle wrapper around the given tensor instance handle.
    /// </summary>
    /// <param name="handle">The void* handle of the tensor instance to wrap.</param>
    public TensorSafeHandle(IntPtr handle) : base(true)
    {
        SetHandle(handle);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.tensor_release(handle);
        return true;
    }
}
