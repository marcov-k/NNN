using Microsoft.Win32.SafeHandles;

namespace NNNCSharp.Components.Interop;

internal sealed class TensorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
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
