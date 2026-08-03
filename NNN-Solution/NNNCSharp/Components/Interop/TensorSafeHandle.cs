using Microsoft.Win32.SafeHandles;
using System;

namespace NNNCSharp.Components.Interop
{
    /// <summary>
    /// SafeHandle subclass for C++ tensor instances.
    /// </summary>
    internal sealed class TensorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        /// <summary>
        /// Creates a new SafeHandle wrapper around the given tensor instance handle.
        /// </summary>
        /// <param name="handle">void* handle of the tensor instance to wrap.</param>
        public TensorSafeHandle(IntPtr handle) : base(true)
        {
            SetHandle(handle);
        }

        /// <summary>
        /// Releases the native C++ memory wrapped by the tensor handle.
        /// </summary>
        /// <returns>Whether the handle was successfully released.</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.tensor_release(handle);
            return true;
        }
    }
}
