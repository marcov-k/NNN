using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Interop;

internal static class NativeMethods
{
    const string DLLName = "NNN.dll";

    [DllImport(DLLName)]
    internal static extern IntPtr tensor_create(int[] dims, int rank, bool requires_grad);

    [DllImport(DLLName)]
    internal static extern void tensor_release(IntPtr handle);

    [DllImport(DLLName)]
    internal static extern int tensor_element_count(IntPtr handle);

    [DllImport(DLLName)]
    internal static extern IntPtr tensor_data_ptr(IntPtr handle);
}
