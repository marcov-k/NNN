using NNNCSharp.Components.Interop;

namespace NNNTester;

/// <summary>
/// Program for testing NNN framework functionality.
/// </summary>
public class NNNTester
{
    public static void Main(string[] args)
    {
        var testTensor = new Tensor([2, 3]);
        for (int i = 0; i < testTensor.ElementCount; i++)
        {
            testTensor.Data[i] = i;
        }

        for (int i = 0; i < testTensor.ElementCount; i++)
        {
            Console.WriteLine($"Data value at index {i}: {testTensor.Data[i]}");
        }
    }
}