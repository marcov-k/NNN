namespace NNNTester;

/// <summary>
/// Program for testing NNN framework functionality.
/// </summary>
public class NNNTester
{
    public static void Main(string[] args)
    {
        var cppA = new NNNCSharp.Components.Interop.Tensor([2, 3], true);
        var cppB = new NNNCSharp.Components.Interop.Tensor([3, 2], true);
        var csA = new NNNCSharp.Components.Autodiff.Tensor([2, 3], true);
        var csB = new NNNCSharp.Components.Autodiff.Tensor([3, 2], true);

        for (int i = 0; i < cppA.ElementCount; i++)
        {
            cppA[i] = i;
            csA[i] = i;
            cppB[i] = i * 2;
            csB[i] = i * 2;
        }

        var cppC = cppA ^ cppB;
        var csC = csA ^ csB;

        cppC.Backward();
        csC.Backward();

        for (int i = 0; i < cppC.ElementCount; i++)
        {
            Console.WriteLine($"Result at index {i}");
            Console.WriteLine($"C# Tensor: A = {csA[i]}, A grad = {csA.Grad[i]}, B = {csB[i]}, B grad = {csB.Grad[i]}, C = {csC[i]}, C grad = {csC.Grad[i]}");
            Console.WriteLine($"C++ Tensor: A = {cppA[i]}, A grad = {cppA.Grad[i]}, B = {cppB[i]}, B grad = {cppB.Grad[i]}, C = {cppC[i]}, C grad = {cppC.Grad[i]}");
            Console.WriteLine();
        }
    }
}