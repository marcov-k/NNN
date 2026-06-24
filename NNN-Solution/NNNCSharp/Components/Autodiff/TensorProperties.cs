using System.Numerics;

namespace NNNCSharp.Components.Autodiff;

/// <summary>
/// Class representing n-dimensional data and autograd graph nodes.
/// </summary>
public partial class Tensor
{
    // Linear value storage
    /// <summary>
    /// Linear array storing all of the tensor's values.
    /// </summary>
    public double[] Data { get; private set; } = [];
    /// <summary>
    /// Linear array storing the gradient of each corresponding value.
    /// </summary>
    public double[] Grad { get; private set; } = [];

    // Shape properties
    /// <summary>
    /// Number of dimensions of the tensor.
    /// </summary>
    public int Rank => Dimensions.Length;
    /// <summary>
    /// Number of individual values in the tensor.
    /// </summary>
    public int ElementCount => Data.Length;
    /// <summary>
    /// Number of individual gradients in the tensor.
    /// </summary>
    public int GradCount => Grad.Length;
    /// <summary>
    /// Array containing the length of each of the tensor's dimensions.
    /// </summary>
    public int[] Dimensions { get; private set; } = [];

    // Index mapping
    /// <summary>
    /// Array containing the strides over the linear array represented by increments in the coordinate indices along each dimension.
    /// </summary>
    public int[] Strides { get; private set; } = [];

    // AutoGrad graph
    /// <summary>
    /// Whether the autograd engine is in graphing or inference mode.
    /// </summary>
    public static bool Inference { get; set; } = false; // controls whether the backward graph is generated during a forward pass
    /// <summary>
    /// Whether it is necessary to calculate the tensor's gradient.
    /// </summary>
    public bool RequiresGrad { get; set; }
    /// <summary>
    /// List of tensors involved in the operation which created this instance.
    /// </summary>
    readonly List<Tensor> _parents = [];
    /// <summary>
    /// List of tensors which were created through operations in which this instance was involved.
    /// </summary>
    readonly List<Tensor> _results = [];
    /// <summary>
    /// Index of the next performed operation involving this tensor in the current autograd graph.
    /// </summary>
    int _opIndex = 0;
    /// <summary>
    /// Gradient calculation function for the parents of this instance.
    /// </summary>
    Action _backward = delegate { };
    /// <summary>
    /// Index of the current autograd graph.
    /// </summary>
    static int _forwardGen = 0;
    /// <summary>
    /// Index of the last autograd graph for which this instance was prepared.
    /// </summary>
    int _lastGen = -1;
    /// <summary>
    /// List of tensors representing the topography of the complete autograd graph.
    /// </summary>
    List<Tensor>? _topo = null;
    /// <summary>
    /// HashSet of all unique autograd tensor nodes visited during topography construction.
    /// </summary>
    HashSet<Tensor>? _visited = null;

    // Optimization parameters
    /// <summary>
    /// Size of vectors in the current CPU architecture.
    /// </summary>
    static readonly int VectorSize = Vector<double>.Count;
    /// <summary>
    /// Minimum tensor element threshold for parallelizing matrix multiplication operator.
    /// </summary>
    const long ParallelThreshold = 500_000;
}
