namespace NNN.Components.Autodiff;

public partial class Tensor
{
    // Autograd graph functions

    /// <summary>
    /// Creates a new gradient array which matches the size of the linear value array of the tensor.
    /// </summary>
    public void RestoreGrad()
    {
        if (RequiresGrad) Grad = new double[ElementCount];
    }

    /// <summary>
    /// Clears the parent and gradient function data of the tensor.
    /// </summary>
    public void ClearGraph()
    {
        _parents.Clear();
        _backward = delegate { };
    }

    /// <summary>
    /// Increments the current autgrad graph generation.
    /// </summary>
    public static void BeginForward()
    {
        if (!Inference) _forwardGen++;
    }

    /// <summary>
    /// Prepares the autograd graph node for another forward pass.
    /// </summary>
    void PrepareForward()
    {
        if (_lastGen == _forwardGen) return;

        _opIndex = 0;

        foreach (var r in _results)
        {
            r.ClearGraph();
        }

        _lastGen = _forwardGen;
    }

    /// <summary>
    /// Finalizes the autograd graph node's internal data after a forward pass.
    /// </summary>
    void FinalizeForward()
    {
        // Trim any excess result tensor references not used in the last autograd graph
        if (_opIndex < _results.Count)
        {
            _results.RemoveRange(_opIndex, _results.Count - _opIndex);
        }
    }

    /// <summary>
    /// Calculates the gradients of all tensor nodes in the current autograd graph.
    /// </summary>
    public void Backward()
    {
        // Initialize topography and visited node buffers if not yet initialized
        _topo ??= [];
        _visited ??= [];

        // Clear previous topography and visited nodes
        _topo.Clear();
        _visited.Clear();

        BuildTopo(this, _topo, _visited); // build topography of current graph

        // Zero out the gradients of all nodes
        foreach (var t in _topo)
        {
            if (t.RequiresGrad) Array.Clear(t.Grad, 0, t.GradCount);
        }

        Array.Fill(Grad, 1.0); // initialize current node's gradient to 1 (assume Backward() was called on the final node)

        // Iterate backwards from last node in the graph
        for (int i = _topo.Count - 1; i >= 0; i--)
        {
            _topo[i]._backward();
        }

        // Finalize forward pass for each node
        foreach (var t in _topo)
        {
            t.FinalizeForward();
        }
    }

    /// <summary>
    /// Builds the topography of the current autograd graph.
    /// </summary>
    /// <param name="t">Node which is being added to the topography.</param>
    /// <param name="topo">Buffer to store topography in.</param>
    /// <param name="visited">HashSet to track visited nodes in.</param>
    static void BuildTopo(Tensor t, List<Tensor> topo, HashSet<Tensor> visited)
    {
        if (visited.Contains(t)) return; // skip if node appears in graph multiple times
        visited.Add(t);

        // Recursively add all parent nodes to the topography
        foreach (var p in t._parents)
        {
            BuildTopo(p, topo, visited);
        }

        topo.Add(t); // append current node at the end of the topography
    }
}
