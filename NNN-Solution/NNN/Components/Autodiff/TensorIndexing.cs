namespace NNN.Components.Autodiff;

public partial class Tensor
{
    // Indexing functions

    /// <summary>
    /// Calculates the strides over the linear array represented by increments in the indices of each dimension.
    /// </summary>
    /// <param name="dims">Array containing the length of each dimension.</param>
    /// <returns>Array containing the strides over the linear array represented by increments in the indices of each dimension.</returns>
    static int[] ComputeStrides(int[] dims)
    {
        int n = dims.Length;
        var strides = new int[n];

        strides[n - 1] = 1;

        for (int i = n - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        return strides;
    }

    /// <summary>
    /// Gets the value at the given coordinate indices.
    /// </summary>
    /// <param name="indices">Coordinate indices of the value to access.</param>
    /// <returns>Value at the given coordinate indices.</returns>
    public double this[params int[] indices]
    {
        get => Data[LinearIndex(indices)];
        set => Data[LinearIndex(indices)] = value;
    }

    /// <summary>
    /// Gets the value at the given linear index.
    /// </summary>
    /// <param name="index">Linear index of the value to access.</param>
    /// <returns>Value at the given linear index.</returns>
    public double this[int index]
    {
        get => Data[index];
        set => Data[index] = value;
    }

    // Convert linear index to multidimensional indices
    /// <summary>
    /// Converts a linear index to the corresponding coordinate indices.
    /// </summary>
    /// <param name="index">Linear index to convert.</param>
    /// <returns>Corresponding coordinate indices of the linear index.</returns>
    public int[] GetFullIndices(int index)
    {
        var indices = new int[Rank];
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = index % Dimensions[i];
            index /= Dimensions[i];
        }
        return indices;
    }

    /// <summary>
    /// Converts a linear index to the corresponding indices.
    /// </summary>
    /// <param name="index">Linear index to convert.</param>
    /// <param name="indices">Span to fill with the corresponding coordinate indices.</param>
    void GetFullIndices(int index, Span<int> indices)
    {
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = index % Dimensions[i];
            index /= Dimensions[i];
        }
    }

    /// <summary>
    /// Converts coordinate indices to the corresponding linear index based on strides.
    /// </summary>
    /// <param name="indices">Coordinate indices to convert.</param>
    /// <returns>Corresponding linear index.</returns>
    public int LinearIndex(params int[] indices)
    {
        int offset = 0;

        for (int i = 0; i < indices.Length; i++)
        {
            offset += indices[i] * Strides[i];
        }

        return offset;
    }

    /// <summary>
    /// Converts a span of coordinate indices to the corresponding linear index based on strides.
    /// </summary>
    /// <param name="indices">Span of coordinate indices to convert.</param>
    /// <returns>Corresponding linear index.</returns>
    int LinearIndex(Span<int> indices)
    {
        int offset = 0;

        for (int i = 0; i < indices.Length; i++)
        {
            offset += indices[i] * Strides[i];
        }

        return offset;
    }
}
