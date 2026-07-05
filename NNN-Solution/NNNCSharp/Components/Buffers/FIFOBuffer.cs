namespace NNNCSharp.Components.Buffers;

/// <summary>
/// Generic First-In First-Out buffer.
/// </summary>
/// <typeparam name="T">Type of the elements contained in the buffer.</typeparam>
/// <param name="maxSize">Maximum size of the buffer.</param>
public class FIFOBuffer<T>(int maxSize)
{
    /// <summary>
    /// Maximum size of the buffer.
    /// </summary>
    public int MaxSize { get; init; } = maxSize;
    /// <summary>
    /// List of elements stored in the buffer.
    /// </summary>
    readonly protected List<T> Buffer = [];
    /// <summary>
    /// Index of the oldest added elements.
    /// </summary>
    protected int FirstIndex = 0;
    /// <summary>
    /// Number of elements stored in the buffer.
    /// </summary>
    public int Count => Buffer.Count;

    /// <summary>
    /// Appends an element to the end of the buffer.
    /// </summary>
    /// <param name="item">Element to append.</param>
    public virtual void Add(T item)
    {
        if (MaxSize <= 0) return;

        if (Count < MaxSize) Buffer.Add(item);
        else
        {
            Buffer[FirstIndex] = item; // replace oldest element
            FirstIndex = (FirstIndex + 1) % MaxSize; // increment index of oldest element
        }
    }

    /// <summary>
    /// Returns the element at the given index in the buffer.
    /// </summary>
    /// <param name="index">Index of the element being accessed.</param>
    /// <returns>Element at the given index.</returns>
    public T this[int index]
    {
        get => Buffer[(FirstIndex + index) % Count]; // adjust the element index based on current index of the oldest element
    }
}
