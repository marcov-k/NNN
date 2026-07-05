namespace NNNCSharp.Components.Buffers;

/// <summary>
/// Sum tree class.
/// </summary>
/// <typeparam name="T">Type to store.</typeparam>
/// <param name="capacity">Maximum size of the sum tree.</param>
public class SumTree<T>(int capacity)
{
    /// <summary>
    /// Maximum size of the sum tree.
    /// </summary>
    readonly int Capacity = capacity;
    /// <summary>
    /// Array representing the sum tree.
    /// </summary>
    readonly double[] Tree = new double[2 * capacity - 1];
    /// <summary>
    /// Data associated with each sum tree node.
    /// </summary>
    readonly T[] Data = new T[capacity];
    /// <summary>
    /// Index to write next new element to.
    /// </summary>
    int WritePointer = 0;
    /// <summary>
    /// Number of elements currently stored in the sum tree.
    /// </summary>
    public int Count { get; private set; } = 0;
    /// <summary>
    /// Total priority of all elements stored in the sum tree.
    /// </summary>
    public double TotalPriority => Tree[0];

    /// <summary>
    /// Adds a new element to the sum tree.
    /// </summary>
    /// <param name="item">Element to add.</param>
    /// <param name="priority">Priority of new element.</param>
    public void Add(T item, double priority)
    {
        // Add item to main data array
        if (Data[WritePointer] is IDisposable disposable) disposable.Dispose();
        Data[WritePointer] = item;

        // Update sum tree with new priority
        int treeIndex = WritePointer + Capacity - 1;
        Update(treeIndex, priority);

        // Increment pointer and count
        WritePointer = (WritePointer + 1) % Capacity;
        if (Count < Capacity) Count++;
    }

    /// <summary>
    /// Updates the sum tree with a new priority.
    /// </summary>
    /// <param name="treeIndex">Index to update the priority of.</param>
    /// <param name="priority">New value of the priority.</param>
    public void Update(int treeIndex, double priority)
    {
        // Update priority at index
        double change = priority - Tree[treeIndex];
        Tree[treeIndex] = priority;

        // Update priority sums of parent nodes
        while (treeIndex > 0)
        {
            treeIndex = (treeIndex - 1) / 2;
            Tree[treeIndex] += change;
        }
    }

    /// <summary>
    /// Gets an element based on a PER sampling value.
    /// </summary>
    /// <param name="value">PER sampling value to use.</param>
    /// <returns>Sum tree index, priority, and element selected via PER sampling.</returns>
    public (int treeIndex, double priority, T item) Get(double value)
    {
        int parentIndex = 0;

        // Select element via PER sampling
        while (parentIndex < Capacity - 1)
        {
            int leftChild = 2 * parentIndex + 1;
            int rightChild = leftChild + 1;

            if (value <= Tree[leftChild])
            {
                parentIndex = leftChild;
            }
            else
            {
                value -= Tree[leftChild];
                parentIndex = rightChild;
            }
        }

        int dataIndex = parentIndex - (Capacity - 1);
        return (parentIndex, Tree[parentIndex], Data[dataIndex]);
    }
}
