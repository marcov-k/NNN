namespace NNN.Components.Utilities;

/// <summary>
/// Static class containing various array utility functions.
/// </summary>
public static class ArrayUtils
{
    /// <summary>
    /// Current Random instance.
    /// </summary>
    static readonly Random Random = new();

    /// <summary>
    /// Randomly selects a given number of elements from an array and their corresponding indices.
    /// </summary>
    /// <typeparam name="T">Type of each element.</typeparam>
    /// <param name="array">Array to select from.</param>
    /// <param name="count">Number of elements to select.</param>
    /// <returns>Array of randomly selected elements and their corresponding indices.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Count argument outside valid range.</exception>
    public static (T Element, int OriginalIndex)[] GetRandomElements<T>(T[] array, int count)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        if (count <= 0 || count > array.Length) throw new ArgumentOutOfRangeException(nameof(count), "Count must be between 0 and length of array.");

        var indices = new int[array.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        var result = new (T Element, int OriginalIndex)[count];

        // Partial Fisher-Yates Shuffle
        for (int i = 0; i < count; i++)
        {
            int randomIndex = Random.Next(i, indices.Length);

            (indices[randomIndex], indices[i]) = (indices[i], indices[randomIndex]);

            result[i] = (array[indices[i]], indices[i]);
        }

        return result;
    }
}
