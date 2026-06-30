using NNNCSharp.Components.Interop;

namespace NNNCSharp.Components.Utilities.DataLoaders;

/// <summary>
/// Static class for loading the MNIST dataset from files.
/// </summary>
public static class MNISTLoader
{
    /// <summary>
    /// Path to the directory containing MNIST dataset files.
    /// </summary>
    const string DirectoryPath = @"../../../../TrainingData/MNIST";
    /// <summary>
    /// Name of the file containing training image data.
    /// </summary>
    const string TrainImageFileName = @"train-images.idx3-ubyte";
    /// <summary>
    /// Name of the file containing training label data.
    /// </summary>
    const string TrainLabelFileName = @"train-labels.idx1-ubyte";
    /// <summary>
    /// Name of the file containing testing image data.
    /// </summary>
    const string TestImageFileName = @"t10k-images.idx3-ubyte";
    /// <summary>
    /// Name of the file containing testing label data.
    /// </summary>
    const string TestLabelFileName = @"t10k-labels.idx1-ubyte";
    /// <summary>
    /// Magic number identifying MNIST image data files.
    /// </summary>
    const int ImageMagicNumber = 2051;
    /// <summary>
    /// Magic number identifying MNIST label data files.
    /// </summary>
    const int LabelMagicNumber = 2049;

    /// <summary>
    /// Extracts the MNIST training dataset from files.
    /// </summary>
    /// <returns>Arrays of image and one-hot encoded label tensors.</returns>
    public static (Tensor[] Images, Tensor[] Labels) GetTrainingData()
    {
        string imagePath = Path.Combine(DirectoryPath, TrainImageFileName);
        string labelPath = Path.Combine(DirectoryPath, TrainLabelFileName);

        return (ReadImageData(imagePath), ReadLabelData(labelPath));
    }

    /// <summary>
    /// Extracts the MNIST testing dataset from files.
    /// </summary>
    /// <returns>Arrays of image and one-hot encoded label tensors.</returns>
    public static (Tensor[] Images, Tensor[] Labels) GetTestData()
    {
        string imagePath = Path.Combine(DirectoryPath, TestImageFileName);
        string labelPath = Path.Combine(DirectoryPath, TestLabelFileName);

        return (ReadImageData(imagePath), ReadLabelData(labelPath));
    }

    /// <summary>
    /// Reads image data from a MNIST image file.
    /// </summary>
    /// <param name="filePath">Path to the image file.</param>
    /// <returns>Array of image tensors extracted from the file.</returns>
    /// <exception cref="Exception">File was not found or incorrect magic number found.</exception>
    static Tensor[] ReadImageData(string filePath)
    {
        if (!File.Exists(filePath)) throw new Exception("Image data file not found.");

        var bytes = File.ReadAllBytes(filePath);
        int offset = 0; // track current offset in main byte array

        // Extract magic number
        int magicNumber = Extract4ByteInt32(bytes, ref offset);
        if (magicNumber != ImageMagicNumber) throw new Exception($"Invalid magic number: found {magicNumber} instead of {ImageMagicNumber}.");

        // Extract image count
        int imageCount = Extract4ByteInt32(bytes, ref offset);
        var images = new Tensor[imageCount];

        // Extract image dimensions
        int rows = Extract4ByteInt32(bytes, ref offset);
        int cols = Extract4ByteInt32(bytes, ref offset);
        int imageBytes = rows * cols;
        var imageDims = new int[3] { rows, cols, 1 };

        // Extract image pixel data
        for (int i = 0; i < imageCount; i++)
        {
            images[i] = new(imageDims);

            for (int p = 0; p < imageBytes; p++)
            {
                images[i][p] = Extract1ByteInt32(bytes, ref offset) / 255.0; // normalize to range 0-1
            }
        }

        return images;
    }

    /// <summary>
    /// Reads label data from a MNIST label file.
    /// </summary>
    /// <param name="filePath">Path to the label file.</param>
    /// <returns>Array of one-hot encoded label tensors.</returns>
    /// <exception cref="Exception">File was not found or incorrect magic number found.</exception>
    static Tensor[] ReadLabelData(string filePath)
    {
        if (!File.Exists(filePath)) throw new Exception("Label data file not found.");

        var bytes = File.ReadAllBytes(filePath);
        int offset = 0; // track current offset in main byte array

        // Extract magic number
        int magicNumber = Extract4ByteInt32(bytes, ref offset);
        if (magicNumber != LabelMagicNumber) throw new Exception($"Invalid magic number: found {magicNumber} instead of {LabelMagicNumber}.");

        // Extract label count
        int labelCount = Extract4ByteInt32(bytes, ref offset);
        var labels = new Tensor[labelCount];

        // Extract and one-hot encode labels
        for (int l = 0; l < labelCount; l++)
        {
            labels[l] = new([10]);
            labels[l][Extract1ByteInt32(bytes, ref offset)] = 1.0;
        }

        return labels;
    }

    /// <summary>
    /// Converts 4 bytes into a 32 bit integer.
    /// </summary>
    /// <param name="bytes">Byte array to read from.</param>
    /// <param name="offset">Offset of integer in the byte array.</param>
    /// <returns>32 bit integer at the given offset in the byte array.</returns>
    static int Extract4ByteInt32(byte[] bytes, ref int offset)
    {
        var intBytes = new byte[4];
        Array.Copy(bytes, offset, intBytes, 0, 4);
        offset += 4;
        Array.Reverse(intBytes); // reverse byte order from big-endian to little-endian
        return BitConverter.ToInt32(intBytes);
    }

    /// <summary>
    /// Converts 1 byte into a 32 bit integer.
    /// </summary>
    /// <param name="bytes">Byte array to read from.</param>
    /// <param name="offset">Offset of integer in the byte arra.</param>
    /// <returns>32 bit integer at the given offset in the byte array.</returns>
    static int Extract1ByteInt32(byte[] bytes, ref int offset)
    {
        offset++;
        return bytes[offset - 1]; // implicitly convert byte to int32
    }
}
