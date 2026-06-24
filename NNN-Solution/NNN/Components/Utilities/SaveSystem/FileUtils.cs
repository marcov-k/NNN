using NNN.Components.Autodiff;
using NNN.Components.Models;
using NNN.Components.Models.Layers;
using System.Buffers.Binary;
using System.Text;

namespace NNN.Components.Utilities.SaveSystem;

/// <summary>
/// Static class containing various file utilities for the .nnn file format.
/// </summary>
public static class FileUtils
{
    // Writing functions

    /// <summary>
    /// Writes a boolean to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">Boolean to write.</param>
    public static void WriteBool(FileStream stream, bool data)
    {
        stream.WriteByte((byte)(data ? 1 : 0));
    }

    /// <summary>
    /// Writes a 64-bit double to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">64-bit double to write.</param>
    public static void WriteDouble(FileStream stream, double data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(double)];
        BinaryPrimitives.WriteDoubleLittleEndian(buffer, data);
        stream.Write(buffer);
    }

    /// <summary>
    /// Writes an array of 64-bit doubles to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">Array of 64-bit doubles to write.</param>
    public static void WriteDoubleArray(FileStream stream, double[] data)
    {
        WriteInt32(stream, data.Length); // write array length

        // Write array data
        foreach (var elem in data)
        {
            WriteDouble(stream, elem);
        }
    }

    /// <summary>
    /// Writes a 32-bit integer to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">32-bit integer to write.</param>
    public static void WriteInt32(FileStream stream, int data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(buffer, data);
        stream.Write(buffer);
    }

    /// <summary>
    /// Writes an array of 32-bit integers to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">Array of 32-bit integers to write.</param>
    public static void WriteInt32Array(FileStream stream, int[] data)
    {
        WriteInt32(stream, data.Length); // write array length

        // Write array data
        foreach (var elem in data)
        {
            WriteInt32(stream, elem);
        }
    }

    /// <summary>
    /// Writes a Neural Network Notions model layer to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="layer">Layer to write.</param>
    public static void WriteLayer(FileStream stream, Layer layer)
    {
        stream.WriteByte(IDManager.GetLayerID(layer)); // write layer type ID
        stream.WriteByte(IDManager.GetActivationID(layer.Activation)); // write activation type ID
        layer.Activation.WriteUniqueData(stream); // write any activation type-specific data
        WriteDouble(stream, layer.Dropout); // write layer dropout
        WriteTensor(stream, layer.Biases); // write layer bias tensor

        // Write any data unique to the specific layer type
        layer.WriteUniqueData(stream);
    }

    /// <summary>
    /// Writes a Neural Network Notions model to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="model">Model to write.</param>
    public static void WriteModel(FileStream stream, Model model)
    {
        WriteInt32(stream, model.Layers.Length); // write layer count

        // Write layer data
        foreach (var layer in model.Layers)
        {
            WriteLayer(stream, layer);
        }
    }

    /// <summary>
    /// Writes a string to the file stream using UTF8 encoding.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">String to write.</param>
    public static void WriteString(FileStream stream, string data)
    {
        Span<byte> buffer = Encoding.UTF8.GetBytes(data);
        WriteInt32(stream, buffer.Length); // write string length
        stream.Write(buffer); // write string characters
    }

    /// <summary>
    /// Writes a Neural Network Notions tensor to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">Tensor to write.</param>
    public static void WriteTensor(FileStream stream, Tensor data)
    {
        WriteInt32Array(stream, data.Dimensions);
        WriteBool(stream, data.RequiresGrad);
        WriteDoubleArray(stream, data.Data); // write linear data array
    }

    /// <summary>
    /// Writes an unsigned 64-bit integer to the file stream.
    /// </summary>
    /// <param name="stream">File stream to write to.</param>
    /// <param name="data">Unsigned 64-bit integer to write.</param>
    public static void WriteUInt64(FileStream stream, ulong data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer, data);
        stream.Write(buffer);
    }

    // Reading functions

    /// <summary>
    /// Reads a boolean from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Boolean at the current position in the file stream.</returns>
    public static bool ReadBool(FileStream stream)
    {
        return stream.ReadByte() == 1;
    }

    /// <summary>
    /// Reads a 64-bit double from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>64-bit double at the current position in the file stream.</returns>
    public static double ReadDouble(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(double)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadDoubleLittleEndian(buffer);
    }

    /// <summary>
    /// Reads an array of 64-bit doubles from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Array of 64-bit doubles at the current position in the file stream.</returns>
    public static double[] ReadDoubleArray(FileStream stream)
    {
        int length = ReadInt32(stream); // read array length

        // Read array data
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = ReadDouble(stream);
        }

        return data;
    }

    /// <summary>
    /// Reads a 32-bit integer from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>32-bit integer at the current position in the file stream.</returns>
    public static int ReadInt32(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadInt32LittleEndian(buffer);
    }

    /// <summary>
    /// Reads an array of 32-bit integers from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Array of 32-bit integers at the current position in the file stream.</returns>
    public static int[] ReadInt32Array(FileStream stream)
    {
        int length = ReadInt32(stream); // read array length

        // Read array data
        var data = new int[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = ReadInt32(stream);
        }
        return data;
    }

    /// <summary>
    /// Reads a Neural Network Notions model layer from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Neural Network Notions layer at the current position in the file stream.</returns>
    public static Layer ReadLayer(FileStream stream)
    {
        var layerType = IDManager.GetLayerByID((byte)stream.ReadByte()); // read layer type ID
        var layer = Activator.CreateInstance(layerType) as Layer;
        layer?.BuildFromData(stream);
        return layer!;
    }

    /// <summary>
    /// Reads a Neural Network Notions model from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Neural Network Notions model at the current position in the file stream.</returns>
    public static Model ReadModel(FileStream stream)
    {
        int layerCount = ReadInt32(stream); // read layer count

        // Read layer data
        var layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            layers[i] = ReadLayer(stream);
        }
        return new(layers); // return new model instance with read layer data
    }

    /// <summary>
    /// Reads a UTF8 encoded string from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>String at the current position in the file stream.</returns>
    public static string ReadString(FileStream stream)
    {
        int length = ReadInt32(stream); // read string length

        // Read string characters
        Span<byte> buffer = stackalloc byte[length];
        stream.ReadExactly(buffer);
        return Encoding.UTF8.GetString(buffer); // decode UTF8 string characters
    }

    /// <summary>
    /// Reads a Neural Network Notions tensor from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Neural Network Notions tensor at the current position in the file stream.</returns>
    public static Tensor ReadTensor(FileStream stream)
    {
        // Read core tensor data
        var dims = ReadInt32Array(stream);
        bool requiresGrad = ReadBool(stream);
        var data = ReadDoubleArray(stream);

        // Create new tensor instance using read data
        Tensor tensor = new(dims, requiresGrad);
        Array.Copy(data, tensor.Data, data.Length);
        return tensor;
    }

    /// <summary>
    /// Reads an unsigned 64-bit integer from the current position in the file stream.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>Unsigned 64-bit integer at the current position in the file stream.</returns>
    public static ulong ReadUInt64(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt64LittleEndian(buffer);
    }

    // File viewing functions

    /// <summary>
    /// Reads data for a Neural Network Notions model layer at the current position in the file stream into a string.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>String containing viewable data for the layer at the current position in the file stream.</returns>
    public static string PrintLayer(FileStream stream)
    {
        var layerType = IDManager.GetLayerByID((byte)stream.ReadByte());
        var layer = Activator.CreateInstance(layerType) as Layer;
        return $"Layer Type: {layerType.Name}\n{layer!.PrintLayer(stream)}";
    }

    /// <summary>
    /// Reads data for a Neural Network Notions tensor at the current position in the file stream into a string.
    /// </summary>
    /// <param name="stream">File stream to read from.</param>
    /// <returns>String containing viewable data for the tensor at the current position in the file stream.</returns>
    public static string PrintTensor(FileStream stream)
    {
        // Read tensor data
        var dims = ReadInt32Array(stream);
        stream.Position++; // skip RequiresGrad
        int dataLength = ReadInt32(stream);
        stream.Position += dataLength * sizeof(double); // skip tensor data

        // Write tensor data to a string
        string data = "Dimensions: [";
        for (int i = 0; i < dims.Length; i++)
        {
            data += dims[i];
            if (i != dims.Length - 1) data += ", ";
        }
        data += $"], # of parameter values: {dataLength}";

        return data;
    }
}
