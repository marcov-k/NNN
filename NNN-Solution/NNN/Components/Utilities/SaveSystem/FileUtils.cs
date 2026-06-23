using NNN.Components.Autodiff;
using NNN.Components.Models;
using NNN.Components.Models.Layers;
using System.Buffers.Binary;
using System.Text;

namespace NNN.Components.Utilities.SaveSystem;

public static class FileUtils
{
    // Encoding functions

    public static void WriteBool(FileStream stream, bool data)
    {
        stream.WriteByte((byte)(data ? 1 : 0));
    }

    public static void WriteDouble(FileStream stream, double data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(double)];
        BinaryPrimitives.WriteDoubleLittleEndian(buffer, data);
        stream.Write(buffer);
    }

    public static void WriteDoubleArray(FileStream stream, double[] data)
    {
        WriteInt32(stream, data.Length);
        foreach (var elem in data)
        {
            WriteDouble(stream, elem);
        }
    }

    public static void WriteInt32(FileStream stream, int data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(buffer, data);
        stream.Write(buffer);
    }

    public static void WriteInt32Array(FileStream stream, int[] data)
    {
        WriteInt32(stream, data.Length);
        foreach (var elem in data)
        {
            WriteInt32(stream, elem);
        }
    }

    public static void WriteLayer(FileStream stream, Layer layer)
    {
        stream.WriteByte(IDManager.GetLayerID(layer));
        stream.WriteByte(IDManager.GetActivationID(layer.Activation));
        layer.Activation.WriteUniqueData(stream);
        WriteDouble(stream, layer.Dropout);
        WriteTensor(stream, layer.Biases);

        layer.WriteUniqueData(stream);
    }

    public static void WriteModel(FileStream stream, Model model)
    {
        WriteInt32(stream, model.Layers.Length);
        foreach (var layer in model.Layers)
        {
            WriteLayer(stream, layer);
        }
    }

    public static void WriteString(FileStream stream, string data)
    {
        Span<byte> buffer = Encoding.UTF8.GetBytes(data);
        WriteInt32(stream, buffer.Length);
        stream.Write(buffer);
    }

    public static void WriteTensor(FileStream stream, Tensor data)
    {
        WriteInt32Array(stream, data.Dimensions);
        WriteBool(stream, data.RequiresGrad);
        WriteDoubleArray(stream, data.Data);
    }

    public static void WriteUInt64(FileStream stream, ulong data)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer, data);
        stream.Write(buffer);
    }

    // Decoding functions

    public static bool ReadBool(FileStream stream)
    {
        return stream.ReadByte() == 1;
    }

    public static double ReadDouble(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(double)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadDoubleLittleEndian(buffer);
    }

    public static double[] ReadDoubleArray(FileStream stream)
    {
        int length = ReadInt32(stream);
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = ReadDouble(stream);
        }
        return data;
    }

    public static int ReadInt32(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadInt32LittleEndian(buffer);
    }

    public static int[] ReadInt32Array(FileStream stream)
    {
        int length = ReadInt32(stream);
        var data = new int[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = ReadInt32(stream);
        }
        return data;
    }

    public static Layer ReadLayer(FileStream stream)
    {
        var layerType = IDManager.GetLayerByID((byte)stream.ReadByte());
        var layer = Activator.CreateInstance(layerType) as Layer;
        layer?.BuildFromData(stream);
        return layer!;
    }

    public static Model ReadModel(FileStream stream)
    {
        int layerCount = stream.ReadByte();
        var layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            layers[i] = ReadLayer(stream);
        }
        return new(layers);
    }

    public static string ReadString(FileStream stream)
    {
        int length = ReadInt32(stream);
        Span<byte> buffer = stackalloc byte[length];
        stream.ReadExactly(buffer);
        return Encoding.UTF8.GetString(buffer);
    }

    public static Tensor ReadTensor(FileStream stream)
    {
        var dims = ReadInt32Array(stream);
        bool requiresGrad = ReadBool(stream);
        var data = ReadDoubleArray(stream);

        Tensor tensor = new(dims, requiresGrad);
        Array.Copy(data, tensor.Data, data.Length);
        return tensor;
    }

    public static ulong ReadUInt64(FileStream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ulong)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt64LittleEndian(buffer);
    }

    // File viewing functions

    public static string PrintLayer(FileStream stream)
    {
        var layerType = IDManager.GetLayerByID((byte)stream.ReadByte());
        var layer = Activator.CreateInstance(layerType) as Layer;
        return $"Layer Type: {layerType.Name}\n{layer!.PrintLayer(stream)}";
    }

    public static string PrintTensor(FileStream stream)
    {
        var dims = ReadInt32Array(stream);
        stream.Position++; // skip RequiresGrad
        int dataLength = ReadInt32(stream);
        stream.Position += dataLength * sizeof(double); // skip tensor data

        string data = "Dimensions:[";
        for (int i = 0; i < dims.Length; i++)
        {
            data += dims[i];
            if (i != dims.Length - 1) data += ", ";
        }
        data += $"], # of parameter values: {dataLength}";

        return data;
    }
}
