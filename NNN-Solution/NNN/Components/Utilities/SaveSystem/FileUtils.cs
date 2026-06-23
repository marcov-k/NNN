using NNN.Components.Autodiff;
using NNN.Components.Models;
using NNN.Components.Models.Layers;
using System.Text;

namespace NNN.Components.Utilities.SaveSystem;

public static class FileUtils
{
    // Encoding functions

    public static void WriteBool(FileStream stream, bool data)
    {
        stream.Write(BitConverter.GetBytes(data));
    }

    public static void WriteDouble(FileStream stream, double data)
    {
        stream.Write(BitConverter.GetBytes(data));
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
        stream.Write(BitConverter.GetBytes(data));
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
        stream.WriteByte((byte)model.Layers.Length);
        foreach (var layer in model.Layers)
        {
            WriteLayer(stream, layer);
        }
    }

    public static void WriteString(FileStream stream, string data)
    {
        var bytes = Encoding.UTF8.GetBytes(data);
        WriteInt32(stream, bytes.Length);
        stream.Write(bytes);
    }

    public static void WriteTensor(FileStream stream, Tensor data)
    {
        WriteInt32Array(stream, data.Dimensions);
        WriteBool(stream, data.RequiresGrad);
        WriteDoubleArray(stream, data.Data);
    }

    // Decoding functions

    public static bool ReadBool(FileStream stream)
    {
        var bytes = new byte[sizeof(bool)];
        stream.ReadExactly(bytes);
        return BitConverter.ToBoolean(bytes);
    }

    public static double ReadDouble(FileStream stream)
    {
        var bytes = new byte[sizeof(double)];
        stream.ReadExactly(bytes);
        return BitConverter.ToDouble(bytes);
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
        var bytes = new byte[sizeof(int)];
        stream.ReadExactly(bytes);
        return BitConverter.ToInt32(bytes);
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
        var bytes = new byte[length];
        stream.ReadExactly(bytes);
        return Encoding.UTF8.GetString(bytes);
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
}
