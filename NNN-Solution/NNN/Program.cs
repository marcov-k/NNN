using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

int epochs = 1000000;
double alpha = 0.0001;
Matrix<double> x = Matrix<double>.Build.Dense(10, 1, new double[] { 1, 3, 7, 17, 21, 44, 56, 71, 88, 100 });
int[] n = { 1, 3, 3, 1 };
Matrix<double> w1 = GenerateMatrix(n[1], n[0]);
Matrix<double> w2 = GenerateMatrix(n[2], n[1]);
Matrix<double> w3 = GenerateMatrix(n[3], n[2]);
Matrix<double> b1Def = GenerateMatrix(n[1], 1);
Matrix<double> b2Def = GenerateMatrix(n[2], 1);
Matrix<double> b3Def = GenerateMatrix(n[3], 1);
Matrix<double> y = Matrix<double>.Build.Dense(10, 1);
for (int i = 0; i < x.RowCount; i++)
{
    y[i, 0] = Math.Sqrt(x[i, 0]);
}
y = y.Transpose();
int m = x.RowCount;
Matrix<double> finalTest = Matrix<double>.Build.Dense(10, 1, new double[10] {1, 4, 9, 16, 25, 36, 49, 64, 81, 100});
finalTest = finalTest.Transpose();
Main();

void Main()
{
    CreateNetworkDefaults(1, 2, 3, 1);
    Train();
    Console.WriteLine($"Inputs: ");
    Matrix<double> output = FeedForward(finalTest).yHat;
    WriteMatrix(finalTest);
    Console.WriteLine($"Outputs: ");
    WriteMatrix(output);
}

void CreateNetworkDefaults(int inputs, int hiddenLayers, int hiddenNodes, int outputs)
{
    n = new int[hiddenLayers + 2];
    n[0] = inputs;
    for (int i = 1; i < n.Length - 1; i++)
    {
        n[i] = hiddenNodes;
    }
    n[n.Length - 1] = outputs;
}

List<double> Train()
{
    Matrix<double> a0 = x.Transpose();
    List<double> costs = new List<double>();
    for (int i = 0; i < epochs; i++)
    {
        var (yHat, cache) = FeedForward(a0);
        double error = Cost(yHat, y, m);
        costs.Add(error);
        var (dC_dW3, dC_db3, dC_dA2) = BackpropLayer3(yHat, y, m, cache.a2, w3);
        var (dC_dW2, dC_db2, dC_dA1) = BackpropLayer2(dC_dA2, cache.a1, cache.a2, w2);
        var (dC_dW1, dC_db1) = BackpropLayer1(dC_dA1, cache.a1, cache.a0, w1);
        w3 -= alpha * dC_dW3;
        w2 -= alpha * dC_dW2;
        w1 -= alpha * dC_dW1;
        b3Def -= Broadcast(alpha * dC_db3, b3Def.ColumnCount);
        b2Def -= Broadcast(alpha * dC_db2, b2Def.ColumnCount);
        b1Def -= Broadcast(alpha * dC_db1, b1Def.ColumnCount);
        if (i % 20 == 0) Console.WriteLine($"epoch {i}: cost = {error}");
    }
    return costs;
}

(Matrix<double> dC_dW3, Matrix<double> dC_db3, Matrix<double> dC_dA2) BackpropLayer3(Matrix<double> yHat, Matrix<double> y, int m, Matrix<double> a2, Matrix<double> w3)
{
    Matrix<double> a3 = yHat;
    Matrix<double> dC_dZ3 = (1 / (double)m) * (a3 - y);
    Matrix<double> dZ3_dW3 = a2;
    Matrix<double> dC_dW3 = dC_dZ3 * dZ3_dW3.Transpose();
    Matrix<double> dC_db3 = CalculateBiasC(dC_dZ3);
    Matrix<double> dZ3_dA2 = w3;
    Matrix<double> dC_dA2 = w3.Transpose() * dC_dZ3;
    return (dC_dW3, dC_db3, dC_dA2);
}

(Matrix<double> dC_dW2, Matrix<double> dC_db2, Matrix<double> dC_dA1) BackpropLayer2(Matrix<double> propagator_dC_dA2, Matrix<double> a1, Matrix<double> a2, Matrix<double> w2)
{
    Matrix<double> dA2_dZ2 = a2.PointwiseMultiply(1 - a2);
    Matrix<double> dC_dZ2 = propagator_dC_dA2.PointwiseMultiply(dA2_dZ2);
    Matrix<double> dZ2_dW2 = a1;
    Matrix<double> dC_dW2 = dC_dZ2 * dZ2_dW2.Transpose();
    Matrix<double> dC_db2 = CalculateBiasC(dC_dW2);
    Matrix<double> dZ2_dA1 = w2;
    Matrix<double> dC_dA1 = dZ2_dA1.Transpose() * dC_dZ2;
    return (dC_dW2, dC_db2, dC_dA1);
}

(Matrix<double> dC_dW1, Matrix<double> dC_db1) BackpropLayer1(Matrix<double> propagator_dC_dA1, Matrix<double> a1, Matrix<double> a0, Matrix<double> w1)
{
    Matrix<double> dA1_dZ1 = a1.PointwiseMultiply(1 - a1);
    Matrix<double> dC_dZ1 = propagator_dC_dA1.PointwiseMultiply(dA1_dZ1);
    Matrix<double> dZ1_dW1 = a0;
    Matrix<double> dC_dW1 = dC_dZ1 * dZ1_dW1.Transpose();
    Matrix<double> dC_db1 = CalculateBiasC(dC_dW1);
    return (dC_dW1, dC_db1);
}

(Matrix<double> yHat, Cache cache) FeedForward(Matrix<double> a0)
{
    Cache cache = new Cache();
    cache.a0 = a0;
    Matrix<double> z1 = w1 * a0;
    z1 += Broadcast(b1Def, z1.ColumnCount);
    Matrix<double> a1 = Sigmoid(z1);
    cache.a1 = a1;
    Matrix<double> z2 = w2 * a1;
    z2 += Broadcast(b2Def, z2.ColumnCount);
    Matrix<double> a2 = Sigmoid(z2);
    cache.a2 = a2;
    Matrix<double> z3 = w3 * a2;
    z3 += Broadcast(b3Def, z3.ColumnCount);
    Matrix<double> a3 = z3;
    Matrix<double> yHat = a3;
    return (yHat, cache);
}

static Matrix<double> CalculateBiasC(Matrix<double> dC_dZ)
{
    Matrix<double> result = Matrix<double>.Build.Dense(dC_dZ.RowCount, 1);
    Vector<double> sums = dC_dZ.RowSums();
    for (int i = 0; i < sums.Count; i++)
    {
        result[i, 0] = sums[i];
    }
    return result;
}

static double Cost(Matrix<double> yHat, Matrix<double> y, int m)
{
    Matrix<double> losses = Matrix<double>.Build.Dense(yHat.RowCount, yHat.ColumnCount);
    double cost = 0;
    for (int i = 0; i < yHat.RowCount; i++)
    {
        for (int j = 0; j < yHat.ColumnCount; j++)
        {
            losses[i, j] = Math.Pow(y[i, j] - yHat[i, j], 2);
        }
    }
    Vector<double> summedLosses = (1 / (double)m) * losses.RowSums();
    cost = summedLosses.Sum();
    return cost;
}

static Matrix<double> GenerateMatrix(int rows, int columns)
{
    Normal nd = new Normal();
    Matrix<double> matrix = Matrix<double>.Build.Dense(rows, columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            double value = nd.Sample();
            matrix[i, j] = value;
        }
    }
    return matrix;
}

static Matrix<double> Sigmoid(Matrix<double> matrix)
{
    Matrix<double> negMatrix = matrix * -1;
    Matrix<double> expNegMatrix = negMatrix.PointwiseExp();
    Matrix<double> denom = expNegMatrix + 1;
    Matrix<double> result = 1 / denom;
    return result;
}

static Matrix<double> Broadcast(Matrix<double> input, int columns)
{
    Matrix<double> result = Matrix<double>.Build.Dense(input.RowCount, columns);
    for (int i = 0; i < input.RowCount; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            result[i, j] = input[i, 0];
        }
    }
    return result;
}

void WriteMatrix<T>(Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
{
    for (int i = 0; i < matrix.RowCount; i++)
    {
        string line = "";
        for (int j = 0; j < matrix.ColumnCount; j++)
        {
            line += $"{matrix[i, j]} ";
        }
        Console.WriteLine(line);
    }
}

public class Cache
{
    public Matrix<double> a0 { get; set; }
    public Matrix<double> a1 { get; set; }
    public Matrix<double> a2 { get; set; }
}