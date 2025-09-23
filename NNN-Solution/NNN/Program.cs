using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

int epochs = 1000;
double alpha = 0.001;
Matrix<double> x = Matrix<double>.Build.Dense(14, 1, new double[14] { 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 });
int[] n;
List<Matrix<double>> wVals = new List<Matrix<double>>();
List<Matrix<double>> bDefVals = new List<Matrix<double>>();
Matrix<double> y = Matrix<double>.Build.Dense(14, 1);
for (int i = 0; i < x.RowCount; i++)
{
    y[i, 0] = Math.Sqrt(x[i, 0]);
}
y = y.Transpose();
int m = x.RowCount;
double reLUAlpha = 0.01;
Matrix<double> finalTest = Matrix<double>.Build.Dense(10, 1, new double[10] {1, 4, 9, 16, 25, 36, 49, 64, 81, 100});
finalTest = finalTest.Transpose();
Main();

void Main()
{
    CreateNetworkDefaults(1, 4, 20, 1);
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
    for (int i = 1; i < n.Length; i++)
    {
        wVals.Add(GenerateMatrix(n[i], n[i - 1]));
        bDefVals.Add(GenerateMatrix(n[i], 1));
    }
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
        HandleBackprop(yHat, y, m, cache);
        if (Double.IsNaN(error)) Console.WriteLine($"Cost failed at epoch {i}");
        if (i % 20 == 0) Console.WriteLine($"epoch {i}: cost = {error}");
    }
    return costs;
}

(Matrix<double> yHat, Cache cache) FeedForward(Matrix<double> a0)
{
    Cache cache = new Cache();
    cache.aVals.Add(a0);
    Matrix<double> z = wVals[0] * a0;
    cache.zVals.Add(z);
    z += Broadcast(bDefVals[0], z.ColumnCount);
    Matrix<double> a = LeakyReLU(z);
    cache.aVals.Add(a);
    cache.zVals.Add(z);
    for (int i = 1; i < n.Length - 1; i++)
    {
        z = wVals[i] * a;
        z += Broadcast(bDefVals[i], z.ColumnCount);
        a = LeakyReLU(z);
        cache.aVals.Add(a);
        cache.zVals.Add(z);
    }
    Matrix<double> yHat = cache.aVals[cache.aVals.Count() - 1];
    cache.aVals.RemoveAt(cache.aVals.Count() - 1);
    cache.zVals.RemoveAt(cache.zVals.Count() - 1);
    return (yHat, cache);
}

void HandleBackprop(Matrix<double> yHat, Matrix<double> y, int m, Cache cache)
{
    List<Matrix<double>> dC_dWs = new List<Matrix<double>>();
    List<Matrix<double>> dC_dbs = new List<Matrix<double>>();
    var (dC_dWL, dC_dbL, dC_dAL) = BackpropFinalLayer(yHat, y, m, cache.aVals[cache.aVals.Count() - 1], wVals[wVals.Count() - 1]);
    dC_dWs.Insert(0, dC_dWL);
    dC_dbs.Insert(0, dC_dbL);
    for (int i = cache.aVals.Count() - 1; i > 1; i--)
    {
        (dC_dWL, dC_dbL, dC_dAL) = BackpropHiddenLayer(dC_dAL, cache.aVals[i - 1], cache.zVals[i], wVals[i - 1]);
        dC_dWs.Insert(0, dC_dWL);
        dC_dbs.Insert(0, dC_dbL);
    }
    (dC_dWL, dC_dbL) = BackpropLayer1(dC_dAL, cache.zVals[1], cache.aVals[0], wVals[0]);
    dC_dWs.Insert(0, dC_dWL);
    dC_dbs.Insert(0, dC_dbL);
    for (int i = wVals.Count() - 1; i >= 0; i--)
    {
        wVals[i] -= alpha * dC_dWs[i];
        bDefVals[i] -= Broadcast(alpha * dC_dbs[i], bDefVals[i].ColumnCount);
    }
}

(Matrix<double> dC_dWL, Matrix<double> dC_dbL, Matrix<double> dC_dAL1) BackpropFinalLayer(Matrix<double> yHat, Matrix<double> y, int m, Matrix<double> aL1, Matrix<double> wL)
{
    Matrix<double> aL2 = yHat;
    Matrix<double> dC_dZL = (1 / (double)m) * (aL2 - y);
    Matrix<double> dZL_dWL = aL1;
    Matrix<double> dC_dWL = dC_dZL * dZL_dWL.Transpose();
    Matrix<double> dC_dbL = CalculateBiasC(dC_dZL);
    Matrix<double> dZL_dAL1 = wL;
    Matrix<double> dC_dAL1 = wL.Transpose() * dC_dZL;
    return (dC_dWL, dC_dbL, dC_dAL1);
}

(Matrix<double> dC_dWL, Matrix<double> dC_dL2, Matrix<double> dC_dAL1) BackpropHiddenLayer(Matrix<double> propagator_dC_dAL2, Matrix<double> aL1, Matrix<double> zL, Matrix<double> wL)
{
    Matrix<double> dAL2_dZL = zL.PointwiseMultiply(LeakyReLUDeriv(zL));
    Matrix<double> dC_dZL = propagator_dC_dAL2.PointwiseMultiply(dAL2_dZL);
    Matrix<double> dZL_dWL = aL1;
    Matrix<double> dC_dWL = dC_dZL * dZL_dWL.Transpose();
    Matrix<double> dC_dbL = CalculateBiasC(dC_dZL);
    Matrix<double> dZL_dAL1 = wL;
    Matrix<double> dC_dAL1 = dZL_dAL1.Transpose() * dC_dZL;
    return (dC_dWL, dC_dbL, dC_dAL1);
}

(Matrix<double> dC_dW1, Matrix<double> dC_db1) BackpropLayer1(Matrix<double> propagator_dC_dA1, Matrix<double> z1, Matrix<double> a0, Matrix<double> w1)
{
    Matrix<double> dA1_dZ1 = z1.PointwiseMultiply(LeakyReLUDeriv(z1));
    Matrix<double> dC_dZ1 = propagator_dC_dA1.PointwiseMultiply(dA1_dZ1);
    Matrix<double> dZ1_dW1 = a0;
    Matrix<double> dC_dW1 = dC_dZ1 * dZ1_dW1.Transpose();
    Matrix<double> dC_db1 = CalculateBiasC(dC_dZ1);
    return (dC_dW1, dC_db1);
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

Matrix<double> LeakyReLU(Matrix<double> matrix)
{
    Matrix<double> result = Matrix<double>.Build.Dense(matrix.RowCount, matrix.ColumnCount);
    for (int i = 0; i < matrix.RowCount; i++)
    {
        for (int j = 0; j < matrix.ColumnCount; j++)
        {
            result[i, j] = matrix[i, j];
            if (result[i, j] < 0)
            {
                result[i, j] *= reLUAlpha;
            }
        }
    }
    return result;
}

Matrix<double> LeakyReLUDeriv(Matrix<double> matrix)
{
    Matrix<double> result = Matrix<double>.Build.Dense(matrix.RowCount, matrix.ColumnCount);
    for (int i = 0; i < matrix.RowCount; i++)
    {
        for (int j = 0; j < matrix.ColumnCount; j++)
        {
            if (matrix[i, j] < 0)
            {
                result[i, j] = reLUAlpha;
            }
            else
            {
                result[i, j] = 1;
            }
        }
    }
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
    public List<Matrix<double>> aVals = new List<Matrix<double>>();
    public List<Matrix<double>> zVals = new List<Matrix<double>>();
}