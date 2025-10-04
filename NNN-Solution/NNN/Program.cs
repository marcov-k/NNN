using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;

public class DQNAgent
{
    int stateSize, actionSize;
    double gamma = 0.95;
    double epsilon = 1;
    double epsilonMin = 0.01;
    double epsilonDecay = 0.995;
    int updateFreq = 1000;
    int stepCounter = 0;
    LinkedList<Experience> memory = new LinkedList<Experience>();
    NeuralNetwork model, targetModel;
    Random rand = new Random();

    public void Replay(int batchSize)
    {
        if (memory.Count < batchSize) return;

        var batchList = memory.ToList();
        List<Experience> batch = new List<Experience>(batchSize);
        for (int n = 0; n < batchSize; n++)
        {
            int idx = rand.Next(batchList.Count);
            batch.Add(batchList[idx]);
        }
        var states = Matrix<double>.Build.Dense(batchSize, stateSize);
        var targets = Matrix<double>.Build.Dense(batchSize, actionSize);
        int i = 0;
        foreach (var exp in batch)
        {
            var target = exp.Reward;
            if (!exp.Done)
            {
                var nextInput = Matrix<double>.Build.Dense(1, exp.NextState.Length, exp.NextState);
                var nextQOnline = model.ProcessInput(nextInput);
                int bestNextAction = nextQOnline.Row(0).MaximumIndex();
                var nextQTarget = targetModel.ProcessInput(nextInput);
                target += gamma * nextQTarget[0, bestNextAction];
            }
            var stateVec = Matrix<double>.Build.Dense(1, exp.State.Length, exp.State);
            var targetVec = model.ProcessInput(stateVec);
            targetVec[0, exp.Action] = target;
            states.SetRow(i, exp.State);
            targets.SetRow(i, targetVec.Row(0));
            i++;
        }
        model.SetTrainingData(states, targets);
        model.Train(1, 0.0005);
        if (epsilon > epsilonMin) epsilon *= epsilonDecay;
    }

    public int Act(double[] state)
    {
        if (rand.NextDouble() <= epsilon) return rand.Next(actionSize);
        var input = Matrix<double>.Build.Dense(1, state.Length, state);
        var qVals = model.ProcessInput(input);
        int bestAction = qVals.Row(0).MaximumIndex();
        return bestAction;
    }

    public void UpdateTargetModel()
    {
        var (wOnline, bOnline) = model.GetWeights();
        targetModel.SetWeights(wOnline, bOnline);
    }

    public void Remember(Experience exp)
    {
        memory.AddLast(exp);
        stepCounter++;
        if (memory.Count > 2000) memory.RemoveFirst();
        if (stepCounter % updateFreq == 0) UpdateTargetModel();
    }

    public NeuralNetwork GetTrainedModel()
    {
        return model;
    }

    public DQNAgent(int stateSize, int actionSize, int hiddenLayers, int hiddenNeurons)
    {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        model = new NeuralNetwork(stateSize, hiddenLayers, hiddenNeurons, actionSize);
        targetModel = new NeuralNetwork(stateSize, hiddenLayers, hiddenNeurons, actionSize);
    }
}

public class NeuralNetwork
{
    int epochs = 1000000;
    double alpha = 0.01;
    double clipThreshold = 500;
    Matrix<double>? x;
    int[]? n;
    List<Matrix<double>> wVals = new List<Matrix<double>>();
    List<Matrix<double>> bDefVals = new List<Matrix<double>>();
    Matrix<double>? y;
    int m;
    double reLUAlpha = 0.01;
    double costDelta = 1;
    bool logTraining = true;

    void CreateLayers(int inputs, int hiddenLayers, int hiddenNodes, int outputs)
    {
        n = new int[hiddenLayers + 2];
        n[0] = inputs;
        for (int i = 1; i < n.Length - 1; i++)
        {
            n[i] = hiddenNodes;
        }
        n[n.Length - 1] = outputs;
    }

    void CreateDefaultWeights()
    {
        for (int i = 1; i < n.Length; i++)
        {
            wVals.Add(GenerateMatrix(n[i], n[i - 1], n[i - 1], reLUAlpha));
            bDefVals.Add(GenerateMatrix(n[i], 1, n[i - 1], reLUAlpha));
        }
    }

    List<double> TrainNetwork()
    {
        double initialAlpha = alpha;
        Matrix<double> a0 = x.Transpose();
        List<double> costs = new List<double>();
        for (int i = 0; i < epochs; i++)
        {
            var (yHat, cache) = FeedForward(a0);
            double error = Cost(yHat, y, m);
            costs.Add(error);
            HandleBackprop(yHat, y, m, cache);
            if (logTraining && i % 100 == 0) { Console.WriteLine($"Epoch {i}: Cost = {error}"); }
            if (i == Math.Round(epochs * 0.8)) { alpha *= 0.5; }
        }
        alpha = initialAlpha;
        return costs;
    }

    (Matrix<double> yHat, Cache cache) FeedForward(Matrix<double> a0)
    {
        Cache cache = new Cache();
        cache.aVals.Add(a0);
        Matrix<double> z = wVals[0] * a0;
        z += Broadcast(bDefVals[0], z.ColumnCount);
        Matrix<double> a = LeakyReLU(z);
        cache.aVals.Add(a);
        for (int i = 1; i < n.Length - 1; i++)
        {
            z = wVals[i] * a;
            z += Broadcast(bDefVals[i], z.ColumnCount);
            a = LeakyReLU(z);
            cache.aVals.Add(a);
        }
        Matrix<double> yHat = cache.aVals[cache.aVals.Count() - 1];
        cache.aVals.RemoveAt(cache.aVals.Count() - 1);
        return (yHat, cache);
    }

    void HandleBackprop(Matrix<double> yHat, Matrix<double> y, int m, Cache cache)
    {
        List<Matrix<double>> dC_dWs = new List<Matrix<double>>();
        List<Matrix<double>> dC_dbs = new List<Matrix<double>>();
        var (dC_dWL, dC_dbL, dC_dAL) = BackpropFinalLayer(yHat, y, m, cache.aVals[cache.aVals.Count() - 1], wVals[wVals.Count() - 1], costDelta);
        dC_dWs.Insert(0, dC_dWL);
        dC_dbs.Insert(0, dC_dbL);
        for (int i = cache.aVals.Count() - 1; i > 1; i--)
        {
            (dC_dWL, dC_dbL, dC_dAL) = BackpropHiddenLayer(dC_dAL, cache.aVals[i - 1], cache.aVals[i], wVals[i - 1], m);
            dC_dWs.Insert(0, dC_dWL);
            dC_dbs.Insert(0, dC_dbL);
        }
        (dC_dWL, dC_dbL) = BackpropLayer1(dC_dAL, cache.aVals[1], cache.aVals[0], wVals[0], m);
        dC_dWs.Insert(0, dC_dWL);
        dC_dbs.Insert(0, dC_dbL);
        (dC_dWs, dC_dbs) = ClipGradients(dC_dWs, dC_dbs, clipThreshold);
        for (int i = wVals.Count() - 1; i >= 0; i--)
        {
            wVals[i] -= alpha * dC_dWs[i];
            bDefVals[i] -= Broadcast(alpha * dC_dbs[i], bDefVals[i].ColumnCount);
        }
    }

    public void GenerateNewWeights()
    {
        CreateDefaultWeights();
    }

    public void SetTrainingData(Matrix<double> inputs, Matrix<double> outputs)
    {
        x = inputs.Clone();
        m = x.RowCount;
        y = outputs.Clone();
    }

    public void SetWeights(List<Matrix<double>> weights, List<Matrix<double>> biases)
    {
        wVals = weights.ToList();
        bDefVals = biases.ToList();
    }

    public (List<Matrix<double>> weights, List<Matrix<double>> biases) GetWeights()
    {
        return (wVals.ToList(), bDefVals.ToList());
    }

    public void LogTraining(bool logTraining)
    {
        this.logTraining = logTraining;
    }

    public List<double> Train(int? epochs = null, double? alpha = null, double? clipThreshold = null, double? costDelta = null)
    {
        this.epochs = epochs ?? this.epochs;
        this.alpha = alpha ?? this.alpha;
        this.clipThreshold = clipThreshold ?? this.clipThreshold;
        this.costDelta = costDelta ?? this.costDelta;
        List<double> costs = TrainNetwork();
        return costs.ToList();
    }

    public Matrix<double> ProcessInput(Matrix<double> input)
    {
        Matrix<double> a0 = input.Clone().Transpose();
        Matrix<double> output = FeedForward(a0).yHat.Clone();
        return output;
    }

    public string CreateJsonString()
    {
        NNTrainingData training = new NNTrainingData(ToSerialMatrixList(wVals), ToSerialMatrixList(bDefVals), n, reLUAlpha);
        string jsonString = JsonSerializer.Serialize(training);
        return jsonString;
    }

    public NeuralNetwork(int inputNeurons, int hiddenLayers, int hiddenNeurons, int outputNeurons)
    {
        CreateLayers(inputNeurons, hiddenLayers, hiddenNeurons, outputNeurons);
        CreateDefaultWeights();
    }

    public NeuralNetwork(string jsonString)
    {
        NNTrainingData data = JsonSerializer.Deserialize<NNTrainingData>(jsonString);
        if (data != null)
        {
            wVals = ToMatrixList(data.weights.ToList());
            bDefVals = ToMatrixList(data.biases.ToList());
            n = data.n.ToArray();
            reLUAlpha = data.reLUAlpha;
        }
    }

    static (List<Matrix<double>> dC_dWs, List<Matrix<double>> dC_dbs) ClipGradients(List<Matrix<double>> inputdC_dWs, List<Matrix<double>> inputdC_dbs, double clipThreshold = 500)
    {
        List<Matrix<double>> dC_dWs = new List<Matrix<double>>();
        List<Matrix<double>> dC_dbs = new List<Matrix<double>>();
        foreach (Matrix<double> gradient in inputdC_dWs)
        {
            Matrix<double> newGradient = gradient.Clone();
            double norm = newGradient.FrobeniusNorm();
            if (norm > clipThreshold)
            {
                newGradient *= clipThreshold / norm;
            }
            dC_dWs.Add(newGradient);
        }
        foreach (Matrix<double> gradient in inputdC_dbs)
        {
            Matrix<double> newGradient = gradient.Clone();
            double norm = newGradient.FrobeniusNorm();
            if (norm > clipThreshold)
            {
                newGradient *= clipThreshold / norm;
            }
            dC_dbs.Add(newGradient);
        }
        return (dC_dWs, dC_dbs);
    }

    static (Matrix<double> dC_dWL, Matrix<double> dC_dbL, Matrix<double> dC_dAL1) BackpropFinalLayer(Matrix<double> yHat, Matrix<double> y, int m, Matrix<double> aL1, Matrix<double> wL, double costDelta)
    {
        Matrix<double> aL2 = yHat;
        Matrix<double> dC_dZL = CostDeriv(aL2, y, costDelta, m);
        Matrix<double> dZL_dWL = aL1;
        Matrix<double> dC_dWL = dC_dZL * dZL_dWL.Transpose();
        Matrix<double> dC_dbL = CalculateBiasC(dC_dZL, m);
        Matrix<double> dZL_dAL1 = wL;
        Matrix<double> dC_dAL1 = wL.Transpose() * dC_dZL;
        return (dC_dWL, dC_dbL, dC_dAL1);
    }

    static (Matrix<double> dC_dWL, Matrix<double> dC_dL2, Matrix<double> dC_dAL1) BackpropHiddenLayer(Matrix<double> propagator_dC_dAL2, Matrix<double> aL1, Matrix<double> zL, Matrix<double> wL, double m)
    {
        Matrix<double> dC_dZL = propagator_dC_dAL2.PointwiseMultiply(LeakyReLUDeriv(zL));
        Matrix<double> dZL_dWL = aL1;
        Matrix<double> dC_dWL = dC_dZL * dZL_dWL.Transpose();
        Matrix<double> dC_dbL = CalculateBiasC(dC_dZL, m);
        Matrix<double> dZL_dAL1 = wL;
        Matrix<double> dC_dAL1 = dZL_dAL1.Transpose() * dC_dZL;
        return (dC_dWL, dC_dbL, dC_dAL1);
    }

    static (Matrix<double> dC_dW1, Matrix<double> dC_db1) BackpropLayer1(Matrix<double> propagator_dC_dA1, Matrix<double> z1, Matrix<double> a0, Matrix<double> w1, double m)
    {
        Matrix<double> dA1_dZ1 = LeakyReLUDeriv(z1);
        Matrix<double> dC_dZ1 = propagator_dC_dA1.PointwiseMultiply(dA1_dZ1);
        Matrix<double> dZ1_dW1 = a0;
        Matrix<double> dC_dW1 = dC_dZ1 * dZ1_dW1.Transpose();
        Matrix<double> dC_db1 = CalculateBiasC(dC_dZ1, m);
        return (dC_dW1, dC_db1);
    }

    static Matrix<double> CalculateBiasC(Matrix<double> dC_dZ, double m)
    {
        Matrix<double> result = Matrix<double>.Build.Dense(dC_dZ.RowCount, 1);
        Vector<double> sums = dC_dZ.ColumnSums() / m;
        for (int i = 0; i < sums.Count; i++) result[i, 0] = sums[i];
        return result;
    }

    double Cost(Matrix<double> yHat, Matrix<double> y, int m)
    {
        Matrix<double> losses = Matrix<double>.Build.Dense(yHat.RowCount, yHat.ColumnCount);
        for (int i = 0; i < yHat.RowCount; i++)
        {
            for (int j = 0; j < yHat.ColumnCount; j++)
            {
                if (Math.Abs(yHat[i, j] - y[i, j]) < costDelta)
                {
                    losses[i, j] = 0.5 * Math.Pow(yHat[i, j] - y[i, j], 2);
                }
                else
                {
                    losses[i, j] = costDelta * (Math.Abs(yHat[i, j] - y[i, j]) - 0.5 * costDelta);
                }
            }
        }
        double cost = losses.Enumerate().Sum() / m;
        return cost;
    }

    static Matrix<double> CostDeriv(Matrix<double> aL2, Matrix<double> y, double costDelta, int m)
    {
        Matrix<double> dC_dZL = Matrix<double>.Build.Dense(aL2.RowCount, aL2.ColumnCount);
        for (int i = 0; i < aL2.RowCount; i++)
        {
            for (int j = 0; j < aL2.ColumnCount; j++)
            {
                double diff = aL2[i, j] - y[i, j];
                if (Math.Abs(diff) <= costDelta) dC_dZL[i, j] = diff / m;
                else dC_dZL[i, j] = costDelta * Math.Sign(diff) / m;
            }
        }
        return dC_dZL.Clone();
    }

    static Matrix<double> GenerateMatrix(int rows, int columns, int distribution = 1, double reLUAlpha = 0.01)
    {
        Normal nd = new Normal(0, Math.Sqrt(2 / distribution));
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

    static Matrix<double> LeakyReLU(Matrix<double> matrix, double reLUAlpha = 0.01)
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

    static Matrix<double> LeakyReLUDeriv(Matrix<double> matrix, double reLUAlpha = 0.01)
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

    static List<SerialMatrix> ToSerialMatrixList(List<Matrix<double>> input)
    {
        List<SerialMatrix> output = new List<SerialMatrix>();
        foreach (Matrix<double> matrix in input.ToList())
        {
            output.Add(new SerialMatrix(matrix));
        }
        return output.ToList();
    }

    static List<Matrix<double>> ToMatrixList(List<SerialMatrix> input)
    {
        List<Matrix<double>> output = new List<Matrix<double>>();
        foreach (SerialMatrix matrix in input.ToList())
        {
            output.Add(Matrix<double>.Build.Dense(matrix.Rows, matrix.Columns, matrix.Data));
        }
        return output.ToList();
    }

    public static void WriteMatrix<T>(Matrix<T> matrix) where T : struct, IEquatable<T>, IFormattable
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
}

public class Cache
{
    public List<Matrix<double>> aVals = new List<Matrix<double>>();
}

public class Experience
{
    public double[] State, NextState;
    public int Action;
    public double Reward;
    public bool Done;

    public Experience(double[] state, int action, double reward, double[] nextState, bool done)
        => (State, Action, Reward, NextState, Done) = (state, action, reward, nextState, done);
}

public class NNTrainingData
{
    public List<SerialMatrix> weights { get; set; }
    public List<SerialMatrix> biases { get; set; }
    public int[] n { get; set; }
    public double reLUAlpha { get; set; }

    public NNTrainingData(List<SerialMatrix> weights, List<SerialMatrix> biases, int[] n, double reLUAlpha)
    {
        this.weights = weights.ToList();
        this.biases = biases.ToList();
        this.n = n.ToArray();
        this.reLUAlpha = reLUAlpha;
    }
}

public class SerialMatrix
{
    public int Rows { get; set; }
    public int Columns { get; set; }
    public double[]? Data { get; set; }

    public SerialMatrix() { }

    public SerialMatrix(Matrix<double> matrix)
    {
        Rows = matrix.RowCount;
        Columns = matrix.ColumnCount;
        Data = matrix.ToColumnMajorArray();
    }

    public Matrix<double> ToMatrix()
    {
        Matrix<double> matrix = Matrix<double>.Build.Dense(Rows, Columns, Data);
        return matrix.Clone();
    }
}