using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;
using System.Linq;
using System.Collections.Generic;
using System;

public class SimpleEnv
{
    public int XPosition { get; private set; }
    public int YPosition { get; private set; }
    public int XGoal = 8;
    public int YGoal = -5;

    public double[] Reset()
    {
        XPosition = 0;
        YPosition = 0;
        return GetState();
    }

    public (double[] nextState, double reward, bool done) Step(int dx, int dy)
    {
        double prevDist = Math.Sqrt(Math.Pow(XGoal - XPosition, 2) + Math.Pow(YGoal - YPosition, 2));

        // Apply movement
        XPosition += dx;
        YPosition += dy;

        double dist = Math.Sqrt(Math.Pow(XGoal - XPosition, 2) + Math.Pow(YGoal - YPosition, 2));
        double deltaDist = prevDist - dist;
        bool done = false;
        double reward = 0.0;
        reward += Math.Clamp(deltaDist, -1.0, 1.0); // Reward for moving toward goal
        if (dist < 0.3) { reward += 10; done = true; }    // Goal reached
        if (dx == 0 && dy == 0) { reward -= 0.1; }
        reward -= 0.3;

        return (GetState(), reward, done);
    }

    public double[] GetState()
    {
        return new double[] { XPosition, YPosition };
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        DQNAgent agent = new DQNAgent();
        SimpleEnv env = new SimpleEnv();

        int episodes = 500;
        int maxSteps = 50;

        for (int e = 0; e < episodes; e++)
        {
            if (e == Math.Round(episodes * 0.9)) { agent.epsilon = 0.0; }
            double[] state = env.Reset();
            double totalReward = 0;

            for (int step = 0; step < maxSteps; step++)
            {
                // Get action (integer 0..8)
                int action = agent.Act(state);

                // Decode to (dx, dy)
                var (dx, dy) = agent.DecodeAction(action);

                // Step environment
                var (nextState, reward, done) = env.Step(dx, dy);

                // Remember experience
                agent.Remember(state, action, reward, nextState, done);

                // Train on replay buffer
                agent.Replay();

                state = nextState;
                totalReward += reward;

                if (done) break;
            }

            Console.WriteLine($"Episode {e + 1}/{episodes}, Total Reward: {totalReward:F2}, Epsilon: {agent.Epsilon:F2}");
        }

        Console.WriteLine("\nTraining complete. Testing greedy policy...\n");

        // --- Test phase (no exploration) ---
        double[] testState = env.Reset();
        double finalReward = 0;
        for (int step = 0; step < 10; step++)
        {
            int action = agent.Act(testState);
            var (dx, dy) = agent.DecodeAction(action);

            var (nextState, reward, done) = env.Step(dx, dy);
            finalReward += reward;
            Console.WriteLine($"Step {step}: State=({testState[0]}, {testState[1]}), " +
                              $"Action=(dx={dx}, dy={dy}), Next=({nextState[0]}, {nextState[1]}), Reward={reward:F2}");

            testState = nextState;
            if (done) break;
        }
        Console.WriteLine($"Total Reward={finalReward:F2}");
        Console.WriteLine("\nTesting finished.");
        Console.WriteLine("Press any key to close...");
        Console.ReadKey();
    }
}

public class DQNAgent
{
    private readonly int stateSize = 2;
    private readonly int actionSize = 9;

    private double gamma = 0.99;
    public double epsilon = 1.0;
    private double epsilonDecay = 0.999;
    private double epsilonMin = 0.05;

    private int batchSize = 64;
    private int targetUpdateFreq = 200;
    private double tau = 0.05;
    private int maxBufferSize = 10000;

    private int stepCount = 0;
    private Random random = new Random();

    private NeuralNetwork model;
    private NeuralNetwork targetModel;

    private List<(double[] state, int action, double reward, double[] nextState, bool done)> replayBuffer
        = new List<(double[], int, double, double[], bool)>();

    private (int dx, int dy)[] moveMap = new (int, int)[]
    {
        (-1,-1), (-1,0), (-1,1),
        ( 0,-1), ( 0,0), ( 0,1),
        ( 1,-1), ( 1,0), ( 1,1)
    };

    public DQNAgent()
    {
        model = new NeuralNetwork(inputNeurons: stateSize, hiddenLayers: 1, hiddenNeurons: 14, outputNeurons: actionSize);
        targetModel = new NeuralNetwork(inputNeurons: stateSize, hiddenLayers: 1, hiddenNeurons: 14, outputNeurons: actionSize);
        SyncTargetModel();
    }

    // --- Action selection (epsilon-greedy)
    public int Act(double[] state)
    {
        if (random.NextDouble() < epsilon)
            return random.Next(actionSize);

        var stateMat = Matrix<double>.Build.Dense(1, state.Length, state);
        var qMatrix = model.ProcessInput(stateMat);
        double[] qValues = qMatrix.Row(0).ToArray();

        int bestAction = Array.IndexOf(qValues, qValues.Max());
        return bestAction;
    }

    public (int dx, int dy) DecodeAction(int action)
    {
        return moveMap[action];
    }

    // --- Store experience
    public void Remember(double[] state, int action, double reward, double[] nextState, bool done)
    {
        if (replayBuffer.Count >= maxBufferSize) replayBuffer.RemoveAt(0);
        replayBuffer.Add((state, action, reward, nextState, done));
    }

    // --- Training via experience replay
    public void Replay()
    {
        if (replayBuffer.Count < batchSize)
            return;

        // --- Sample random minibatch
        var batch = replayBuffer.OrderBy(x => random.Next())
                                .Take(batchSize)
                                .ToList();

        var inputList = new List<double[]>();
        var targetList = new List<double[]>();

        foreach (var (state, action, reward, nextState, done) in batch)
        {
            // Predict current Q-values
            var qCurrent = model.ProcessInput(Matrix<double>.Build.Dense(1, state.Length, state))
                               .Row(0).ToArray();

            // Predict next Q-values using *target network*
            var qNext = targetModel.ProcessInput(Matrix<double>.Build.Dense(1, nextState.Length, nextState))
                                   .Row(0).ToArray();

            double qTarget = reward + (done ? 0.0 : gamma * qNext.Max());

            qCurrent[action] = qTarget;

            inputList.Add(state);
            targetList.Add(qCurrent);
        }

        var inputMatrix = Matrix<double>.Build.DenseOfRowArrays(inputList);
        var targetMatrix = Matrix<double>.Build.DenseOfRowArrays(targetList);

        // Train model
        model.SetTrainingData(inputMatrix, targetMatrix);
        model.Train(epochs: 2, alpha: 0.0015, clipThreshold: 10);

        // Epsilon decay
        if (epsilon > epsilonMin)
            epsilon *= epsilonDecay;

        stepCount++;
        if (stepCount % targetUpdateFreq == 0)
            SoftUpdateTarget(tau);
    }

    private void SoftUpdateTarget(double tau = 0.01)
    {
        var (weightsModel, biasesModel) = model.GetWeights();
        var (weightsTarget, biasesTarget) = targetModel.GetWeights();

        for (int i = 0; i < weightsModel.Count; i++)
        {
            weightsTarget[i] = (1 - tau) * weightsTarget[i] + tau * weightsModel[i];
            biasesTarget[i] = (1 - tau) * biasesTarget[i] + tau * biasesModel[i];
        }
        targetModel.SetWeights(weightsTarget, biasesTarget);
    }

    private void SyncTargetModel()
    {
        var (weights, biases) = model.GetWeights();
        targetModel.SetWeights(weights, biases);
    }

    public double Epsilon => epsilon;
}

public class NeuralNetwork
{
    int epochs = 10;
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
    bool logTraining = false;

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

    public (Matrix<double> yHat, Cache cache) FeedForward(Matrix<double> input)
    {
        Cache cache = new Cache();
        Matrix<double> a = input.Clone();
        cache.aVals.Add(a);
        for (int i = 0; i < wVals.Count; i++)
        {
            Matrix<double> z = wVals[i] * a + Broadcast(bDefVals[i], a.ColumnCount);
            cache.zVals.Add(z);
            if (i < wVals.Count - 1) a = LeakyReLU(z);
            else a = z;
            cache.aVals.Add(a);
        }
        return (a, cache);
    }

    void HandleBackprop(Matrix<double> yHat, Matrix<double> y, int m, Cache cache)
    {
        List<Matrix<double>> dC_dWs = new List<Matrix<double>>();
        List<Matrix<double>> dC_dbs = new List<Matrix<double>>();
        int L = wVals.Count;
        Matrix<double> aL = yHat;
        Matrix<double> aPrev = cache.aVals[L - 1];
        Matrix<double> dC_dZ = CostDeriv(aL, y, costDelta, m);
        Matrix<double> dC_dW = dC_dZ * aPrev.Transpose();
        Matrix<double> dC_db = CalculateBiasC(dC_dZ, m);
        dC_dWs.Insert(0, dC_dW);
        dC_dbs.Insert(0, dC_db);
        Matrix<double> dC_dA_prev = wVals[L - 1].Transpose() * dC_dZ;
        for (int l = L - 2; l >= 0; l--)
        {
            Matrix<double> aPrevHidden = cache.aVals[l];
            Matrix<double> zHidden = cache.zVals[l];
            Matrix<double> dZ = dC_dA_prev.PointwiseMultiply(LeakyReLUDeriv(zHidden));
            Matrix<double> dW = dZ * aPrevHidden.Transpose();
            Matrix<double> db = CalculateBiasC(dZ, m);
            dC_dWs.Insert(0, dW);
            dC_dbs.Insert(0, db);
            if (l > 0) dC_dA_prev = wVals[l].Transpose() * dZ;
        }
        (dC_dWs, dC_dbs) = ClipGradients(dC_dWs, dC_dbs, clipThreshold);
        for (int i = 0; i < wVals.Count; i++)
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
        Matrix<double> xT = input.Clone().Transpose();
        var (yHat, _) = FeedForward(xT);
        return yHat.Transpose();
    }

    public string CreateJsonString()
    {
        NNTrainingData training = new NNTrainingData(ToSerialMatrixList(wVals), ToSerialMatrixList(bDefVals), n, reLUAlpha);
        string jsonString = JsonSerializer.Serialize(training);
        return jsonString;
    }

    public NeuralNetwork(int inputNeurons, int hiddenLayers, int hiddenNeurons, int outputNeurons)
    {
        wVals = new List<Matrix<double>>();
        bDefVals = new List<Matrix<double>>();
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

    static Matrix<double> CalculateBiasC(Matrix<double> dC_dZ, int batchSize)
    {
        // average across columns (examples)
        Vector<double> sums = dC_dZ.RowSums() / batchSize;
        Matrix<double> result = Matrix<double>.Build.Dense(dC_dZ.RowCount, 1);
        for (int i = 0; i < sums.Count; i++) result[i, 0] = sums[i];
        return result;
    }

    double Cost(Matrix<double> yHat, Matrix<double> y, int m)
    {
        Matrix<double> losses = Matrix<double>.Build.Dense(yHat.RowCount, yHat.ColumnCount);
        y = y.Clone().Transpose();
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
        y = y.Clone().Transpose();
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
    public List<Matrix<double>> zVals = new List<Matrix<double>>();
}

public class Experience
{
    public double[] State, NextState;
    public int[] Action;
    public double Reward;
    public bool Done;

    public Experience(double[] state, int[] action, double reward, double[] nextState, bool done)
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