using NNN;
using System.Text.Json;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;

string tokenFileName = "tokens";
string inputTextFile = "input.txt";
string modelFile = "model.dat";
Console.WriteLine("Tokenize text input? y/n");
var userInput = Console.ReadLine();
string text = InputPreparer.ReadTextFile(inputTextFile);
if (userInput == "y")
{
    var (token, key) = InputPreparer.TokenizeText(text);
    InputPreparer.SaveTokens(tokenFileName, token, key);
}
var (stringKey, keyString) = InputPreparer.LoadTokens(tokenFileName);
var input = InputPreparer.DivideInput(text, stringKey);

int vocabSize = stringKey.Count;
int sequenceLength = 20; // Number of previous words to look at
int embedDim = 64;
int hiddenSize = 128;
int logInterval = 10;
float temperature = 0f;

var model = new MyLSTM(vocabSize, embedDim, hiddenSize);
var optimizer = torch.optim.Adam(model.parameters(), lr: 0.001);
var criterion = CrossEntropyLoss();

var (X, Y) = InputPreparer.CreateWindows(input, sequenceLength);
using var dataset = new MyDataset(X, Y);
int batchSize = 32;
var loader = DataLoader(dataset, batchSize, shuffle: true);

Console.WriteLine("Load model from file? y/n");
userInput = Console.ReadLine();
bool train = true;
if (userInput == "y")
{
    model.load(modelFile);
    Console.WriteLine("\nContinue training model? y/n");
    userInput = Console.ReadLine();
    if (userInput != "y")
    {
        train = false;
    }
}

if (train)
{
    Console.WriteLine("\nEnter number of training epochs: ");
    int epochs = Convert.ToInt32(Console.ReadLine());
    Console.WriteLine();
    Console.WriteLine("------------------------------");
    Console.WriteLine("Training neural network...");
    Console.WriteLine("------------------------------\n");
    model.train();
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float totalEpochLoss = 0;
        int batchCount = 0;
        DateTime startTime = DateTime.Now;

        foreach (var batch in loader)
        {
            using var scope = torch.NewDisposeScope();

            var inputs = batch["data"];
            var targets = batch["label"];

            var prediction = model.forward(inputs);
            var loss = torch.nn.functional.cross_entropy(prediction, targets);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            float currentLoss = loss.item<float>();
            totalEpochLoss += currentLoss;
            batchCount++;

            if (batchCount % logInterval == 0)
            {
                Console.WriteLine($"Epoch: {epoch} | Batch: {batchCount} | Current Loss: {currentLoss:F4}");
            }
        }

        float averageLoss = totalEpochLoss / batchCount;
        var duration = DateTime.Now - startTime;

        Console.WriteLine($"--- Epoch {epoch} Complete ---");
        Console.WriteLine($"Average Loss: {averageLoss:F4}");
        Console.WriteLine($"Duration: {duration.TotalSeconds:F2}s");
        Console.WriteLine("------------------------------");
    }

    Console.WriteLine("\nSave model data? y/n");
    userInput = Console.ReadLine();
    if (userInput == "y")
    {
        model.save(modelFile);
    }
}

int wordsToGenerate = 100;

Console.WriteLine();
Console.WriteLine("Testing RNN text generation...");
Console.WriteLine("Enter seed string: ");
string testString = Console.ReadLine();
Console.WriteLine();
long[] seed = InputPreparer.DivideInput(testString, stringKey)[0..sequenceLength];
Console.WriteLine($"Seed: {TextGenerator.ConvertToText(seed.ToList(), keyString)}");
Console.WriteLine();
Console.WriteLine($"Generating next {wordsToGenerate} words");
Console.WriteLine();
var finalIndexes = TextGenerator.GenerateText(model, seed, wordsToGenerate, sequenceLength, temperature);
Console.WriteLine($"Final text: {TextGenerator.ConvertToText(finalIndexes, keyString)}");
Console.WriteLine("\nPress any key to close...");
Console.ReadKey();

namespace NNN
{
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    public class MyLSTM : Module<Tensor, Tensor>
    {
        private readonly Embedding embedding; // Added
        private readonly LSTM lstm;
        private readonly Module<Tensor, Tensor> linear;

        public MyLSTM(int vocabSize, int embedDim, int hiddenSize) : base("MyLSTM")
        {
            this.embedding = Embedding(vocabSize, embedDim); // [Batch, Seq] -> [Batch, Seq, EmbedDim]
            this.lstm = LSTM(embedDim, hiddenSize, numLayers: 1, batchFirst: true);
            this.linear = Linear(hiddenSize, vocabSize); // Output size must match vocab size for CrossEntropy
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var embedded = embedding.forward(input);
            var (output, _, _) = lstm.forward(embedded);
            var lastStep = output.select(1, -1);
            return linear.forward(lastStep);
        }
    }

    public class MyDataset : Dataset
    {
        private readonly Tensor _inputs;
        private readonly Tensor _targets;

        public MyDataset(Tensor inputs, Tensor targets)
        {
            _inputs = inputs;
            _targets = targets;
        }

        public override long Count => _inputs.shape[0];

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            return new Dictionary<string, Tensor> {
            { "data", _inputs[TensorIndex.Single(index)] },
            { "label", _targets[TensorIndex.Single(index)] }
            };
        }
    }

    public static class InputPreparer
    {
        static readonly Regex tokenRegex = new(@"(?:[a-z'\-]+| |!|\.|\?|,|;|:|[0-9\.]+)+?");

        public static string ReadTextFile(string name)
        {
            return File.ReadAllText(name).ToLowerInvariant();
        }

        public static (Dictionary<string, long> stringKey, Dictionary<long, string> keyString) TokenizeText(string text)
        {
            text = text.ToLowerInvariant();
            var (stringKey, keyString) = (new Dictionary<string, long>(), new Dictionary<long, string>());
            var matches = tokenRegex.Matches(text);
            int index = 0;
            foreach (Match match in matches)
            {
                var token = match.Value;
                if (stringKey.TryAdd(token, index))
                {
                    keyString.Add(index, token);
                    index++;
                }
            }
            return (stringKey, keyString);
        }

        public static void SaveTokens(string fileName, Dictionary<string, long> stringKey, Dictionary<long, string> keyString)
        {
            string json = JsonSerializer.Serialize(stringKey);
            File.WriteAllText($"{fileName}.string", json);
            json = JsonSerializer.Serialize(keyString);
            File.WriteAllText($"{fileName}.key", json);
        }

        public static (Dictionary<string, long> stringKey, Dictionary<long, string> keyString) LoadTokens(string fileName)
        {
            string json = File.ReadAllText($"{fileName}.string");
            Dictionary<string, long> stringKey = JsonSerializer.Deserialize<Dictionary<string, long>>(json);
            json = File.ReadAllText($"{fileName}.key");
            Dictionary<long, string> keyString = JsonSerializer.Deserialize<Dictionary<long, string>>(json);
            return (stringKey, keyString);
        }

        public static long[] DivideInput(string text, Dictionary<string, long> stringKey)
        {
            text = text.ToLowerInvariant();
            var matches = tokenRegex.Matches(text);
            var output = new long[matches.Count];
            for (int i = 0; i < matches.Count; i++)
            {
                string word = matches[i].Value;
                if (stringKey.TryGetValue(word, out var index))
                {
                    output[i] = index;
                }
            }
            return output;
        }

        public static (Tensor X, Tensor Y) CreateWindows(long[] tokenIndices, int windowSize)
        {
            int numSamples = tokenIndices.Length - windowSize;
            var inputs = new long[numSamples, windowSize];
            var targets = new long[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < windowSize; j++)
                    inputs[i, j] = tokenIndices[i + j];

                targets[i] = tokenIndices[i + windowSize];
            }

            return (torch.tensor(inputs), torch.tensor(targets));
        }
    }

    public static class TextGenerator
    {
        public static List<long> GenerateText(MyLSTM model, long[] seedIndices, int wordsToGenerate, int windowSize, float temperature)
        {
            model.eval();
            var generated = new List<long>(seedIndices);

            for (int i = 0; i < wordsToGenerate; i++)
            {
                using var scope = torch.NewDisposeScope();

                var recentIndices = generated.Skip(Math.Max(0, generated.Count - windowSize)).ToArray();

                var input = torch.tensor(recentIndices).reshape(1, -1);

                using var logits = model.forward(input);

                long nextWordIndex = 0;
                if (temperature == 0.0f)
                {
                    nextWordIndex = logits.argmax(1).item<long>();
                }
                else
                {
                    var scaledLogits = logits.div(temperature);
                    var probabilities = softmax(scaledLogits, dim: -1);
                    var nextWordTensor = multinomial(probabilities, num_samples: 1);
                    nextWordIndex = nextWordTensor[0].item<long>();
                }

                generated.Add(nextWordIndex);
            }

            return generated;
        }

        public static string ConvertToText(List<long> indexes, Dictionary<long, string> keyString)
        {
            string text = string.Empty;
            foreach (var index in indexes)
            {
                if (keyString.TryGetValue(index, out var word))
                {
                    text += word;
                }
            }
            return text;
        }
    }
}