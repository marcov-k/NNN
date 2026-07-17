using NNNCSharp.Components.Episodes;
using System;
using System.Collections.Generic;

namespace NNNCSharp.Components.Buffers
{
    /// <summary>
    /// PER replay buffer class.
    /// </summary>
    /// <param name="capacity">Maximum size of the replay buffer.</param>
    /// <param name="alpha">Alpha value to use during sampling.</param>
    public class ReplayBuffer
    {
        // Internal data
        /// <summary>
        /// Internal sum tree for experiences.
        /// </summary>
        readonly SumTree<Experience> SumTree;
        /// <summary>
        /// Highest current priority of a stored experience.
        /// </summary>
        double MaxPriority = 1.0;

        // PER Parameters
        /// <summary>
        /// Sampling alpha value.
        /// </summary>
        readonly double Alpha; // determines how heavily experiences are prioritized
        /// <summary>
        /// Current importance sampling exponent.
        /// </summary>
        double Beta = 0.4; // determines how heavily sampling bias is corrected in gradient updates
        /// <summary>
        /// Linear importance sampling exponent increment.
        /// </summary>
        const double BetaIncrement = 0.001;

        // Public properties
        /// <summary>
        /// Number of experiences currently stored in the buffer.
        /// </summary>
        public int Count => SumTree.Count;

        // Utilities
        /// <summary>
        /// Current Random instance.
        /// </summary>
        readonly Random Random = new();

        public ReplayBuffer(int capacity, double alpha = 0.6)
        {
            SumTree = new SumTree<Experience>(capacity);
            Alpha = alpha;
        }

        /// <summary>
        /// Adds a new experience to the buffer.
        /// </summary>
        /// <param name="experience">Experience to add.</param>
        public void Add(Experience experience)
        {
            // Assign max priority initially
            double priority = Math.Pow(MaxPriority, Alpha);
            SumTree.Add(experience, priority);
        }

        /// <summary>
        /// Gets a batch of experiences using PER sampling.
        /// </summary>
        /// <param name="batchSize">Number of experiences in the batch.</param>
        /// <returns>List of experiences in the batch, their corresponding sum tree indices, and their PER training weights.</returns>
        public (List<Experience> batch, int[] indices, double[] weights) GetBatch(int batchSize)
        {
            List<Experience> batch = new(batchSize);
            var indices = new int[batchSize];
            var weights = new double[batchSize];

            double totalPriority = SumTree.TotalPriority;
            double segment = totalPriority / batchSize;
            double maxWeight = 0;

            // PER sample batchSize experiences
            for (int i = 0; i < batchSize; i++)
            {
                // Calculate PER sampling value
                double low = segment * i;
                double high = segment * (i + 1);
                double sampleVal = low + (Random.NextDouble() * (high - low));

                // Sample next experience
                var (treeIndex, priority, item) = SumTree.Get(sampleVal);

                batch.Add(item);
                indices[i] = treeIndex;
                weights[i] = priority;
            }

            // Calculate training weights for batch experiences
            for (int i = 0; i < batchSize; i++)
            {
                // Calculate probability of experience being sampled
                double prob = weights[i] / totalPriority;
                if (prob == 0) prob = 1e-8;

                // Calculate weight account for sampling bias
                double weight = Math.Pow(1.0 / (SumTree.Count * prob), Beta);
                weights[i] = weight;
                if (weight > maxWeight) maxWeight = weight;
            }

            // Normalize all experience training weights
            if (maxWeight > 0)
            {
                for (int i = 0; i < batchSize; i++)
                {
                    weights[i] /= maxWeight;
                }
            }

            // Increment importance sampling exponent
            Beta = Math.Min(1.0, Beta + BetaIncrement);

            return (batch, indices, weights);
        }

        /// <summary>
        /// Updates the PER sampling priorities of a given set of experiences.
        /// </summary>
        /// <param name="indices">Indices of experiences to update.</param>
        /// <param name="priorities">New priority values.</param>
        public void UpdatePriorities(int[] indices, double[] priorities)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                // Update maximum priority if necessary
                if (priorities[i] > MaxPriority)
                {
                    MaxPriority = priorities[i];
                }

                // Update priority of experience in sum tree
                double powerPriority = Math.Pow(priorities[i], Alpha);
                SumTree.Update(indices[i], powerPriority);
            }
        }
    }
}
