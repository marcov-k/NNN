using System;
using System.Collections.Generic;
using System.Linq;

namespace NNNCSharp.Components.Episodes
{
    /// <summary>
    /// Record containing all of the experiences within a single DQN training episode.
    /// </summary>
    public record Episode : IDisposable
    {
        /// <summary>
        /// List of experiences contained in the episode.
        /// </summary>
        public List<Experience> Experiences { get; init; }

        /// <summary>
        /// Creates a new episode instance.
        /// </summary>
        /// <param name="experiences">List of experiences within the episode.</param>
        public Episode(List<Experience> experiences)
        {
            Experiences = experiences.ToList(); // create a new copy of each experience
        }

        public void Dispose()
        {
            foreach (var exp in Experiences) exp.Dispose();
        }
    }
}
