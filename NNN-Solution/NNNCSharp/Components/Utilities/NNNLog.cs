using System;

namespace NNNCSharp.Components.Utilities
{
    /// <summary>
    /// Static class for writing logs to a given output.
    /// </summary>
    public static class NNNLog
    {
        /// <summary>
        /// Output to write logs to.
        /// </summary>
        public static Action<string>? Output { get; set; }

        /// <summary>
        /// Prints the given message to the predefined output.
        /// </summary>
        /// <param name="message"></param>
        public static void Write(string message = "")
        {
            Output?.Invoke(message);
        }

        /// <summary>
        /// Prints the given message to the predefined output and moves the cursor to the next line.
        /// </summary>
        /// <param name="message"></param>
        public static void WriteLine(string message = "")
        {
            Output?.Invoke(message + "\n");
        }
    }
}
