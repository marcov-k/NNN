using System;

namespace NNNCSharp.Components.Utilities
{
    public static class NNNLog
    {
        public static Action<string>? Output { get; set; }

        public static void Write(string message = "")
        {
            Output?.Invoke(message);
        }

        public static void WriteLine(string message = "")
        {
            Output?.Invoke(message + "\n");
        }
    }
}
