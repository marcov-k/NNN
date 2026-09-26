using System;

namespace NNNCSharp.Components.Utilities.Exceptions
{
    /// <summary>
    /// Represents an error due to attempting to parse an invalid file format.
    /// </summary>
    public class InvalidFileFormatException : Exception
    {
        /// <summary>
        /// Creates a new InvalidFileFormatException instance.
        /// </summary>
        public InvalidFileFormatException() : base() { }

        /// <summary>
        /// Creates a new InvalidFileFormatException instance.
        /// </summary>
        /// <param name="message">Error message to display.</param>
        public InvalidFileFormatException(string message) : base(message) { }

        /// <summary>
        /// Creates a new InvalidFileFormatException instance.
        /// </summary>
        /// <param name="message">Error message to display.</param>
        /// <param name="innerException">Reference to the inner exception which caused this exception.</param>
        public InvalidFileFormatException(string message, Exception innerException) : base(message, innerException) { }
    }
}
