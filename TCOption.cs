using Microsoft.ML.SearchSpace;

namespace CustomTrialTest
{
    internal class TCOption
    {
        [Range(64, 128, 32)]
        public int BatchSize { get; set; }
    }
}
