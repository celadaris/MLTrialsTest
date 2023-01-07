using Microsoft.ML.Data;
using System.Numerics;

internal class ResultModel
{
    public bool PredictedLabel { get; set; }
    public float[] Score { get; set; }
}