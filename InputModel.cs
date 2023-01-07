using Microsoft.ML.Data;

internal class InputModel
{
    [LoadColumn(0)]
    public string col0 { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
}