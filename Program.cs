using CustomTrialTest;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.TorchSharp;
using static Microsoft.ML.DataOperationsCatalog;


// Initialize MLContext
MLContext ctx = new MLContext();

// Define data path
var dataPath = Path.GetFullPath(@"..\..path to \yelp_labelled.csv");

// Infer column information
ColumnInferenceResults columnInference =
    ctx.Auto().InferColumns(dataPath, labelColumnName: "Label", groupColumns: false, allowQuoting: true, separatorChar: ',');

// Create text loader
TextLoader loader = ctx.Data.CreateTextLoader(columnInference.TextLoaderOptions);

// Load data into IDataView
IDataView data = loader.Load(dataPath);

TrainTestData trainValidationData = ctx.Data.TrainTestSplit(data, testFraction: 0.2);

// Initialize search space
var tcSearchSpace = new SearchSpace<TCOption>();

// Create factory for Text Classification trainer
var tcFactory = (MLContext context, TCOption param) =>
{
    return context.MulticlassClassification.Trainers.TextClassification(
        sentence1ColumnName: nameof(InputModel.col0),
        batchSize: param.BatchSize);
};

// Create text classification sweepable estimator
var tcEstimator = ctx.Auto().CreateSweepableEstimator(tcFactory, tcSearchSpace);

// Define text classification pipeline
SweepablePipeline pipeline =
    ctx.Transforms.Conversion.MapValueToKey("Label", "Label")
.Append(tcEstimator)
.Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

var tcRunner = new TCRunner(context: ctx, data: trainValidationData, pipeline: pipeline);

AutoMLExperiment experiment = ctx.Auto().CreateExperiment();

experiment
    .SetPipeline(pipeline)
    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName)
    .SetTrainingTimeInSeconds(60)
    .SetDataset(trainValidationData)
    .SetTrialRunner(tcRunner);

//Log experiment trials
ctx.Log += (_, e) =>
{
    if (e.Source.Equals("AutoMLExperiment"))
    {
        Console.WriteLine(e.RawMessage);
    }
};

var tcCts = new CancellationTokenSource();
TrialResult experimentResults = await experiment.RunAsync(tcCts.Token);

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("\nBEST METRIC: " + experimentResults.Metric);
Console.ResetColor();

var predictor = ctx.Model.CreatePredictionEngine<InputModel, ResultModel>(experimentResults.Model);

var res = predictor.Predict(new InputModel { col0 = "its bad its terrible worst" });


Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"It is {res.PredictedLabel} that it is a positive comment");
Console.ResetColor();