---
layout: post
title: "End-to-end AutoML pipeline on VertexAI with Go"
date: 2023-06-08 08:00:00
categories: golang vertexai
summary: ""
authors:
    - pgaleone
    - chatgpt
---

Automated Machine Learning (AutoML) has revolutionized the way we approach building and deploying machine learning models. Gone are the days of painstakingly handcrafting complex models from scratch. With AutoML, the process becomes faster, more accessible, and remarkably efficient.

AutoML encompasses a suite of techniques and tools that automate various stages of the machine learning pipeline, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and even deployment. By automating these steps, AutoML eliminates much of the manual labor and expertise traditionally required, allowing developers and data scientists to focus on higher-level tasks and domain-specific challenges.

One of the key advantages of AutoML is its ability to accelerate the model development process. By automating repetitive and time-consuming tasks, such as data preprocessing and feature engineering, AutoML drastically reduces the time and effort needed to go from raw data to a fully trained model. This acceleration enables faster experimentation, iteration, and deployment of models, empowering teams to deliver valuable insights and predictions more rapidly.

Moreover, AutoML democratizes machine learning by making it accessible to a wider audience. With AutoML, individuals without extensive machine learning expertise can leverage sophisticated algorithms and techniques to build high-performing models. This accessibility fosters collaboration, encourages innovation, and breaks down barriers to entry in the field of machine learning.

In this article, we will explore how Google Cloud's VertexAI and the power of AutoML can be harnessed to develop and deploy tabular models efficiently. The focus of the article on tabular data is not a case: there's pretty much 0 documentation on tabular data AutoML pipelines on VertexAI and even less about the VertexAI Go client.

## VertexAI Go client

Unfortunately, the official documentation for the Go client is not at the same level of the clients for other languages: Python is extremely well documented and rich of examples, as well as Java (!). Go, instead, have close to 0 examples on the official page (e.g. [Create a dataset for Tabular Cloud Storage](https://cloud.google.com/vertex-ai/docs/samples/aiplatform-create-dataset-tabular-gcs-sample) has examples only in Node.js, Python and Java).

The only available documentation is the godoc of the **auto-generated** [(beta!) aiplatform/apiv1beta1](https://pkg.go.dev/cloud.google.com/go/aiplatform/apiv1beta1). We **must** use the beta package because the stable one does **not** have the support for the **tabular data**.

Being auto-generated the interface of the Go client is not simplified and clean as (for example) the Python client. We'll see how cumbersome using it can be soon.

## The Scalable AutoML workflow

Ideally, the AutoML workflow should be so trivial to be almost no-code: one should upload the data, define the task, let the AutoML service do its magic and we should get our trained model.

In practice, instead, if we want to use the various clients and not the web interface there's **a lot** of code to write. This is, however, the only way to create a scalable workflow. Using a client we can automatize every process of the AutoML pipeline.

When working with tabular data we have a fixed number of steps to follow (and to automatize):

1. Gather the data and upload to a supported storage. VertexAI allows 2 type of storages for structured data: BigQuery and GCS (Google Cloud Storage). When using BigQuery, the structure of the data is defined by the tables themselves; when using GCS the structure is defined by the header of the first CSV (in alphabetically order) read.
2. Create a **dataset**: in this context a dataset is an association between the data and the "abstract entity dataset" that will be created on VertexAI.
3. Create the training pipeline. This is the expensive step. The training pipeline instantiates some hardware (configurable) and start using it to train various models, comparing them, and choosing the best one. The best model is chosen according to the specified metric. The training pipeline can also apply transformations to the previously created dataset, and it's in this step that we can decide what task to solve (classification, regression, ...) and what's the target column.
4. Deploy the model: create an endpoint and decide how this endpoints reacts. Should the model be used for batch or online predictions?

Let's see how to perform these steps completely in Go.

## The task

As usual, it would make no sense to implement all of this without a Goal. In our case the goal is to predict the sleep efficiency (a numerical score between 0 and 100, that represents the percentage of sleep respect to the time spent in bed) given a set of other attributes gathered during the day.

The data comes in CSV format. This is a (simplified, only a few attributes are shown) view of the data: there's the header and a row.

|CaloriesSum|StepsSum|AverageHeartRate|ActivityCalories|Calories|Distance|PeakMinBPM|**SleepEfficiency**|
|-----------|--------|----------------|----------------|--------|--------|----------|---------------|
|569        |1015    |127.00          |1889.00         |3283.00 |6.09    |169       |43             |

Now that we have the task defined, we can go straight to the code.

## Prerequisites: the project and the service file

It's mandatory to correctly organize the workspace for this project. So we must create a new project in the Google Cloud console. This is a standard practice, so I just link to the [documentation](https://developers.google.com/workspace/guides/create-project).

From now on we suppose that the project ID is contained inside the environment variable `VAI_PROJECT_ID`.

The idea is to deploy this code somewhere, and thus the best thing to do is to create a Service Account. This is a standard practice when working with Google Cloud so I'll just link to the official documentation: [Create service accounts](https://cloud.google.com/iam/docs/service-accounts-create).

The most important part, is the role setup. For the AutoML workflow we need the service account to be both:

- Vertex AI Administrator
- Storage Admin

Once created, please download the **service account key** (JSON) and store it in a secure and known location. From now on we suppose that the environment variable `VAI_SERVICE_ACCOUNT_KEY` points to the path of the service account key file.

## Data Storage

As anticipated, tabular data has only two possible locations: BigQuery and CSV files on GCS. Since in our task the data comes in a CSV we'll use a bucket on GCS.

Although tightly connected with VertexAI the data storage is not managed by the VertexAI package (that's still called aiplatform), but it has its own package: [cloud.google.com/go/storage](https://cloud.google.com/go/storage).

Buckets on GCP have their own peculiarities: the bucket name **must** be both globally unique and DNS compliant ([reference](https://cloud.google.com/storage/docs/naming-buckets)).

The pattern used to interact with all the cloud service (storage or VertexAI) is always the same: create a client and use the client to interact with the cloud service. All the clients require a context (in practice always created with `context.Background()`).

```go
ctx := context.Background()

// Create the storage client using the service account key file.
var storageClient *storage.Client
if storageClient, err = storage.NewClient(ctx, option.WithCredentialsFile(os.Getenv("VAI_SERVICE_ACCOUNT_KEY"))); err != nil {
    return err
}
defer storageClient.Close()

// Globally unique: we can use the project id as a prefix
bucketName := fmt.Sprintf("%s-user-data", os.Getenv("VAI_PROJECT_ID"))
bucket := storageClient.Bucket(bucketName)
if _, err = bucket.Attrs(ctx); err != nil {
    // GCP bucket.Attrs returns an error if the bucket does not exist
    // In theory it should be storage.ErrBucketNotExist, but in practice it's a generic error
    // So we try to create the bucket hoping that the error is due to the bucket not existing
    if err = bucket.Create(ctx, os.Getenv("VAI_PROJECT_ID"), nil); err != nil {
        return err
    }
}
```

Please read the comment in the error handling section, because it's a bad behavior of the client and it took me a bit of time to discover it.

With these few lines, we have been able to create a bucket (with a globally unique name) once and avoid to ask for a new creation if the bucket already exists.

The second step is to upload the CSV on the bucket. However, since we want to use AutoML we need **at least** 1000 rows. This is not documented anywhere, and I discovered it only after setting up the whole pipeline and receiving an error (lol).

Supposing that we have all the CSV rows gathered in the slice `allUserData` we can **sample** the rows to reach at least 1000 rows.

```go
// VertexAI autoML pipeline requires at least 1000 rows.
// Thus we random sample the data to get 1000 rows
if len(allUserData) < 1000 {
	tot := len(allUserData) - 1
	for i := 0; i < 1000-tot+2; i++ {
		allUserData = append(allUserData, allUserData[rand.Intn(tot)])
	}
}
```

The `bucket` object can be used to check if a file exists (this time it works, differently for the existence of the bucket itself), and write the csv only if it doesn't exists.

Let's suppose that we have a function that accepts all the CSV rows (`allUserData`) prepends a line containing the header, and eventually concantenates in a single strings all the lines. This is the `userDataToCsv` function.

```go
var csv string
if csv, err = userDataToCSV(allUserData); err != nil {
    return err
}
// csv now is our complete csv file content, to be stored on the bucket

csvOnBucket := "user_data.csv"
obj := bucket.Object(csvOnBucket)
if _, err = obj.Attrs(ctx); err == storage.ErrObjectNotExist {
    w := obj.NewWriter(ctx)
    if _, err := w.Write([]byte(csv)); err != nil {
        return err
    }
    if err := w.Close(); err != nil {
        return err
    }
}
```

First step completed. We have a bucket and we have a CSV file on the bucket.

## Create a Dataset on VertexAI

As previously mentioned, in the VertexAI context a dataset can be thought as an association between the data on the bucket and an abstract entity used by the training pipeline to loop over and transform the data.

Now can can start using the VertexAI client, imported as follows. Reminder: we must use the beta package because the stable package doesn't provide support for tabular data.

```go
import (
	vai "cloud.google.com/go/aiplatform/apiv1beta1"
	vaipb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
    "google.golang.org/protobuf/types/known/structpb"
)
```

Together with the VertexAI client (`vai`) we import also the protobuf package of VertexAI containing the protobuf to use to make request together with the `structpb` package that we'll later use to easily go from JSON to protobuf and vice-versa.

To work with datasets we have to create a `DatasetClient`. Although **no documented anywhere** it's **mandatory** to specify the VertexAI location of the project (the region, e.g. `us-central1`): [reference](https://github.com/googleapis/google-cloud-go/issues/6649#issuecomment-1242515615). We suppose that the location is available in the `VAI_LOCATION` environment variable.

```go
// Create a dataset from the training data (associate the bucket with the dataset)
var datasetClient *vai.DatasetClient
vaiEndpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", os.Getenv("VAI_LOCATION"))
if datasetClient, err = vai.NewDatasetClient(
    ctx,
    option.WithEndpoint(vaiEndpoint),
    option.WithCredentialsFile(os.Getenv("VAI_SERVICE_ACCOUNT_KEY"))); err != nil {
    return err
}
defer datasetClient.Close()
```

With the `datasetClient` object we can manager datasets available in the specified project (that's present in the service account key) and in the specified location (region).

We don't want to create a new dataset if it already exists, so we can use the `ListDatasets` method that allows us to list and filter the available datasets. If the `err != nil` (so no dataset has been found matching the filter criteria) then we create the dataset that will loop over the CSV stored in the bucket.

```go
datasetName := "user-dataset"
var dataset *vaipb.Dataset
datasetsIterator := datasetClient.ListDatasets(ctx, &vaipb.ListDatasetsRequest{
    Parent: fmt.Sprintf("projects/%s/locations/%s", os.Getenv("VAI_PROJECT_ID"), os.Getenv("VAI_LOCATION")),
    Filter: fmt.Sprintf(`display_name="%s"`, datasetName),
})

if dataset, err = datasetsIterator.Next(); err != nil {
    // The correct format is: {"input_config": {"gcs_source": {"uri": ["gs://bucket/path/to/file.csv"]}}}
    // Ref the code here: https://cloud.google.com/vertex-ai/docs/samples/aiplatform-create-dataset-tabular-gcs-sample
    csvURI := fmt.Sprintf("gs://%s/%s", bucketName, csvOnBucket)
    var metadata structpb.Struct
    err = metadata.UnmarshalJSON([]byte(fmt.Sprintf(`{"input_config": {"gcs_source": {"uri": ["%s"]}}}`, csvURI)))
    if err != nil {
        return err
    }

    req := &vaipb.CreateDatasetRequest{
        // Required. The resource name of the Location to create the Dataset in.
        // Format: `projects/{project}/locations/{location}`
        Parent: fmt.Sprintf("projects/%s/locations/%s", os.Getenv("VAI_PROJECT_ID"), os.Getenv("VAI_LOCATION")),
        Dataset: &vaipb.Dataset{
            DisplayName: datasetName,
            Description: "user data",
            // No metadata schema because it's a tabular dataset, and "tabular dataset does not support data import"
            MetadataSchemaUri: "gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml",
            Metadata: structpb.NewStructValue(&metadata),
        },
    }

    var createDatasetOp *vai.CreateDatasetOperation
    if createDatasetOp, err = datasetClient.CreateDataset(ctx, req); err != nil {
        return err
    }
    if dataset, err = createDatasetOp.Wait(ctx); err != nil {
        return err
    }
}
```

The correct parameters for the `CreateDatasetRequest` haven't been trivial to discover, so it's worth detailing the non-trivial one:

- `MetadataSchemaUri`: Thus parameter must contain a valid schema in YAML format among the supported schemas. You can use the `gcloud storage ls <path>` command to see all the available schemas. For tabular data the full path of the schema is `gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml`
- `MetadataSchema`: for tabular data this attribute **must be empty**. And it makes sense! Since the schema of the data is available in the data itself (the CSV header or the tables descriptions).
- `Metadata`: the most complex parameter to find. But it's where the association between the CSV on the bucket and the dataset happens. The Go API wants this value as a `structpb.Value`, but in practice is just JSON that will be sent to the server. Of course, the JSON must be well-formatted and all the required fields must be present. I found the correct JSON to send looking at the [Java (!) example](https://cloud.google.com/vertex-ai/docs/samples/aiplatform-create-dataset-tabular-gcs-sample). The easiest way to go from JSON to `structpb.Value` is to use the `UnmarshalJSON` method of `structpb.Struct`, that will create a `structpb.Struct` from a JSON, and than convert the `structpb.Struct` to a `structpb.Value` with the `structpb.NewStructValue` function call.

So far so good. We now have a `*vaipb.Dataset` to use in our training pipeline. Let's go create it!

## Training pipeline on VertexAI

As usual, the first step for interacting with a service on VertexAI is to create a dedicate client. In this case we need a `PipelineClient`.

```go
var pipelineClient *vai.PipelineClient
if pipelineClient, err = vai.NewPipelineClient(ctx, option.WithEndpoint(vaiEndpoint)); err != nil {
    return err
}
defer pipelineClient.Close()
```

After that, we need to formulate the correct request with the client. Once again, the documentation is very hard to find and the only place where I found all the parameters required to formulate a REST requies is the [Italian documentation](https://cloud.google.com/vertex-ai/docs/training/automl-api?hl=it#regression) (if I switch to English the documentation disappears!), where I can find:

```json
{
    "displayName": "TRAININGPIPELINE_DISPLAY_NAME",
    "trainingTaskDefinition": "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_tabular_1.0.0.yaml",
    "trainingTaskInputs": {
        "targetColumn": "TARGET_COLUMN",
        "weightColumn": "WEIGHT_COLUMN",
        "predictionType": "regression",
        "trainBudgetMilliNodeHours": TRAINING_BUDGET,
        "optimizationObjective": "OPTIMIZATION_OBJECTIVE",
        "transformations": [
            {"TRANSFORMATION_TYPE_1":  {"column_name" : "COLUMN_NAME_1"} },
            {"TRANSFORMATION_TYPE_2":  {"column_name" : "COLUMN_NAME_2"} },
            ...
    },
    "modelToUpload": {"displayName": "MODEL_DISPLAY_NAME"},
    "inputDataConfig": {
      "datasetId": "DATASET_ID",
    }
}
```

That's the body of the request to send to the API endpoint, if we are interacting with VertexAI using a pure REST client. With our `pipelineClient` we need to create the very same request, but according to the API requirements.

TODO

All the fields in UPPER CASE are the variable fields that we need to adapt to our problem. We must specify **all the columns** and **the transformations** to apply (mandatory).

```go
var modelDisplayName string = "sleep-efficiency-predictor"
var targetColumn string = "SleepEfficiency"

var trainingPipeline *vaipb.TrainingPipeline

// Create the Training Task Inputs
var trainingTaskInput structpb.Struct
// reference: https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1/schema/trainingjob.definition#automltablesinputs

// Create the transformations for all the columns (required)
var transformations string
tot := len(csvHeaders(allUserData)) - 1
for i, header := range csvHeaders(allUserData) {
    if header == targetColumn {
        // required because with auto the pipeline fails with error message:
        // "The values in target column SleepEfficiency have to be numeric for regression model."
        transformations += fmt.Sprintf(`{"numeric": {"column_name": "%s"}}`, header)
    } else {
        transformations += fmt.Sprintf(`{"auto": {"column_name": "%s"}}`, header)
    }
    if i < tot {
        transformations += ","
    }
}

if err = trainingTaskInput.UnmarshalJSON([]byte(
    fmt.Sprintf(
        `{
            "targetColumn": "%s",
            "predictionType": "regression",
            "trainBudgetMilliNodeHours": "1000",
            "optimizationObjective": "minimize-rmse",
            "transformations": [%s]
        }`, targetColumn, transformations))); err != nil {
    return err
}

```

TODO

```go
if trainingPipeline, err = pipelineClient.CreateTrainingPipeline(ctx, &vaipb.CreateTrainingPipelineRequest{
    // Required. The resource name of the Location to create the TrainingPipeline
    // in. Format: `projects/{project}/locations/{location}`
    Parent: fmt.Sprintf("projects/%s/locations/%s", _vaiProjectID, _vaiLocation),
    TrainingPipeline: &vaipb.TrainingPipeline{
        DisplayName:            modelDisplayName,
        TrainingTaskDefinition: "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_tables_1.0.0.yaml",
        InputDataConfig: &vaipb.InputDataConfig{
            DatasetId: datasetId,
        },
        TrainingTaskInputs: structpb.NewStructValue(&trainingTaskInput),
    },
}); err != nil {
    if s, ok := status.FromError(err); ok {
        log.Println(s.Message())
        for _, d := range s.Proto().Details {
            log.Println(d)
        }
    }
    return err
}
```

## Conclusion

For any feedback or comment, please use the Disqus form below - thanks!
