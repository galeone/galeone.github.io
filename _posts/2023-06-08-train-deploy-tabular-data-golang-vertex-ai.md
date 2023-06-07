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

The data comes in CSV format. This is a (simplified, only a few attributes are shown) view.

|CaloriesSum|StepsSum|AverageHeartRate|ActivityCalories|Calories|Distance|PeakMinBPM|**SleepEfficiency**|
|-----------|--------|----------------|----------------|--------|--------|----------|---------------|
|569        |1015    |127.00          |1889.00         |3283.00 |6.09    |169       |43             |



## Conclusion

For any feedback or comment, please use the Disqus form below - thanks!
