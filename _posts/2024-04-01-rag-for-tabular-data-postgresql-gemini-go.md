---
layout: post
title: "Building a RAG for tabular data with PostgreSQL & Gemini - in Go"
date: 2024-04-01 08:00:00
categories: golang vertexai
summary: ""
authors:
    - pgaleone
---

Large Language Models (LLMs) are well-suited for working with non-structured data. So far, their usage with structured data hasn't been explored in depth although structured data is everywhere relational databases are. Make a LLM able to interact with a relational database can be an interesting idea since it will unlock the possibility of letting an user "chatting with the data" and to let the LLM discover relationship inside the [data lake](https://cloud.google.com/learn/what-is-a-data-lake)

In this article, we'll explore a possible integration of Gemini (the multimodal large language model developed by Google) with PostgreSQL, and how to build a Retrieval-Augmented Generation (RAG) system to navigate in the structured data. Everything will be done in Go.

This is the fourth article of a series about the usage of Vertex AI in Go, and as such, it will share the same prerequisite presented in both of them: the services account creation, the environment variables, and so on. The prerequisite parts can be read in each of those articles.

- [Custom model training & deployment on Google Cloud using Vertex AI in Go](/golang/vertexai/2023/08/27/vertex-ai-custom-training-go-golang/)
- [AutoML pipeline for tabular data on VertexAI in Go](/golang/vertexai/2023/06/14/automl-pipeline-tabular-data-vertexai-go-golang/).
- [Using Gemini in a Go application: limits and details](/golang/vertexai/2024/02/26/gemini-go-limits-details/)

This article tries to implement the idea presented at the end of the last article. Given that converting all the user-data to a textual representation overflow the maximum context length of Gemini, we implement a RAG to overcome this limitation.

## RAG and Embeddings

Before going to the implementation in PostgreSQL, Go, and Gemini (through Vertex AI) we need to understand how a RAG system works. The analogy with a detective searching inside a  massive document archive for clues works well. In a RAG we have three components:

- **The Detective:** This is the generative model, like Gemini, that uses its knowledge to answer your questions or complete tasks.
- **The Archive:** This is your PostgreSQL database, holding all the tabular data (your documents).
- **The Informant:** This is the retriever, a special tool that understands both your questions and the data in the archive. It acts like your informant, scanning the archive to find the most relevant documents (clues) for the detective.

But how does the informant know which documents are relevant? Here's where **embeddings** come in. Embeddings are like condensed summaries of information. Imagine each document and your question are shrunk down into unique sets of numbers. The closer these numbers are in space, the more similar the meaning.

The informant uses an embedding technique to compare your question's embedding to all the document embeddings in the archive. It then retrieves the documents with the most similar embeddings, essentially pointing the detective in the right direction.

With these relevant documents at hand, the detective (generative model) can then analyze them and use its knowledge to answer your question or complete your request.

Given this structure we need:

- The detective: in our case it will be Gemini used through Vertex AI.
- The **embedding model**: a model able to create embeddings from a document.
- The archive: PostgreSQL. We need to **convert** the structured information from the database to a format valid for the embedding model. Then store the embeddings on the database.
- The informant: [pgvector](https://github.com/pgvector/pgvector). The open-source vector similarity search extension for PostgreSQL.

The **embedding model** is able to create embeddings only of *a document*. So, we need to find a way to convert the structured representation into a document as first step.

## From structured to Unstructured data

LLMs are very good at extracting information from textual data and to execute tasks described using text. Depending on our data, we may be lucky to have something easy "to narrate".

In the case described in this article we are going to use all the data related to sleep, physical activities, food, heart rate, number of steps (and other) gathered during a day for a single user. With these information it's quite easy to extract a regular description of an user day, section by section. Being the data so regular, we can try to make it fit in a **template**.

### The template: the daily report

We can define a template that summarized/highlights the important part we want to be able to retrieve while searching through our RAG. The template will be used by Gemini as part of its prompt in a chat session. In this chat session, we are going to ask the model to extract from the JSON data the information that we want to display in the report.

```markdown
### Date [LLM to write date]

## Activity

- Total Active Time: [LLM to fill from activities_summaries.active_minutes]
- Calories Burned: [LLM to fill from activities_summaries.calories_out]
- Steps Taken: [LLM to fill from activities_summaries.steps]
- Distance Traveled: [LLM to fill from activities_summaries.distance / activities_summary_distances.distance]
- List of activities: [LLM to iterate through activities_summary_activities and fill name, duration, calories burned]

### Active Minutes Breakdown

- Lightly Active Minutes: [LLM to fill from activities_summaries.lightly_active_minutes]
- Fairly Active Minutes: [LLM to fill from activities_summaries.fairly_active_minutes]
- Very Active Minutes: [LLM to fill from activities_summaries.very_active_minutes]

### Heart Rate Zones

- [LLM to iterate through activities_summary_heart_rate_zones and fill from heart_rate_zones (zone name, minutes)]

## Sleep

- Total Sleep Duration: [LLM to fill from sleep_logs.duration]
- Sleep Quality: [LLM to fill from sleep_logs.efficiency]
- Deep Sleep: [LLM to fill from sleep_stage_details where sleep_stage='deep sleep'] (minutes)
- Light Sleep: [LLM to fill from sleep_stage_details where sleep_stage='light sleep'] (minutes)
- REM Sleep: [LLM to fill from sleep_stage_details where sleep_stage='rem sleep'] (minutes)
- Time to Fall Asleep: [LLM to fill from sleep_logs.minutes_to_fall_asleep]

## Exercise Activities

- [LLM to iterate through daily_activity_summary_activities / minimal_activities and fill name, duration, calories burned (from activity_logs)]
...
```

Before digging inside the Go code, we have to design the structure for our data in the database.

The simplest solution is to create a table containing the textual reports that our LLM will generate together with its "compact representation" (the embeddings).

## The table creation

Being our data already stored on PostgreSQL it would be ideal to use the same database also for storing the embeddings and perform spatial query on them, and not introduce a new "vector database".

[pgvector](https://github.com/pgvector/pgvector) is the extension for postgreSQL that allows us to define a data type "vector" and it gives our operators and function to perform measures like cosine distance, l2 distance and many others.

Once installed and granted the superuser access to our database user, we can enable the extension and define the table for storing our data.

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    report_type TEXT NOT NULL,
    report TEXT NOT NULL,
    embedding VECTOR
);
```

After enabling the `vector` extension we can define the `embedding` field of type `vector`. There's no need to specify the maximum length of the vector since the extension supports dynamically shaped vectors.

The table is defined to store all the users report. In this article we are going to cover only the daily reports (so `start_date` will be equal to `end_date`) but to concept is easily generalizable to different kind of reports. This also the reason for the `report_type` field.

### The Go data structure

It's a good practice to map a table to a struct. Using [galeone/igor](https://github.com/galeone/igor) for interacting with PostgreSQL from Go this is also mandatory.

```go
import (
	"time"
	"github.com/pgvector/pgvector-go"
)

type Report struct {
	ID         int64 `igor:"primary_key"`
	UserID     int64
	StartDate  time.Time
	EndDate    time.Time
	ReportType string
	Report     string
	Embedding  pgvector.Vector
}

func (r *Report) TableName() string {
	return "reports"
}
```

That's all we are now ready to interact with Vertex AI to:

1. Go from structured to unstructured data, making Gemini filling the previously defined template
2. Generate the embeddings of the report
3. Let the user create a chat session with Gemini and create the embeddings of its prompt
4. Doing a spatial query for retrieving the (hopefully) relevant documents we have in the database
5. Pass these documents to Gemini as its search context.
6. Ask the model to answer the user question looking at the provided document

### Go: Generate reports and embedding

In Go, we can embed files directly inside the binary using the [embed](https://pkg.go.dev/embed) package. Embedding a template is the perfect use-case for this module:

```go
import "embed"

var (
	//go:embed templates/daily_report.md
	dailyReportTemplate string
)
```

We can design a data type `Reporter` whose goal is to to generate these reports. Using the (well-known after these 3 articles) pattern for the interaction with Vertex AI, we are going to create 2 different clients:

- The generative AI client for Gemini
- The prediction client for our embedding model

```go
type Reporter struct {
	user             *types.User
	predictionClient *vai.PredictionClient
	genaiClient      *genai.Client
	ctx              context.Context
}

// NewReporter creates a new Reporter
func NewReporter(user *types.User) (*Reporter, error) {
	ctx := context.Background()

	var predictionClient *vai.PredictionClient
	var err error
	if predictionClient, err = vai.NewPredictionClient(ctx, option.WithEndpoint(_vaiEndpoint)); err != nil {
		return nil, err
	}

	var genaiClient *genai.Client
	const region = "us-central1"
	if genaiClient, err = genai.NewClient(ctx, _vaiProjectID, region, option.WithCredentialsFile(_vaiServiceAccountKey)); err != nil {
		return nil, err
	}

	return &Reporter{user: user, predictionClient: predictionClient, genaiClient: genaiClient, ctx: ctx}, nil
}

// Close closes the client
func (r *Reporter) Close() {
	r.predictionClient.Close()
	r.genaiClient.Close()
}
```


## Conclusion

For any feedback or comments, please use the Disqus form below - Thanks!

<small>
This article has been possible thanks to the Google ML Developer Programs team that supported this work by providing Google Cloud Credit.
</small>
