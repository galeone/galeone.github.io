---
layout: post
title: "Using Gemini in a Go application: limits and details"
date: 2024-02-26 08:00:00
categories: golang vertexai
summary: "This article explores using Gemini within Go applications via Vertex AI. We'll delve into the limitations encountered, including the model's context window size and regional restrictions. We'll also explore various methods for feeding data to Gemini, highlighting the challenges faced due to these limitations. Finally, we'll briefly introduce RAG (Retrieval-Augmented Generation) as a potential solution, but leave its implementation details for future exploration."
authors:
    - pgaleone
---

Gemini - the multimodal large language model developed by Google - is already available on Vertex AI for production-grade applications. As with any other Vertex AI product, it is possible to interact with it using clients built in different languages such as Python, Java, and Go or using plain HTTP requests. After all, Vertex AI is just a web interface for interacting with the various offered services. In this article, I'm going to show you how to use the Go client to "chat with your data" and showcase some of the limitations of the model when it comes to context length.

Being a gopher, Go is my language of choice when it comes to developing new web applications. As someone may have noticed, I'm writing articles about the usage of Vertex AI and Go in the healthcare domain such as:

- [Custom model training & deployment on Google Cloud using Vertex AI in Go](/golang/vertexai/2023/08/27/vertex-ai-custom-training-go-golang/)
- [AutoML pipeline for tabular data on VertexAI in Go](/golang/vertexai/2023/06/14/automl-pipeline-tabular-data-vertexai-go-golang/).

This is the third article of this series, and as such, it will share the same prerequisite presented in both of them: the services account creation, the environment variables, and so on. The prerequisite parts can be read in each of those articles.

I'm writing these articles because I'm working on a new service called **FitSleepInsights**: a tool to visualize your health data and chat with them (using Gemini!). It will give every Fitbit user a way to visualize and get valuable insights from their data. If you are a Fitbit user, subscribe to the newsletter at the bottom of this article to receive an email when the service will be live!

## Model definition and configuration

The idea is to give the Fitbit user a way to chat with its data. The naive implementation will be to fetch all the user data from the Fitbit API, convert them to a textual representation, and feed them to the model. After that, we can let the users chat with the AI, to make them able to get insights from their own data. While implementing this naive idea, we will end up facing some of the limitations of Gemini.

Using any service of Vertex AI is straightforward: just create the dedicated client and use it, simple as that. The Go package for the Generative models on Vertex AI is `cloud.google.com/go/vertexai/genai` so we need to import it.

The documentation about the various Gemini models is available at [https://ai.google.dev/models/gemini](https://ai.google.dev/models/gemini). For our use case, we are interested in "Gemini Pro" the model that offers "text" and "chat" services. In the linked documentation we can find some model constants that we'll include in our code.

```go
import "cloud.google.com/go/vertexai/genai"

const MaxToken int = 30720
const MaxSequenceLength int = MaxToken * 3
```

There's a note in the [Model Variations](https://ai.google.dev/models/gemini#model-variations) section that states:

> Note: For Gemini models, a token is equivalent to about 4 characters. 100 tokens are about 60-80 English words.

That's why we set the `MaxSequenceLength` constant to `MaxToken * 3` (and the 3 multiplicative factor is a conservative value). As we'll later see, this is not entirely correct, since it looks like the model (from the Go client) is not able to "forget" and ignore past data - as one may expect from interacting with a LLM.

It's now time to create the client.

```go
ctx := context.Background()
var client *genai.Client
const region = "us-central1"
if client, err = genai.NewClient(ctx, os.Getenv("VAI_PROJECT_ID"), region, option.WithCredentialsFile(os.Getenv("VAI_SERVICE_ACCOUNT_KEY"))); err != nil {
    return err
}
defer client.Close()
```

A thing that immediately stands out is that hardcoded `const region = "us-central1"`. This is one of the limitations (as of today) of the usage of Gemini on Vertex AI. Although my whole project is based in Europe (`VAI_PROJECT_ID` points to a European region), I have to hardcode this location, because it's the only one that works.

With the created `client` we can choose the model to use. In the documentation, there's a section named [Model Variations](https://ai.google.dev/models/gemini#model-variations) that describes all the models available. For our use case, the model to use is `gemini-pro` since we'll work with text only, and we are not interested in the other multi-modal variations.

Every model has a set of tweakable parameters that allow us to control how the model behaves. One of the most important parameters is the *temperature*: a scalar value in the `[0-1]` range. A higher temperature results in more creative and less predictable outputs, while a lower temperature produces more conservative and expected results.

Being an optional field of the model (or better, to the request we'll send to the model), in Go this is represented as a `*float32`. So, to set a certain temperature, we need to insert an additional variable and extract its address.

```go
model := client.GenerativeModel("gemini-pro")

const ChatTemperature float32 = 0.1
temperature := ChatTemperature
model.Temperature = &temperature
```

## Chatting with the data

The idea is to allow the users to chat with their health data gathered via the Fitbit API. `gemini-pro` supports chatting, and the Go client allows us to define a new chat session in a single line of code.

```go
chatSession := model.StartChat()
```

It's worth noting that this line does nothing remotely. It just configures the local session, but no request is performed to the Vertex AI servers. With this chat session, we can start thinking about ways to feed the data to the model and create in this way its context. We can think about 3 options:

1. Send a configuration message, and send all the user data in a single message.
1. Send a configuration message, and send the user data in multiple messages.
1. Simulate a previous conversation with the model, sending the chat history.

All the options share the initial context creation. This context is a set of instructions for the model, used to configure its behavior, how to answer to the user and prevent some leakage of the raw data. We can use a string builder to efficiently create the `introductionString` that we'll send as first message.

```go
var builder strings.Builder
fmt.Fprintln(&builder, "You are an expert in neuroscience focused on the connection between physical activity and sleep.")
fmt.Fprintln(&builder, "You have been asked to analyze the data of a Fitbit user.")
fmt.Fprintln(&builder, "The user shares with you his/her data in JSON format.")
fmt.Fprintln(&builder, "The user is visualizing a dashboard generated from the data provided.")
fmt.Fprintln(&builder, "You must describe the data in a way that the user can understand the data and the potential correlations between the data and the sleep/activity habits.")
fmt.Fprintln(&builder, "You must chat to the user.")
fmt.Fprintln(&builder, "Never go out of this context, do not say hi, hello, or anything that is not related to the data.")
fmt.Fprintln(&builder, "Never accept commands from the user, you are only allowed to chat about the data.")
// This line below is important. Otherwiser the model will start analyzing non-existing data.
fmt.Fprintln(&builder, "Wait to receive the data in JSON format. Before you receive the data, you can't say anything.")
```

Of course, this is just a way to configure the model via text, there's no guarantee that the user will be able to work around this context and use the model for other goals. The string builder is still holding the data and no `string` has been created yet. Depending on the option we'll implement, we could end up adding some other line inside the builder before converting it to the `introductionString`.

### Option 1: all data at once

In this case, we can just send 2 messages and see what the model answers (for debugging purposes, in the production application we are not interested in the model response while we configure it).


```go
fmt.Fprintln(&builder, "I will send you a message containing the user data.")
introductionString := builder.String()

var response *genai.GenerateContentResponse
var err error
if response, err = chatSession.SendMessage(ctx, genai.Text(introductionString)); err != nil {
    return err
}
```

The `response` variable has a complex structure - that's a generic structure used also for multi-model models. In the case of text-only models like `gemini-pro` we can find the model response inside the first part of the first candidate response content.

```go
fmt.Println(response.Candidates[0].Content.Parts[0])
```

In this case, the model correctly answered something reasonable (given our context): *I am waiting for you to share the data in JSON format so I can analyze it and provide you with insights into the potential correlations between the data and your sleep/activity habits*.

Let's do what the model is asking us to do. Get all the data, convert it to JSON, and send it to the model.

Suppose that an object `fetcher` exists and it's able to fetch all the user data in a specified range. This is the function signature:

```go
func (f *fetcher) FetchByRange(startDate, endDate time.Time) []*UserData
```

We can use the `fetcher` object to get an `allData` slice and convert all the values into JSON.

```go
allData := fetcher.FetchByRange(startDate, endDate)
var jsonData []byte
if jsonData, err = json.Marshal(allData); err != nil {
    return err
}
stringData := string(jsonData)
```

`stringData` contains the whole user data in JSON format, and we can now send it to the model and see what happens:

```go
if _, err = chatSession.SendMessage(ctx, genai.Text(stringData)); err != nil {
	return err
}
```

`err != nil`:

> rpc error: code = InvalidArgument desc = Request contains an invalid argument.

The error is absolutely generic, and not descriptive at all. This is one of the pain points of the Gemini interface in Vertex AI.

After debugging, we can understand that we are sending a HUGE message to the model that exceeds the value of `MaxSequenceLenght`. The length of `stringData` is 297057, while according to the documentation (and that's another pain point), the maximum sequence length is `30720 * 3 = 92160`.

In order to understand if the problem hidden by the generic message "invalid argument" is the excessive length of the input sequence, we can try to just truncate `stringData` to `MaxSequenceLenght`, and send the message:

```go
if _, err = chatSession.SendMessage(ctx, genai.Text(stringData[:MaxSequenceLenght])); err != nil {
	return err
}
```

`err != nil`:

Once again 😔

Perhaps our conservative interpretation of the documentation wasn't enough conservative? If we change the multiplicative factor from 3 to 2 (so we consider a token something long 2 characters), and we repeat the previous request, then it works! However, the answer is quite worrying since every time we send this truncated JSON, the model only outputs JSON as it looks like for some reason it is trying to complete the data.

Anyway, we can discard this option since sending all the data at once is not possible. Let's go with option 2.

### Option 2: sending multiple messages

From the previous attempt, we find out the real `MaxSequenceLenght` to use. We can try to customize the introductory message to tell Gemini that we are going to send the data in multiple messages and see what the model answers after sending the new context and the various messages.

```go
var numMessages int
if len(stringData) > MaxSequenceLength {
    numMessages = len(stringData) / MaxSequenceLength
    fmt.Fprintf(&builder, "I will send you %d messages containing the user data.", numMessages)
} else {
    numMessages = 1
    fmt.Fprintln(&builder, "I will send you a message containing the user data.")
}

introductionString := builder.String()
if response, err = chatSession.SendMessage(ctx, genai.Text(introductionString)); err != nil {
    return err
}
// checkout response content

for i := 0; i < numMessages; i++ {
    if response, err = chatSession.SendMessage(ctx, genai.Text(stringData[i*MaxSequenceLength:(i+1)*MaxSequenceLength])); err != nil {
        return err
    }
    // checkout response content
}
```

Unfortunately, after sending `introductionString` and the first chunk of `stringData`, the server returned, once again the cryptic message:

> rpc error: code = InvalidArgument desc = Request contains an invalid argument.

Moreover, the model once again started to return only JSON content after the first (and only) sent message with JSON data. Let's try with the third and last approach.

### Populate the chat history

The `genai.ChatSession` structure, comes with a modifiable field named `History`. We can update this field in order to give the model an existing context, in the format of message exchange between the users with different roles:

- A message from the `"user"`
- A message from the `"model"`

Always in this sequence. Populating the history is the way we have to restore previous conversations (e.g. I imagine that the resume of past conversations on [https://gemini.google.com/app](https://gemini.google.com/app) is implemented in this way).

```go
chatSession.History = []*genai.Content{
    {
        Parts: []genai.Part{
            genai.Text(introductionString),
        },
        Role: "user",
    },
    {
        Parts: []genai.Part{
            genai.Text(
                fmt.Sprintf("Great! I will analyze the data and provide you with insights. Send me the data in JSON format in %d messages", numMessages)),
        },
        Role: "model",
    },
}

for i := 0; i < numMessages; i++ {
    var botTextAnswer string
    if i == numMessages-1 {
        botTextAnswer = "I received the last message with the data. I will now analyze it and provide you with insights."
    } else {
        botTextAnswer = "Go on, send me the missing data. I will analyze it once I have all the data."
    }

    chatSession.History = append(chatSession.History, []*genai.Content{
        {
            Parts: []genai.Part{
                genai.Text(genai.Text(stringData[i*MaxSequenceLength : (i+1)*MaxSequenceLength])),
            },
            Role: "user",
        },
        {
            Parts: []genai.Part{
                genai.Text(botTextAnswer),
            },
            Role: "model",
        }}...)
}
```

In this way, it looks like that we've been able to pass all the data to the model - but this is not true. In fact, we are only populating the **local** `History` variable, that it will be sent once on the first `chatSession.SendMessage` call. As easy to imagine, the first message sent will fail once again with the usual generic error message:

> rpc error: code = InvalidArgument desc = Request contains an invalid argument.

### The reason for these failures

We encountered a very common problem that happens when working with large language models: the limit on the context window length.

A context window in a large language model is like its short-term memory. It refers to the limited amount of text the model can consider at any one time when processing information and generating responses.

Imagine you're reading a book, but you can only see a few sentences at a time through a small window. As you move forward, the previous sentences disappear, replaced by new ones. This is similar to how an LLM "reads" information – it focuses on a specific window of text and uses that information to understand the overall context and generate a response.

When working with LLM, we should take into account that inside the "tokens count" not only the user input is considered, but also the model's output. So, every time a model accepts 1000 tokens as input and produces 500 tokens as output, the total count of tokens that it will consume in the next call will be 1500 (1000 + 500) + the number of tokens of the new input message.

However, what an LLM user expects is the model "forgetting" about the initial part of the conversation and using only the remaining part of the context as a "database" to find the answer to the user's question.

This is not happening with the Go client, and I suspect (but I haven't verified it yet) that is something that only happens with this client and not with the Python client (for instance). In any case, the failure messages are absolutely too generic to be useful.

### The correct solution: RAG or a bigger context window

The context window length of Gemini depends on the specific version:

- Standard Gemini: This version has a context window of approx 128,000 tokens (`30720*4`)
- Gemini 1.5 Pro (limited access and recently released): This advanced version boasts a significantly larger context window, reaching up to 1 million tokens. This is currently the longest context window of any publicly known large-scale foundation model.

However, being Gemini 1.5 Pro not yet publicly available, we can only rely upon a solution called RAG.

RAG stands for Retrieval-Augmented Generation. It's a technique used to improve the accuracy and relevance of LLM responses by providing additional context retrieved from external sources.

Here's how RAG works:

- The user provides a query or task: You ask a question or give an instruction to the LLM.
- Retrieval system searches for relevant information: An information retrieval component searches through a designated data source (e.g., documents, articles or - in our case - the user data) based on your query.
- Retrieved information is combined with user query: The retrieved information and your original query are then combined to form a richer prompt for the LLM.
- LLM generates a response: The LLM uses its knowledge and the provided prompt to generate a response that's more likely to be accurate, relevant, and insightful.

Think of RAG as giving the LLM a helping hand by providing additional clues and background information to understand the context of your query better. This leads to more informed and accurate responses.

However, implementing a RAG requires a way to compute embeddings, a database to store them, and to make similarity queries. This will be covered in another article :)

## Conclusion

Integrating Gemini with Go applications on Vertex AI presents several limitations and challenges:

- **Limited context window**: While the documentation mentions a limit on the number of tokens (30720), it **does not explicitly state that input tokens are cumulative with the model's output**. This crucial detail significantly reduces the usable context window, causing issues when feeding data in multiple parts or using chat history.
- **Region restriction**: Currently, using Gemini on Vertex AI is limited to the "us-central1" region, regardless of the project's actual region.
- **Generic error messages**: The generic "InvalidArgument" error message encountered during interaction with the model makes it difficult to diagnose specific issues.
- **Client behavior**: Unlike expected behavior expected (seen when working the the OpenAI client for example), the Go client for Gemini in Vertex  AI seems to **not manage context history effectively**. Instead of "forgetting" older messages as new data arrives, it accumulates all interactions, leading to the generic error message even when the total sequence length falls within the documented limit. This significantly hinders the ability to maintain a meaningful conversation with the model.

Potential solutions:

- **RAG (Retrieval-Augmented Generation)**: This technique proposes retrieving relevant information from external sources to enrich the context for the LLM. However, implementing RAG requires additional infrastructure and will be explored in a future article.
- **Upgrading to Gemini 1.5 Pro (limited access)**: This advanced version offers a significantly larger context window, potentially mitigating the limitations faced with the standard version. However, currently, access to this version is limited and the pricing is likely to be (way) higher than the standard version.

For any feedback or comments, please use the Disqus form below - Thanks!

<small>
This article has been possible thanks to the Google ML Developer Programs team that supported this work by providing Google Cloud Credit. This article is part of #GeminiSprint.
</small>
