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

- The detective: Gemini, used through Vertex AI.
- The **vectorizer** a model able to create embeddings from a document
- The archive: PostgreSQL. We need to **convert** the structured information from the database to a format valid for the vectorizer. Then store the embeddings on the database.
- The informant: [pgvector](https://github.com/pgvector/pgvector). The open-source vector similarity search extension for PostgreSQL

## Conclusion

For any feedback or comments, please use the Disqus form below - Thanks!

<small>
This article has been possible thanks to the Google ML Developer Programs team that supported this work by providing Google Cloud Credit.
</small>
