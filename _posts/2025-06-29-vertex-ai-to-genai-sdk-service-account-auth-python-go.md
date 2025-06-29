---
layout: post
title: "From Vertex AI SDK to Google Gen AI SDK: Service Account Authentication for Python and Go"
date: 2025-06-29 08:00:00
categories: cloud
summary: "Complete migration guide from Vertex AI SDK to Google Gen AI SDK for Python and Go developers. Covers service account authentication, OAuth2 scope limitations, and the critical implementation details missing from Google's official documentation."
authors:
    - pgaleone
---

On June 25th, 2025, Google sent an important announcement to all Vertex AI users: the [Google Gen AI SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) has become the new preferred method for accessing generative models on Vertex AI. While Google provided a [Vertex AI SDK migration guide](https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk) to help with the transition, there's a critical gap in the documentation—it doesn't adequately cover authentication using service accounts, which many production applications rely on.

## The Migration Guide Gap

Google's announcement included a helpful comparison table showing the SDK replacements for different programming languages. This article focuses on the two most commonly used languages in cloud applications: Python and Go.

| Language | Vertex AI SDKs | Replacement Google Gen AI SDKs |
|---|---|---|
| **Python** | [google-cloud-aiplatform](https://pypi.org/project/google-cloud-aiplatform/) | [google-genai](https://pypi.org/project/google-genai/) |
| **Go** | [cloud.google.com/go/vertexai/genai](https://pkg.go.dev/cloud.google.com/go/vertexai/genai) | [google.golang.org/genai](https://pkg.go.dev/google.golang.org/genai) |

The [official migration guide](https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk) uses a straightforward before-and-after approach, showing you exactly how to update your existing code. However, there's a significant limitation: it primarily focuses on [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials).

**Here's where many developers get stuck:** If your application uses a JSON [Service Account](https://cloud.google.com/iam/docs/service-account-overview) for authentication—a common pattern in production environments—the official guide leaves you without clear direction. Following the migration steps blindly will result in authentication failures that can be frustrating to debug.

The following sections provide the missing pieces to successfully migrate your service account-based authentication in both Python and Go.

## Python

For Python applications, migrating service account authentication requires creating a `Credentials` object using the `google.oauth2.service_account` package. The key insight here is that OAuth2 scopes are critical—without the correct scopes, Google's servers will reject your requests with authentication errors.

```python
    from google import genai
    from google.genai import types
    from google.oauth2.service_account import Credentials

    # Define the OAuth2 scopes required for accessing Google Cloud Platform
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    
    # Create credentials from the service account JSON file with the required scopes
    credentials = Credentials.from_service_account_file(
        os.getenv("SERVICE_ACCOUNT_FILE_PATH"), scopes=scopes
    )
    
    # Create the Gen AI Client specifying the location by env var
    # and use the project ID from environment variables
    client = genai.Client(
        vertexai=True,
        project=os.getenv("GCLOUD_PROJECT_ID"),
        location=os.getenv("GCLOUD_LOCATION"),
        credentials=credentials,
    )
```

This approach leverages Python's convenient `from_service_account_file()` method, which automatically handles the JSON parsing and credential creation. The environment variables (`SERVICE_ACCOUNT_FILE_PATH`, `GCLOUD_PROJECT_ID`, `GCLOUD_LOCATION`) should point to your service account JSON file path, Google Cloud project ID, and preferred region respectively.

## Go

Go requires a more hands-on approach since there's no equivalent to Python's convenient `from_service_account_file()` function. Instead, we need to manually parse the service account JSON and construct a credential object with its own OAuth2 token provider.


```go

import (
    "fmt"
    "json"
    "os"

    "cloud.google.com/go/auth"
    "google.golang.org/genai"
)

// ...

// Get the service account content
key, err := os.ReadFile(os.getenv("SERVICE_ACCOUNT_FILE_PATH"))
if err != nil {
    return fmt.Errorf("failed to read service account key: %s", err)
    
}

var serviceAccount struct {
    ClientEmail string `json:"client_email"`
    PrivateKey  string `json:"private_key"`
    TokenURI    string `json:"token_uri"`
    ProjectID   string `json:"project_id"`
}
if err := json.Unmarshal(key, &serviceAccount); err != nil {
    return fmt.Errorf("invalid service-account JSON: %s", err)
}

// Create the 2-legged OAuth token provider
tp, err := auth.New2LOTokenProvider(&auth.Options2LO{
    Email:      serviceAccount.ClientEmail,
    PrivateKey: []byte(serviceAccount.PrivateKey),
    TokenURL:   serviceAccount.TokenURI,
    Scopes:     []string{"https://www.googleapis.com/auth/cloud-platform"},
})
if err != nil {
    return fmt.Errorf("failed to create 2LO token provider: %s", err)
}

// Create the credentials using the token provider
credentials := auth.NewCredentials(&auth.CredentialsOptions{
    TokenProvider: tp,
    JSON:          key,
})

// Create the Gen AI Client specifying the location by env var
// and use the project ID specified in the service account
ctx := context.Background()
genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
    Project:     serviceAccount.ProjectID,
    Location:    os.getenv("GCLOUD_LOCATION"),
    Backend:     genai.BackendVertexAI,
    Credentials: credentials,
})
if err != nil {
    return fmt.Errorf("failed to create genai client: %s", err)
}
```

## Understanding OAuth2 Scopes for Vertex AI

Throughout both implementations, we used the `https://www.googleapis.com/auth/cloud-platform` scope because it provides comprehensive access to Vertex AI endpoints and other Google Cloud services. This broad scope essentially allows our applications to perform any operation that the service account is authorized to do within the Google Cloud project.

For developers who want to understand all available scopes across Google's APIs, the complete reference is available in the [OAuth 2.0 Scopes for Google APIs documentation](https://developers.google.com/identity/protocols/oauth2/scopes#aiplatform). However, when it comes to Vertex AI specifically, Google's scope options are surprisingly limited:

| Scope | Description |
|-------|-------------|
| `https://www.googleapis.com/auth/cloud-platform` | Full access: See, edit, configure, and delete your Google Cloud data |
| `https://www.googleapis.com/auth/cloud-platform.read-only` | Read-only access: View your data across Google Cloud services |

**The Fine-Grained Access Problem**

This binary choice between "full access" and "read-only" represents a significant limitation in Google's OAuth2 implementation for Vertex AI. In production environments, you might want more granular permissions—for example, allowing an application to create and run inference requests while preventing it from deleting models or modifying training jobs. Unfortunately, Google doesn't provide such fine-grained scopes for Vertex AI, forcing developers to choose between overly broad permissions or overly restrictive read-only access.

This lack of granular control is a notable pain point when implementing the principle of least privilege in cloud applications. I wasn't expecting it from Google.

## Key Takeaways

Migrating from Vertex AI SDK to Google Gen AI SDK with service account authentication requires careful attention to OAuth2 scope configuration and credential management. While the official migration guide covers the basic API changes, the authentication patterns shown here are essential for production applications that can't rely on Application Default Credentials.

- Always specify the `https://www.googleapis.com/auth/cloud-platform` scope for OAuth2 authentication, since it's pretty much the only scope we have
- Python developers can leverage the convenient `from_service_account_file()` method
- Go developers need to manually construct the OAuth2 token provider but gain more control over the authentication flow
