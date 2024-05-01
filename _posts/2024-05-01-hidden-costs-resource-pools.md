---
layout: post
title: "The (Hidden?) Costs of Vertex AI Resource Pools: A Cautionary Tale"
date: 2024-05-01 08:00:00
categories: golang vertexai
summary: "In the article \"Custom model training & deployment on Google Cloud using Vertex AI in Go\" we explored how to leverage Go to create a resource pool and train a machine learning model using Vertex AI's allocated resources. While this approach offers flexibility, there's a crucial aspect to consider: the cost implications of resource pools. This article details my experience with a sudden price increase in Vertex AI and the hidden culprit – a seemingly innocuous resource pool."
authors:
    - pgaleone
---

In the article [Custom model training & deployment on Google Cloud using Vertex AI in Go](/golang/vertexai/2023/08/27/vertex-ai-custom-training-go-golang/) we explored how to leverage Go to create a resource pool and train a machine learning model using Vertex AI's allocated resources. While this approach offers flexibility, there's a crucial aspect to consider: the cost implications of resource pools.

This article details my experience with a sudden price increase in Vertex AI and the hidden culprit – a seemingly innocuous resource pool.

## A Unexplained Cost Surge

<div markdown="1" class="blog-image-container">
![Vertex AI price report for April and March](/images/vertex-ai/price-spike.png){:class="blog-image"}
</div>

The green color in the graph represents Vertex AI expenditure. The image above shows that something happened around the 8 of March. During that period I was only working on the dashboard of [fitsleepinsights.app](https://fitsleepinsights.app/), so definitely nothing changed in the infrastructure, the code, or the load of the Vertex AI services I was using. Anyway, from the graph it's clearly visible an increase of more than 500% in the cost of Vertex AI. I was spending literally less than 1 €/day until the 8 March, and from that day onward I started spending more than 5 €/day.

Reaching out to Google Cloud support proved unhelpful. They couldn't pinpoint the reason behind the cost increase. Left to my own devices, I embarked on a multi-day investigation through various Vertex AI dashboards.

## The Culprit Revealed (with a Glitch)

The first hint came from the pricing dashboard, in the cost table breakdown.

<div markdown="1" class="blog-image-container">
![Cost table for March](/images/vertex-ai/cost-table.png){:class="blog-image"}
</div>

We can see from it that the Vertex AI costs come mainly from 2 Online/Batch prediction resources:

- The **Instance Core** running in EMEA for AI platform
- The **Instance RAM** running in EMEA for AI platform

All the other costs are negligible. Unfortunately, there's no detailed information on where these prices are coming from. It's only clear that something in the online/batch prediction is using core (CPUs) and RAM.

In [Custom model training & deployment on Google Cloud using Vertex AI in Go](/golang/vertexai/2023/08/27/vertex-ai-custom-training-go-golang/) we created a resource pool for training our models. After digging inside the online/batch dashboards of Vertex AI, I stumbled upon the resource pool dashboard – the potential culprit.

Unfortunately, displaying the dashboard resulted in an error 😒

<div markdown="1" class="blog-image-container">
![Vertex AI resource pool error](/images/vertex-ai/resource-pool-unavailable.png){:class="blog-image"}
</div>

So, no dashboard is available. Lucky me? To be sure that there wasn't something flooding the endpoints deployed (but the logs were clear) I, anyway, deleted all the models and all the endpoints deployed. I did this around mid-March. As it is visible from the graph at the beginning of the article, nothing changed. Therefore the resource pool is the major suspect.

## Taking Back Control: Deleting the Resource Pool (Go Code Included)

The resource pool, likely active for months, might have been unknowingly incurring charges. After all, I couldn't find [any documentation](https://cloud.google.com/vertex-ai/pricing) regarding a free tier for resource pools.

To curb the runaway costs and return to normalcy (ideally, zero cost since no resources were actively used), I had to delete the resource pool. Here's the Go code that did the trick:

```go
ctx := context.Background()
var resourcePoolClient *vai.DeploymentResourcePoolClient
if resourcePoolClient, err = vai.NewDeploymentResourcePoolClient(ctx, option.WithEndpoint(_vaiEndpoint)); err != nil {
  log.Error("error creating resource pool client: ", err)
  return err
}
defer resourcePoolClient.Close()

deploymentResourcePoolId := "resource-pool"
var deploymentResourcePool *vaipb.DeploymentResourcePool = nil

iter := resourcePoolClient.ListDeploymentResourcePools(ctx, &vaipb.ListDeploymentResourcePoolsRequest{
  Parent: fmt.Sprintf("projects/%s/locations/%s", _vaiProjectID, _vaiLocation),
})

var item *vaipb.DeploymentResourcePool
for item, _ = iter.Next(); err == nil; item, err = iter.Next() {
  if strings.Contains(item.GetName(), deploymentResourcePoolId) {
    deploymentResourcePool = item
    log.Print("Found deployment resource pool: ", deploymentResourcePool.GetName())

    // Delete the resource pool
    var deleteResourcePoolOp *vai.DeleteDeploymentResourcePoolOperation
    if deleteResourcePoolOp, err = resourcePoolClient.DeleteDeploymentResourcePool(ctx, &vaipb.DeleteDeploymentResourcePoolRequest{
      Name: deploymentResourcePool.GetName(),
    }); err != nil {
      log.Error("Error deleting deployment resource pool: ", err)
      return err
    }
    if err = deleteResourcePoolOp.Wait(ctx); err != nil {
      log.Error("Error waiting for deployment resource pool deletion: ", err)
      return err
    }
    log.Print("Deleted deployment resource pool: ", deploymentResourcePool.GetName())
    break
  }
}
```

For the sake of completeness, I report below the code I used in the past to create the resource pool. This way, I hope the article becomes a self-contained resource about the management of the resource pool lifetime using Vertex AI in Go.

<details>
    <summary>How to create a resource pool - click me to see the code</summary>

{% highlight go %}
var resourcePoolClient *vai.DeploymentResourcePoolClient
if resourcePoolClient, err = vai.NewDeploymentResourcePoolClient(ctx, option.WithEndpoint(vaiEndpoint)); err != nil {
    return err
}
defer resourcePoolClient.Close()

deploymentResourcePoolId := "resource-pool"
var deploymentResourcePool *vaipb.DeploymentResourcePool = nil
iter := resourcePoolClient.ListDeploymentResourcePools(ctx, &vaipb.ListDeploymentResourcePoolsRequest{
    Parent: fmt.Sprintf("projects/%s/locations/%s", os.Getenv("VAI_PROJECT_ID"), os.Getenv("VAI_LOCATION")),
})
var item *vaipb.DeploymentResourcePool
for item, _ = iter.Next(); err == nil; item, err = iter.Next() {
    fmt.Println(item.GetName())
    if strings.Contains(item.GetName(), deploymentResourcePoolId) {
        deploymentResourcePool = item
        fmt.Printf("Found deployment resource pool %s\n", deploymentResourcePool.GetName())
        break
    }
}

if deploymentResourcePool == nil {
    fmt.Println("Creating a new deployment resource pool")
    // Create a deployment resource pool: FOR SHARED RESOURCES ONLY
    var createDeploymentResourcePoolOp *vai.CreateDeploymentResourcePoolOperation
    if createDeploymentResourcePoolOp, err = resourcePoolClient.CreateDeploymentResourcePool(ctx, &vaipb.CreateDeploymentResourcePoolRequest{
        Parent:                   fmt.Sprintf("projects/%s/locations/%s", os.Getenv("VAI_PROJECT_ID"), os.Getenv("VAI_LOCATION")),
        DeploymentResourcePoolId: deploymentResourcePoolId,
        DeploymentResourcePool: &vaipb.DeploymentResourcePool{
            DedicatedResources: &vaipb.DedicatedResources{
                MachineSpec: &vaipb.MachineSpec{
                    MachineType:      "n1-standard-4",
                    AcceleratorCount: 0,
                },
                MinReplicaCount: 1,
                MaxReplicaCount: 1,
            },
        },
    }); err != nil {
        return err
    }

    if deploymentResourcePool, err = createDeploymentResourcePoolOp.Wait(ctx); err != nil {
        return err
    }
    fmt.Println(deploymentResourcePool.GetName())
}
{% endhighlight %}

</details>


## The Sweet Relief of Reduced Costs

Following the deletion of the resource pool, my Vertex AI costs (green) plummeted back to almost zero (only some cents are still there for the Gemini requests). This confirmed my suspicion – the resource pool was indeed the culprit behind the cost increase.

<div markdown="1" class="blog-image-container">
![Vertex AI price report for April and March - highlighted the day of resource pool deletion](/images/vertex-ai/price-spike2.png){:class="blog-image"}
</div>

## Conclusions

Resource pools in Vertex AI offer a convenient way to manage shared compute resources for training and deploying models. However, it's crucial to understand their cost implications. Here are some key takeaways:

- The documentation of the resource pool doesn't mention a free period. This period apparently exists since Vertex AI started to charge for this service out of nowhere.
- Resource pools incur charges even when not actively used for training or prediction.
- Carefully monitor your Vertex AI resource pool usage to avoid unexpected cost increases.
- Consider deleting unused resource pools to optimize your spending.
- The resource pool dashboard is broken (or at least it was and still is on my account).

By being mindful of these hidden costs, you can leverage Vertex AI resource pools effectively without jeopardizing your budget.

For any feedback or comments, please use the Disqus form below - Thanks!
