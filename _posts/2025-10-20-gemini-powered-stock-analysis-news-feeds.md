---
layout: post
title: "Gemini-Powered Stock Analysis: Parsing Financial News for Automated Trading Decisions"
date: 2025-10-20 02:00:00
categories: golang vertexai trading
summary: "How I built an automated stock analysis system that leverages Gemini to parse Italian financial news feeds, providing real-time trading recommendations. This article explores the architecture, challenges, and implementation details of integrating AI-powered news analysis into a Go-based trading system."
authors:
    - pgaleone
---

Over the past weeks I built a small Go service that reads Italian finance RSS feeds, extracts the article body, and asks Gemini for a buy/sell/hold view for the tickers it finds. Below I show the pieces that make it work and the trade‑offs I hit along the way.

This runs inside a broader trading setup that talks to a broker API. I'm working on software to automate trading on a well‑known Italian broker; if you're interested, please leave a comment below or get in touch via the contact form.

## The Challenge: Processing Italian Financial News at Scale

Financial news moves markets, but manually processing hundreds of articles daily is impractical for algorithmic trading. The challenge becomes even more complex when dealing with Italian financial news sources, which often use specific terminology and reference stocks with various ticker formats (e.g., "STLAM" for Stellantis, "LDO.MI" for Leonardo).

My solution needed to:
1. **Parse multiple RSS feeds** from Italian financial sources
2. **Extract full article content** from various HTML formats
3. **Analyze articles using AI** to identify mentioned stocks and trading signals
4. **Route recommendations** to appropriate trading strategies
5. **Handle multilingual content** (Italian and English)

## Architecture Overview

The system consists of several key components working together:

```go
// Core components of the news analysis system
type Stream struct {
    feeds             []string
    client            *http.Client
    newsChan          chan *NewsItem
    stopChan          chan struct{}
    fetchedItems      map[string]bool
    vertexAiModelName string
    analysisEnabled   bool
    genaiClient       *genai.Client
}

type AIAnalysisService struct {
    engine    *engine.Engine
    longChan  chan *AIRecommendation
    shortChan chan *AIRecommendation
    stopChan  chan struct{}
}
```

The architecture follows a producer-consumer pattern where the news stream fetches and analyzes articles, while the AI analysis service routes recommendations to trading strategies.

## Italian News Sources and Content Extraction

The system monitors several Italian financial news sources:

```go
func NewStream() *Stream {
    // Initialize NewsStream with default RSS feeds
    defaultNewsFeeds := []string{
        "https://news.teleborsa.it/NewsFeed.ashx",
        "https://investire.biz/feed/analisi/azioni",
    }
    // ... initialization code
}
```

### Handling Different HTML Formats

One of the most challenging aspects was extracting clean article content from different Italian news websites. Each source has its own HTML structure:

```go
// ExtractArticleFromHTML supports both Teleborsa and Investire.biz formats
func ExtractArticleFromHTML(htmlContent string) string {
    // First, try to extract from Investire.biz format
    if content := extractInvestireBizArticle(htmlContent); content != "" {
        return content
    }

    // If not found, try Teleborsa format
    if content := extractTeleborsaArticle(htmlContent); content != "" {
        return content
    }

    return ""
}

func extractTeleborsaArticle(htmlContent string) string {
    // Find the start of the article content
    startMarker := "(Teleborsa) - "
    startIndex := strings.Index(htmlContent, startMarker)
    if startIndex == -1 {
        return ""
    }

    // Extract and clean the content
    articleHTML := htmlContent[startIndex:endIndex]
    cleanText := RemoveHTMLTags(articleHTML)
    cleanText = html.UnescapeString(cleanText)
    cleanText = NormalizeWhitespace(cleanText)

    return cleanText
}
```

The extraction process handles the specific formatting patterns used by Italian financial news sites. For example, Teleborsa articles always start with "(Teleborsa) - " followed by the actual content, while Investire.biz uses a specific `<div id="articleText">` container.

Note on robustness: parsing articles by matching specific HTML patterns is inherently brittle—publishers can change their markup at any time, breaking custom extractors. An alternative is to feed the entire page HTML to the LLM and let it identify the relevant content. That approach is often robust but more expensive in tokens due to boilerplate HTML. Extracting the article body upfront helps reduce token usage by avoiding irrelevant markup while accepting the maintenance cost if page structures change.

## Gemini Integration for Stock Analysis

The heart of the system is the Gemini-powered analysis that processes the extracted article content and generates trading recommendations.

Authentication and client setup are covered in a separate article: [From Vertex AI SDK to Google Gen AI SDK: Service Account Authentication for Python and Go](/cloud/2025/06/29/vertex-ai-to-genai-sdk-service-account-auth-python-go/).

### The Analysis Prompt

The key to effective AI-powered stock analysis lies in crafting a precise prompt that handles the multilingual nature of Italian financial news:

```go
func (s *Stream) analyzeArticleWithGemini(articleText string) (*NewsAnalysis, error) {
    prompt := fmt.Sprintf(`Analyze the following news article and provide a JSON response with an array of stock analyses.
The article can be in Italian or English..
For each stock mentioned, include the ticker symbol (e.g., for "Stellantis" it should be "STLAM"), a suggested action (buy, sell, or hold), and a brief reason for the suggestion in English.
It's not mandatory to include all the stocks mentioned in the article, only the ones that are relevant to the article.
If there are no stocks mentioned in the article, return an empty array.
The suggested action should be based on the article content and the stock's performance. The suggested action will be used for day trading.

Article:
%s

IMPORTANT: 
- Your response must contain ONLY valid JSON, no explanatory text before or after
- Do not include any markdown formatting or code blocks
- Return exactly this JSON structure and nothing else:

{
  "items": [
    {
      "ticker": "STOCK_TICKER",
      "action": "buy|sell|hold", 
      "reason": "Brief explanation in English."
    }
  ]
}`, articleText)

    config := &genai.GenerateContentConfig{
        Temperature: genai.Ptr[float32](0.1),
    }

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    resp, err := s.genaiClient.Models.GenerateContent(ctx, s.vertexAiModelName, genai.Text(prompt), config)
    // ... error handling and JSON parsing
}
```

The prompt is carefully designed to:
- Handle both Italian and English content
- Request specific ticker formats used in Italian markets
- Provide structured JSON output for easy parsing
- Focus on actionable trading recommendations

## Real-World Example: Analyzing Italian Financial News

Let's look at how the system processes a typical Italian financial news article:

```go
// Sample Italian article from Teleborsa
articleText := `(Teleborsa) - Seduta in ribasso per Stellantis, che mostra un calo dell'1,26%. A pesare sulle azioni è la notizia di un richiamo di oltre un milione di veicoli negli Stati Uniti per un difetto alla telecamera posteriore. Brilla invece Leonardo, che avanza del 2,5% grazie a nuove commesse nel settore della difesa.`
```

When processed by Gemini, this article generates the following analysis:

```json
{
  "items": [
    {
      "ticker": "STLAM",
      "action": "sell",
      "reason": "Stock declining 1.26% due to recall of over 1 million vehicles in US for rear camera defect."
    },
    {
      "ticker": "LDO.MI",
      "action": "buy", 
      "reason": "Stock rising 2.5% on new defense sector contracts."
    }
  ]
}
```

## Routing Recommendations to Trading Strategies

Once Gemini analyzes the articles, the system routes recommendations to appropriate trading channels:

```go
func (s *AIAnalysisService) Start() error {
    go func() {
        for {
            select {
            case newsItem := <-newsChan:
                if newsItem.Analysis == nil || len(newsItem.Analysis.Items) == 0 {
                    continue
                }

                for _, analysis := range newsItem.Analysis.Items {
                    // Generate ticker variations for different markets
                    allTickers := []string{analysis.Ticker}
                    
                    // Handle Italian stocks (.MI suffix)
                    if strings.Contains(analysis.Ticker, ".MI") {
                        miRemovedTicker := strings.Replace(analysis.Ticker, ".MI", "", 1)
                        allTickers = append(allTickers, miRemovedTicker)
                    }

                    // Lookup actual tradeable instruments
                    titles, err := s.engine.ListTitles(allTickers)
                    if err != nil {
                        continue
                    }

                    // Route to appropriate channels
                    for _, titleList := range titles {
                        for _, title := range titleList {
                            recommendation := &AIRecommendation{
                                Title:    title,
                                Analysis: analysis,
                                Source:   newsItem.Source,
                            }

                            switch analysis.Action {
                            case news.BuyAction:
                                s.longChan <- recommendation
                            case news.SellAction:
                                s.shortChan <- recommendation
                            }
                        }
                    }
                }
            }
        }
    }()
}
```

The system intelligently handles ticker symbol variations common in Italian markets, where stocks might be referenced as "STLAM", "STLA.MI", or other formats depending on the exchange.

## Performance and Reliability Considerations

### Concurrent Processing

The system uses Go's concurrency features to handle multiple news sources simultaneously:

```go
func (s *Stream) Start() {
    s.wg.Add(1)
    go func() {
        defer s.wg.Done()
        ticker := time.NewTicker(s.fetchInterval)
        defer ticker.Stop()

        for {
            select {
            case <-ticker.C:
                for _, feed := range s.feeds {
                    go s.fetchFeedWithRetry(feed)
                }
            case <-s.stopChan:
                return
            }
        }
    }()
}
```

### Error Handling and Retries

Given the critical nature of financial data, the system implements robust retry mechanisms:

```go
func (s *Stream) fetchFeedWithRetry(feedURL string) error {
    var lastErr error
    delay := s.retryConfig.InitialDelay

    for attempt := 0; attempt <= s.retryConfig.MaxRetries; attempt++ {
        if attempt > 0 {
            time.Sleep(delay)
            delay = time.Duration(float64(delay) * s.retryConfig.BackoffFactor)
            if delay > s.retryConfig.MaxDelay {
                delay = s.retryConfig.MaxDelay
            }
        }

        if err := s.fetchFeed(feedURL); err != nil {
            lastErr = err
            continue
        }
        return nil
    }
    
    return fmt.Errorf("failed to fetch feed after %d attempts: %w", 
        s.retryConfig.MaxRetries+1, lastErr)
}
```

## Lessons Learned and Challenges

### Prompt Engineering for Financial Analysis

Crafting effective prompts for financial analysis required several iterations. Key learnings:

1. **Be explicit about output format**: Gemini can be verbose, so explicitly requesting JSON-only responses is crucial
2. **Handle multilingual content**: Italian financial news often mixes Italian and English terms
3. **Specify ticker formats**: Different markets use different conventions (STLAM vs STLA.MI vs STLA)
4. **Set appropriate temperature**: Low temperature (0.1) provides more consistent, factual analysis

### Managing API Costs

Vertex AI costs can add up quickly with frequent news analysis. Optimization strategies:

1. **Content filtering**: Only analyze articles that pass initial relevance filters
2. **Batch processing**: Aggregate similar articles when possible
3. **Caching**: Avoid re-analyzing identical content
4. **Timeout management**: Set reasonable timeouts to prevent hanging requests

## Integration with Trading Strategies

The AI analysis service integrates seamlessly with various trading strategies:

```go
// Example: Trend following strategy with AI recommendations
longStocks, err := stockSelector.SelectStocksLong(strategy.SelectFromAll)
if err != nil {
    log.Error("Error starting SelectStocks: %s", err)
    return
}

// Process AI recommendations
go func() {
    for recommendation := range aiService.GetLongChannel() {
        // Create trend following strategy for recommended stock
        trendStrategy := strategy.NewTrendFollowing(
            tradingEngine, 
            recommendation.Title,
            strategy.LongDirection,
        )
        
        if err := trendStrategy.Start(); err != nil {
            log.Error("Failed to start trend strategy: %s", err)
            continue
        }
        
        log.Info("Started AI-recommended long strategy for %s: %s", 
            recommendation.Title.GetPriceCode(), 
            recommendation.Analysis.Reason)
    }
}()
```

## Future Enhancements

Several improvements are planned for the system:

1. **Sentiment scoring**: Add numerical sentiment scores alongside buy/sell/hold recommendations
2. **Multi-model analysis**: Compare recommendations from different AI models
3. **Historical backtesting**: Evaluate AI recommendation accuracy over time
4. **Real-time alerts**: Push critical news analysis to mobile devices
5. **Portfolio integration**: Consider existing positions when generating recommendations

## Conclusion

Key takeaways:

- Extract only the article body to save tokens, but expect HTML to change.
- Keep the prompt strict (JSON-only, low temperature).
- Normalize tickers for local markets before routing to strategies.
- Concurrency and retries matter more than clever code.

I’ll keep hardening the extractors and add backtesting and alerts. If you want to try it, share feedback, or chat about the broker integration, leave a comment or reach out via the contact form.
