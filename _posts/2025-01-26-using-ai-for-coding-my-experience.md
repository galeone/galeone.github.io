---
layout: post
title: "Using AI for Coding: My Journey with Cline and Large Language Models"
date: 2025-01-26 08:00:00
categories: ai coding
summary: "How I leveraged AI tools like Cline to enhance the UI/UX of a website and streamline backend tasks. From redesigning pages and translating content to navigating the benefits and challenges of AI-assisted development, this blog post highlights the potential of using large language models to boost productivity while sharing key lessons learned."
authors:
    - pgaleone
---

In recent months, I embarked on a journey to transform the UI/UX of a side project—[bot.eofferte.eu](https://bot.eofferte.eu), a SaaS platform that automates Amazon affiliate marketing on Telegram and streamlines the Amazon Associates onboarding process.

The project's architecture is straightforward: a Go backend powered by the [labstack/echo](https://github.com/labstack/echo) framework, with UI rendering handled by Go's standard `html/template` package. To accelerate development and improve the overall user experience, I experimented with [Cline](https://github.com/cline/cline) through its VSCode plugin as my primary AI coding assistant. Here's a detailed breakdown of my experience.

### Experiments with Frontend Development

As someone who primarily focuses on backend development, UI/UX has always been a challenge. My limited knowledge of modern web frameworks and general aversion to CSS made frontend work particularly daunting. However, leveraging AI tools transformed this weakness into an opportunity for rapid improvement.

The impact was immediate and substantial. I tasked the LLMs with redesigning every page of the website:
- The landing page underwent a complete transformation
- The management interface (where users configure their services) received significant upgrades
- The overall design evolved from basic to professional-grade

Beyond visual improvements, the LLMs proved invaluable for generating and refining essential content like privacy policies, terms of service, and other compliance documentation.

I experimented with two leading models:

1. **Claude Sonnet 3.5:**
   - **Performance:** Exceptional response speed
   - **Accuracy:** Deep understanding of web technologies (HTML, CSS, JavaScript)
   - **Effectiveness:** Made intelligent framework suggestions (Font Awesome, Bootstrap) that enhanced both aesthetics and functionality
   - **Limitations:** Context window restrictions often interrupted complex tasks

2. **Gemini:**
   - **Performance:** Slower processing speed compared to Sonnet
   - **Advantage:** Larger context window enabled handling more comprehensive instructions
   - **Versatility:** Better suited for tasks requiring extensive context analysis

The AI's ability to suggest appropriate frameworks and create cohesive designs proved transformative, especially for someone with limited frontend expertise.

### Prompt Engineering for Success

Working with Cline proved intuitive thanks to its ability to analyze open files and understand repository context. A prime example was the redesign of our bot management interface (`bot.html`).

The original design required users to complete an extensive form in a single session. To improve user experience, I decided to implement a step-by-step wizard using [Enchanter.js](https://github.com/brunnopleffken/enchanter). The integration process highlighted the importance of precise prompt engineering:

```
analyze bot.html - it's a Go (golang) html template.

bot.html contains both html template code and JavaScript. Both are mixed with Go template syntax.

You need to rewrite bot.html using static/enchanter.js in order to convert the form in bot.html to a guided wizard.

Do not touch any JavaScript already present in bot.html and ignore every JavaScript error.

Your <form> tag should wrap the .nav and .tab-content elements. The footer of the form must contain "Back", "Next" and "Finish" buttons with the data-enchanter attributes.
```

This prompt succeeded because it:
1. **Established Context:** Clearly identified the technology stack and file structure
2. **Defined Scope:** Provided specific implementation requirements
3. **Set Boundaries:** Prevented unnecessary JavaScript modifications
4. **Specified Requirements:** Detailed the exact structure needed for the wizard implementation

### Backend Development Insights

The backend experience revealed a crucial distinction in AI-assisted development. The project's backend (Go) and bot component (Python) provided different insights:

1. **Expertise Matters:**
   - With strong domain knowledge, AI became a powerful accelerator
   - Code refinement and optimization suggestions were particularly valuable
   - Repetitive tasks were handled efficiently

2. **Pitfalls for Non-Experts:**
   - Limited domain knowledge led to accepting suboptimal AI suggestions
   - Understanding the underlying technology proved crucial for effective AI collaboration
   - Code review skills remained essential for quality assurance

In particular, I asked ChatGPT to write a function by knowing **precisely** how I would have implemented it. This is crucial since without knowing the expected result and what every line does, I might end up with a wrong implementation. The function I asked to implement is a simple similarity score among strings. Since the bot is posting deals by searching for keywords on Amazon, the PA-API is likely to return on the same request multiple version of the same product (e.g. a blue camera vs a black camera of the very same model).

With a very precise prompt, that must contain also a valid example, the generated function is very similar to the one I had in mind to solve this problem.

```text
I need a function that for similar strings returns me a similarity score.

The case that I need to address is, for example:

string A: "Canon EOS 2000D DSLR Camera and EF-S 18-55 mm f/3.5-5.6 IS II Lens - Black"
string B: "Canon EOS 2000D + EF-S 18-55mm III Lens - Easy-to-use DSLR Camera with a Versatile Lens, Ideal for Portraits and Landscapes"

Should return a similarity score higher, since they are referring to the same camera
```

Generates:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_similarity(string_a: str, string_b: str) -> float:
    """
    Computes a similarity score between two strings based on their content.

    Args:
        string_a (str): The first string.
        string_b (str): The second string.

    Returns:
        float: A similarity score between 0 and 1, where higher indicates more similarity.
    """
    def preprocess(text):
        """
        Preprocess the input text: remove special characters, lowercase, and tokenize.
        """
        # Remove non-alphanumeric characters and lowercase
        return re.sub(r'\W+', ' ', text).lower()

    # Preprocess the input strings
    clean_a = preprocess(string_a)
    clean_b = preprocess(string_b)

    # Use TF-IDF to vectorize the strings
    vectorizer = TfidfVectorizer().fit([clean_a, clean_b])
    vectors = vectorizer.transform([clean_a, clean_b])

    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0], vectors[1])
    return float(similarity[0][0])
```

The generation time is just a couple of seconds, plus one minute to read and understand if the function does what I requested. The boost in productivity is huge since I would have spent at least 3 or 4 times this time to write by myself the function.

### Multilingual Content Generation

One of the most impressive applications was in content translation. The service needed to support multiple Amazon Affiliate regions: AU, BR, CA, EG, FR, DE, IN, IT, JP, MX, NL, PL, SG, SA, ES, SE, TR, AE, UK, and US.

The bot's functionality includes Telegram posting and article generation (platinum plan), utilizing message templates stored in JSON format. For example, `US.json` contains structured messages:

```json
"NOW_AVAILABLE_MESSAGE": "💰*{title}*💰\r\n\r\n Is now available at only 💣 *{new_price}{currency}* 💣\r\n\r\n ➡️ [Go to the offer]({url})"
```

A single, well-crafted prompt handled the entire translation process:

```
Translate - if not already translated in the target language - all the JSON files in the defaults folder.

Translate only the text in the TELEGRAM section to the target language, keeping the markdown formatting, the JSON structure, the variables, the emojis, and the line breaks.

The target language is identified by the two-letter code in the filename. For example, SE.json means Swedish, FR.json means French, etc.

Do not translate files already in the target language.
```

The model efficiently processed each file:
- Identified language requirements based on filename codes
- Preserved technical elements (markdown, variables, formatting)
- Maintained consistency across translations
- Skipped already-translated content

As an example, the model correctly generated the Arabic translation while preserving the formatting, variables and emoji:

```json
"NOW_AVAILABLE_MESSAGE": "💰*{title}*💰\r\n\r\nمتوفر الآن بسعر 💣 *{new_price}{currency}* 💣 فقط\r\n\r\n➡️ [اذهب إلى العرض]({url})"
```

### Challenges and Considerations

While the overall experience was positive, several challenges emerged:

- **Context Management:** Claude Sonnet 3.5's window limits required careful task segmentation
- **Performance Trade-offs:** Balancing speed (Sonnet) versus capacity (Gemini)
- **Expertise Requirements:** Backend development highlighted the importance of human expertise
- **Quality Assurance:** Continuous review remained essential for maintaining code standards

### Conclusion

AI-assisted development has fundamentally changed my approach to coding. For frontend tasks, tools like Cline with models such as Claude Sonnet 3.5 have proven invaluable, offering rapid solutions to design challenges. In backend development, these tools excel when guided by experienced developers who can effectively validate and integrate AI suggestions.

The key to success lies in understanding that AI tools are powerful amplifiers of existing skills rather than replacements for fundamental knowledge. They excel at accelerating development cycles, improving designs, and streamlining workflows, particularly in areas outside one's core expertise.

The distinction between specialized coding models and general-purpose LLMs is significant. While both offer value, models optimized for development tasks, like Claude Sonnet 3.5, provide more focused and efficient assistance in coding scenarios.
