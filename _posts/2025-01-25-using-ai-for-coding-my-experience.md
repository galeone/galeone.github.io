---
layout: post
title: "Using AI for Coding: My Journey with Cline and Large Language Models"
date: 2025-01-25 08:00:00
categories: ai coding
summary: "How I leveraged AI tools like Cline to enhance the UI/UX of a website and streamline backend tasks. From redesigning pages and translating content to navigating the benefits and challenges of AI-assisted development, this blog post highlights the potential of using large language models to boost productivity while sharing key lessons learned."
authors:
    - pgaleone
---

In recent months, I embarked on a journey to improve the UI/UX of a side project—[bot.eofferte.eu](https://bot.eofferte.eu), a SaaS designed to automate Amazon affiliate marketing on Telegram and help users get started with Amazon Associates.

The backend of the project is built entirely in Go, utilizing the [labstack/echo](https://github.com/labstack/echo) framework, with UI rendering handled by the standard `html/template` package. Curious about how AI could assist in enhancing the design and functionality of the website, I used [Cline](https://github.com/cline/cline) through its VSCode plugin as my primary AI coding tool. Here’s what I discovered.

---

### Experiments with Frontend Development

As someone who struggles with UI/UX, I found AI to be an invaluable partner in addressing these challenges.

I have minimal knowledge of modern web frameworks and generally dislike writing CSS or dealing with layout nuances. I prefer focusing on backend development. Given these limitations, I was thrilled with the results produced by the LLMs. The output was professional, and the time saved by relying on the AI instead of my limited design skills was immeasurable.

I tasked the LLMs with redesigning every page of the website. The landing page was completely redefined, while the management page (where users configure the service) underwent massive improvements. Previously, the design was amateurish, but now it feels polished and professional.

The LLMs also helped with the more tedious tasks, such as drafting the privacy policy, terms of service, and other legal compliance sections. These were generated entirely by the models and later refined manually.

I experimented with multiple models, including [Claude Sonnet 3.5]([https://www.anthropic.com/news/claude-3-5-sonnet]) and **Gemini**, and here are my key takeaways:

1. **Claude Sonnet 3.5:**

   - **Performance:** Lightning-fast responses made it a pleasure to use.
   - **Accuracy:** It understood tasks well and excelled at handling HTML, CSS, and JavaScript.
   - **Effectiveness:** The model suggested practical solutions using frameworks like Font Awesome and Bootstrap, greatly improving the organization and aesthetics of the site.
   - **Limitations:** I frequently hit the context window limit, leading to redundant loops and unnecessary credit consumption.

2. **Gemini:**

   - **Performance:** Much slower compared to Sonnet.
   - **Advantage:** Its larger context window allowed it to handle more extensive instructions.

The AI’s ability to propose frameworks and cohesive designs was transformative. For someone like me, who lacks a natural aptitude for UI/UX, these tools were a significant productivity booster.

---

### Prompt Experiments

Working with Cline is intuitive. It analyzes open files to create a context and traverses the repository to fulfill the requested tasks. For example, I had a file named `bot.html` containing the `html/template` code for the bot management section.

The previous version of the management page required users to complete all the fields in a large form and submit it in one go. I wanted to redesign it into a guided wizard, splitting the form into manageable sections. I found a library, [Enchanter.js](https://github.com/brunnopleffken/enchanter), that seemed easy to use and embedded it into the repository.

To integrate the library properly, I needed to redesign the page layout and adjust the HTML to match the navigation tags required by the library. After some trial and error, I refined the following prompt:

```
analyze bot.html - it's a Go (golang) html template.

bot.html contains both html template code and JavaScript. Both are mixed with Go template syntax.

You need to rewrite bot.html using static/enchanter.js in order to convert the form in bot.html to a guided wizard.

Do not touch any JavaScript already present in bot.html and ignore every JavaScript error.

Your <form> tag should wrap the .nav and .tab-content elements. The footer of the form must contain "Back", "Next" and "Finish" buttons with the data-enchanter attributes.
```

Key aspects of this prompt:

1. **Initial Context:** Clearly describe the technologies and languages involved, as well as the scope (a single page in this case).
2. **Task Description:** Provide a detailed explanation of the task, including file paths if relevant.
3. **Constraints:** Specify behavior guidelines. This was crucial. Since JavaScript code was mixed with `html/template` syntax, it created syntax errors in the VSCode JavaScript parser. Without the constraint to "ignore every JavaScript error," Cline would repeatedly loop and waste credits trying to fix non-issues.

---

### Experiments with Backend Development

For backend development, the experience was markedly different. The backend of my website is built in Go, and the bot itself is written in Python. I wanted to see how well the models could assist in this domain. Here’s what I found:

1. **Expertise Matters:**

   - When I had a clear understanding of the task and could implement it myself, using AI was a massive productivity boost. The models helped refine my code, suggest optimizations, and handle repetitive tasks efficiently.

2. **Pitfalls for Non-Experts:**

   - Without a solid understanding of the language or domain, the solutions provided by the AI were often suboptimal or incorrect. This highlighted the importance of expertise when using AI for backend tasks.

---

### Translation Use Case

Another huge productivity boost came in the area of translation. The capability of LLMs to translate text accurately is simply astonishing.

Amazon Affiliates operates in multiple regions, including AU, BR, CA, EG, FR, DE, IN, IT, JP, MX, NL, PL, SG, SA, ES, SE, TR, AE, UK, and US. This meant the service had to be available in all these regions and in their respective languages.

The bot posts on Telegram and creates articles on the website (with the platinum plan). It provides users with ready-to-use templates for messages, such as notifications for deals, out-of-stock products, and more.

I organized these templates in a folder with JSON files. Each file contains markdown text for the messages, along with syntax for variables. For example, `US.json` contains entries like this:

```
"NOW_AVAILABLE_MESSAGE": "💰*{title}*💰\r\n\r\n Is now available at only 💣 *{new_price}{currency}* 💣\r\n\r\n ➡️ [Go to the offer]({url})",
```

Using a single prompt, I was able to translate all the JSON files correctly in just a few seconds:

```
Translate - if not already translated in the target language - all the JSON files in the defaults folder.

Translate only the text in the TELEGRAM section to the target language, keeping the markdown formatting, the JSON structure, the variables, the emojis, and the line breaks.

The target language is identified by the two-letter code in the filename. For example, SE.json means Swedish, FR.json means French, etc.

Do not translate files already in the target language.
```

The model responded accurately, listing all files and their corresponding languages, and only translating where necessary:

```
Looking at the environment details, I can see there are multiple JSON files in the defaults folder that need to be checked and potentially translated. From the previous interaction, I can see that FR.json and SE.json have already been translated. Let me list all the files and their corresponding languages:

AE.json - Arabic (UAE)
AU.json - English (Australia) - No translation needed
BR.json - Portuguese (Brazil)
CA.json - English (Canada) - No translation needed
DE.json - German
EG.json - Arabic (Egypt)
ES.json - Spanish
FR.json - French (Already translated)
IN.json - Hindi/English (India)
IT.json - Italian
JP.json - Japanese
MX.json - Spanish (Mexico)
NL.json - Dutch
PL.json - Polish
SA.json - Arabic (Saudi Arabia)
SE.json - Swedish (Already translated)
SG.json - English (Singapore) - No translation needed
TR.json - Turkish
UK.json - English (UK) - No translation needed
US.json - English (US) - No translation needed
```

---

### Challenges and Limitations

While my experiments were largely positive, there were notable challenges:

- **Context Window Limits:** Especially with Claude Sonnet 3.5, hitting the context limit was a recurring issue, leading to redundant loops and wasted credits.
- **Speed vs. Capability:** The trade-off between Sonnet’s speed and Gemini’s larger context window meant balancing efficiency with capacity.
- **Dependency Risks:** While AI suggestions were reliable for frontend tasks, over-reliance on AI for backend work without sufficient expertise could result in subpar solutions.

---

### Final Thoughts

Using AI to code has been a transformative experience. For frontend development and UI/UX improvements, tools like Cline and models such as Claude Sonnet 3.5 proved indispensable, offering fast and effective solutions. For backend development, these tools shine brightest when used by experienced developers who can guide the models and validate their outputs.

While LLMs are not a silver bullet, they are remarkable productivity boosters when used wisely. They enable faster development cycles, improved designs, and smoother workflows, particularly in areas where one might lack expertise or motivation.

However, caution is necessary. Over-reliance on these tools without a strong understanding of the underlying technologies can lead to inefficiencies and subpar outcomes. AI is best viewed as a collaborator—a powerful tool to amplify your capabilities, not replace them.

The differences between models targeted for assisting with coding and general purpose LLMs is also worth noting. While general-purpose models like GPT-3 can provide valuable insights and suggestions, models like Claude Sonnet 3.5 are tailored specifically for coding tasks, making them more efficient and effective in this domain.