---
layout: post
title: "Use Your Brain: Engineering Standards in the Age of LLMs"
date: 2026-07-26 12:00:00
categories: ai
summary: "Software development is not a niche anymore. The advent of LLMs made it possible for everyone to write their own software - and this is a good thing. However, I see a huge number of software engineers (the real ones!) falling into the vibe-coding trap. Those people were used to design - think! - and later implement. Now, instead, they are just producing a huge amount of dubious quality code, something that I would expect from the average Joe, not from a qualified software engineer."
authors:
    - pgaleone
---

Software development is not a niche anymore. The advent of LLMs made it possible for everyone to write their own software - and this is a good thing. However, I see a huge number of software engineers (the real ones!) falling into the vibe-coding trap. Those people were used to design - think! - and later implement. Now, instead, they are just producing a huge amount of dubious quality code, something that I would expect from the average Joe, not from a qualified software engineer.

To be clear: I'm sharing my own philosophy here, not preaching a gospel. I'm not trying to convince anyone to adopt my workflow.

## The velocity trap

LLMs produce code at the speed of light, they crunch and generate tokens[^1] and give you pretty much what you asked for. Depending on the codebase size and status, the usage of LLMs should be correctly guided - and companies and solo-developers are not following any clear guideline apart from "deliver fast".

Delivering fast is perfect, if and only if, you are developing your own script, for yourself, starting from an idea and seeing it working in minutes.

Delivering fast becomes a problem when the tool you developed exits your garage and starts being used by others. Immediately, issues start to pop up and you have to fix them. You can ask an LLM to fix them - and this is precisely how you'll lose control of your codebase.

Let's think about the concept of code ownership for a second.

Is a fully generated codebase *really* your codebase? I have a rule of thumb for understanding it.

<span id="code-ownership"></span>

> If you can explain how the code works and pinpoint bugs on your own, it’s your codebase. But if you rely on an LLM for every single edit, you aren't managing software - you're just letting an AI manipulate random code you no longer understand.


When the codebase you're working on is not yours and it is a massive work being developed by several developers over the years - delivering as fast as possible is still a goal (not the only one. I'm not a fan of speed at all cost. I prefer well designed solutions, with initial effort and a steady ramp up afterwards). You can develop as fast as possible with the help of AI - but you need to be very careful about everything that's committed, potential side effects, tests to be kept up-to-date and best practices to respect. At the very end, even if the LLM generated part or even all the code, you should be able to tell what the LLM did. If, instead, you just throw an LLM over any open issue and no [software development process](https://en.wikipedia.org/wiki/Software_development_process) is stable and mandatory, then this codebase goes out of control quite fast, and no-one knows what's happening anymore very quickly.

Being pushed to deliver as much value as possible in the shortest time possible, makes everyone a [cowboy coder](https://en.wikipedia.org/wiki/Cowboy_coding) - every good software development methodology is seen as a problem, not as something that brings value (here's the trap). After all, what really matters (and - in pathologic organizations - what is measured and evaluated as a good thing) is the development speed, isn't it?

## Ownership vs Custodianship

Do you want to move fast *only*? Or do you want - also - to build something maintainable, scalable, secure and that you *really* own?

If you're a pure cowboy, you just care about getting things done as fast as possible, and perhaps delegate the maintenance of your code to someone else after doing the first PoC - understandable, but it's not scalable and whoever inherits your cowboy-developed codebase is going to look at that code, wondering if rewriting it from scratch makes more sense, and blaming you for your (not really yours) design decisions.

Being a cowboy you have no ownership, you are acting as a *custodian* rather than an owner. The whole mental model and the relationship with AI changes when you're an owner vs a custodian. The cowboys see the codebase as a black box more or less, they just know what goes in, what comes out, but the internal logic is blurred. They rely on the LLM to do the heavy lifting, from design to execution and debugging.

The fun part is when bugs happen: they can't debug their own (own?) code, and they can only hope the LLM knows how to fix the code.
Of course the core value is raw speed and immediate output (fast fast fast!).

Who wants to be a cowboy? I, as a software engineer, don't. I see myself, and all the software engineers, being owners rather than custodians. The owner mental model is different: they have a complete map of the system in their head, they know *why* a specific architecture was chosen and where the edge cases hide. How changing something in a module affects other dependent modules. Their relationship with AI is more healthy too: they use the LLM as a productivity/speed booster. Every output is reviewed and the committed code can be explained, line by line. In case of bugs, they can pinpoint the area of the code that is likely to be responsible for the issue, and act timely, knowing what to change for fixing, and how the changes affect the entire codebase.

Organizations, and the general AI usage at all costs even in solo-projects, are pushing developers to move fast-fast-fast, with the serious risk for them to fall into the vibe-coding trap. They quietly downgrade their role from owners to custodians, thinking they are moving faster, but in reality surrendering control of the system to an AI - until the technical debt comes back to bite them.


## Correctly Leveraging AI: infrastructure and netiquette

It's surreal, but solo-developers and companies (!) should go back to the basics and set up very strong engineering practices and guardrails when it comes to coding - especially AI-assisted coding.

If you want to work on something where:

- People work towards a shared goal.
- People have a real ownership of the committed code.
- Agents and people work seamlessly in a structured environment that guarantees code quality, safety, and long-term maintainability.

The correct environment should be set up. I think that this environment already exists, but it should be enforced company-level and applied by solo-developers. The rules are quite simple:

1. Define the rules: define the human roles, the agents roles, the responsibilities of every person and rule the AI coding practices. Ban the cowboys.
2. Set the standards to follow.
3. Set up the tooling that supports developers and agents in respecting the rules and following the standards.
4. Enforce everything through CI/CD.
5. People should talk with people.
6. AI-assisted coding should be done with proper ownership.
7. Agent-generated code should be human reviewed - mandatory.
8. Responding to a human-written issue with raw AI output is prohibited.

I'm deliberately being generic since every company and every developer has their own set of rules and standards - but some of them are like the new *netiquette*.

 We are literally reinventing the wheel and going back to the old practices that people defined in the 80s - the netiquette, born with early computer networks and Usenet is more relevant today than ever. I'm actually totally pissed off when someone answers something I wrote with some AI slop. I took the time to write it, you must take your time to read it and write a proper answer. Use your brain.

For the tooling part, my suggestion is the obvious: strong CI/CD. Use linters, formatters, pre-commit, ensure the testing suite is human reviewed and the tests are correct. Moreover, take advantage of AI customization: a strongly-defined `AGENTS.md` file with clear guardrails; followed by a small set of human-written skills - to guide the agents during their thinking and tooling usage process.

## Conclusion

AI *can* be a productivity booster - it's an amplifier. Amplifying good engineering habits creates velocity without compromising quality, guaranteeing real ownership over the codebase. Amplifying laziness just fills your (once again - yours?) repository with noise, and you'll end up being a custodian with no ownership in a short time.

Leverage the tools, speed up your workflow, but never surrender your engineering judgement.

Use your brain.

---

[^1]: Everyone talks about tokens nowadays. I remember when the tokens were just the result of "tokenization", a deterministic step used when doing stuff like TF-IDF. Every token was a word, full-stop. Now instead, a token is "a token" there's no standard definition or fixed rule, since every model decides what a token is. If you have a locally deployed model you have the control (more or less) of what the tokens are. If the models run on the cloud, well, a token is what the model provider wants to sell you as a token.
