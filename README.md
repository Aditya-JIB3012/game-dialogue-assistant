# Game Dialogue Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ryf33jdr9b/badge.svg)](https://ask.thili.ai/b/ryf33jdr9b)

This project, "Game Dialogue Assistant," helps game designers rapidly prototype and ensure character consistency by generating dialogue snippets for NPCs. It leverages a next-word predictor to produce coherent dialogue based on learned patterns from existing game scripts.

## What you are building

**Goal.** You will build a command-line tool that generates dialogue snippets for Non-Player Characters (NPCs) based on a corpus of game scripts.

**Done when.** The project is done when users can input a starting phrase into your command-line interface and receive a generated dialogue snippet that is coherent and consistent with the trained game script corpus.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Corpus preprocessing and tokenization.
- [ ] **2. Build the n-gram language model with smoothing.**  
      Hint: N-gram counting and probability calculation with smoothing.
- [ ] **3. Implement dialogue generation logic using the model.**  
      Hint: Probabilistic word sampling for sequence generation.
- [ ] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Command-line argument parsing and output formatting.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the desired length of the generated dialogue.
- Implement an option to generate multiple alternative dialogue snippets for a given starting phrase.
- Add a feature to save the generated dialogue to a text file.
