# Game Dialogue Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ryf33jdr9b/badge.svg)](https://ask.thili.ai/b/ryf33jdr9b)

This project, "Game Dialogue Assistant," helps game designers rapidly prototype and ensure character consistency by generating dialogue snippets for NPCs. It leverages an n-gram next-word predictor built from scratch with smoothing to produce coherent dialogue based on learned patterns from game scripts.

## What you are building

**Goal.** You will build a command-line tool that generates dialogue snippets for Non-Player Characters (NPCs) based on a corpus of game scripts.

**Done when.** The project is done when users can input a starting phrase into your command-line interface and receive a generated dialogue snippet that is coherent and consistent with the trained game script corpus.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Corpus preprocessing and tokenization.
- [x] **2. Build the n-gram language model with smoothing.**  
      Hint: N-gram counting and probability calculation with smoothing.
- [x] **3. Implement dialogue generation logic using the model.**  
      Hint: Probabilistic word sampling for sequence generation.
- [x] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Command-line argument parsing and output formatting.

## Architecture & How It Works

1. **Corpus Collection & Preprocessing (`app/core.py`)**:
   - A rich corpus of authentic NPC dialogues spanning Guards, Merchants, Wizards, Innkeepers, Heroes, Villains, Elders, and Thieves is included in `data/game_dialogue_corpus.txt`.
   - Handles speaker-tag extraction, word and punctuation tokenization, sentence boundary padding (`<s>` and `</s>`), and vocabulary indexing with unknown token (`<unk>`) support.
   - Text detokenization formats generated tokens cleanly back into natural dialogue with appropriate capitalization and punctuation attachment.

2. **N-gram Language Model with Smoothing**:
   - Implements configurable $N$-gram counting ($N=1, 2, 3$).
   - Calculates next-word probability distributions using Add-$k$ (Laplace / Lidstone) smoothing.
   - Includes backoff logic: when higher-order contexts are unseen, backs off to lower orders ($N-1 \dots 1$) to ensure resilient, high-quality transitions.
   - Evaluates test perplexity and guarantees mathematical probability distribution summation ($\sum P = 1.0$).
   - Full model persistence with `save()` and `load()`.

3. **Probabilistic Dialogue Generation**:
   - Generates autoregressive sequences token-by-token from an input starting phrase.
   - Features temperature scaling ($T$), Top-$k$ filtering, and Top-$p$ (nucleus) sampling.
   - Supports single or multiple alternative dialogue snippet generation.

4. **CLI Interface (`app/cli.py`)**:
   - `python -m app.cli hello`: Sanity check command.
   - `python -m app.cli generate "<prompt>"`: Generates dialogue from a prompt.
   - `python -m app.cli interactive`: Launches interactive conversational generator.
   - `python -m app.cli stats`: Shows corpus and vocabulary metrics.

## Stretch Features Implemented

- **Configurable Dialogue Length**: Use `-l` / `--length` to control maximum tokens generated.
- **Multiple Alternative Snippets**: Use `-s` / `--samples` to generate $N$ alternative dialogue choices for a single prompt.
- **File Saving**: Use `-o` / `--save <file>` to automatically save generated dialogue outputs to a text file.

## Run It

You need Python 3.12+ and git.

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy settings
cp .env.example .env

# Verify setup
python -m app.cli hello

# Run full test suite
pytest
```

## CLI Examples

### 1. Generate Dialogue from a Prompt
```bash
python -m app.cli generate "Halt! Who goes"
```

Output:
```text
==================================================
GAME DIALOGUE ASSISTANT
==================================================
Starting Prompt: "Halt! Who goes"
--------------------------------------------------
Generated Dialogue:
  Halt! Who goes there? State your business in the royal city.
--------------------------------------------------
```

### 2. Generate Multiple Alternative Snippets with Length Control & File Output
```bash
python -m app.cli generate "Welcome stranger" --length 20 --samples 3 --save dialogue_output.txt
```

### 3. Interactive Dialogue Generator
```bash
python -m app.cli interactive
```

### 4. Corpus Statistics
```bash
python -m app.cli stats
```

## Settings and Secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub, treat it as leaked and make a new one.
