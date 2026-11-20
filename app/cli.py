"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import (
    Corpus,
    DEFAULT_GAME_DIALOGUE,
    NGramModel,
    format_cli_output,
    generate_dialogue,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="game-dialogue-assistant",
        description="Game designers use this tool to generate dialogue snippets for NPCs based on a corpus of game scripts, ensuring character consistency and rapid prototyping.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Built-in check
    sub.add_parser("hello", help="check that everything is wired up")

    # Generate command
    gen_parser = sub.add_parser("generate", help="generate dialogue snippet(s) from a starting phrase")
    gen_parser.add_argument("prompt", nargs="?", default="Halt", help="starting phrase or keyword for the NPC dialogue")
    gen_parser.add_argument("-l", "--length", type=int, default=25, help="desired length/maximum tokens (default: 25)")
    gen_parser.add_argument("-s", "--samples", type=int, default=1, help="number of alternative dialogue snippets (default: 1)")
    gen_parser.add_argument("-o", "--save", type=str, default=None, help="save generated dialogue to a text file")
    gen_parser.add_argument("-t", "--temperature", type=float, default=0.7, help="sampling temperature (0.0 for deterministic, default: 0.7)")
    gen_parser.add_argument("-n", "--n-gram", type=int, default=3, help="n-gram model order (default: 3)")
    gen_parser.add_argument("-k", "--smoothing", type=float, default=0.05, help="Add-k smoothing parameter (default: 0.05)")
    gen_parser.add_argument("-c", "--corpus", type=str, default=None, help="path to custom game dialogue text file")
    gen_parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible generation")

    # Interactive mode
    interactive_parser = sub.add_parser("interactive", help="start an interactive dialogue generation session")
    interactive_parser.add_argument("-c", "--corpus", type=str, default=None, help="path to custom game dialogue text file")
    interactive_parser.add_argument("-n", "--n-gram", type=int, default=3, help="n-gram model order (default: 3)")
    interactive_parser.add_argument("-s", "--samples", type=int, default=2, help="number of alternative snippets per prompt (default: 2)")

    # Stats command
    stats_parser = sub.add_parser("stats", help="display statistics about the dialogue corpus")
    stats_parser.add_argument("-c", "--corpus", type=str, default=None, help="path to custom game dialogue text file")

    return parser


def load_model(corpus_path: str | None = None, n: int = 3, k: float = 0.05) -> tuple[NGramModel, Corpus]:
    default_file = Path("data/game_dialogue_corpus.txt")
    if corpus_path and Path(corpus_path).exists():
        corpus = Corpus.from_file(corpus_path, n=n)
    elif default_file.exists():
        corpus = Corpus.from_file(default_file, n=n)
    else:
        corpus = Corpus(raw_lines=list(DEFAULT_GAME_DIALOGUE), n=n)

    model = NGramModel(n=n, k=k)
    model.train(corpus.tokenized_sentences, vocab=corpus.vocab)
    return model, corpus


def run_interactive(corpus_path: str | None = None, n: int = 3, samples: int = 2) -> None:
    print("=" * 60)
    print("Welcome to the Game Dialogue Assistant - Interactive Mode")
    print("Type a starting phrase to generate NPC dialogue (or 'exit' to quit).")
    print("=" * 60)
    
    model, _ = load_model(corpus_path=corpus_path, n=n)
    
    while True:
        try:
            prompt = input("\nEnter starting phrase > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode. Farewell, adventurer!")
            break

        if not prompt or prompt.lower() in ("exit", "quit", "q"):
            print("Exiting interactive mode. Farewell, adventurer!")
            break

        snippets = generate_dialogue(
            model=model,
            prompt=prompt,
            max_length=25,
            num_samples=samples,
            temperature=0.8,
        )
        snippet_list = snippets if isinstance(snippets, list) else [snippets]
        output = format_cli_output(prompt=prompt, snippets=snippet_list)
        print(output)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Game Dialogue Assistant is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        model, _ = load_model(corpus_path=args.corpus, n=args.n_gram, k=args.smoothing)
        snippets = generate_dialogue(
            model=model,
            prompt=args.prompt,
            max_length=args.length,
            num_samples=args.samples,
            temperature=args.temperature,
            seed=args.seed,
            save_path=args.save,
        )
        snippet_list = snippets if isinstance(snippets, list) else [snippets]
        output = format_cli_output(prompt=args.prompt, snippets=snippet_list, save_path=args.save)
        print(output)
    elif args.command == "interactive":
        run_interactive(corpus_path=args.corpus, n=args.n_gram, samples=args.samples)
    elif args.command == "stats":
        _, corpus = load_model(corpus_path=args.corpus)
        stats = corpus.stats()
        print("Corpus Statistics:")
        print(f"  Dialogue lines: {stats['num_dialogues']}")
        print(f"  Total tokens:   {stats['total_tokens']}")
        print(f"  Vocabulary:     {stats['vocab_size']} unique words")


if __name__ == "__main__":
    main()
