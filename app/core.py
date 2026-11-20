"""Game Dialogue Assistant - Core functionality for n-gram language modeling and dialogue generation."""
from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# Special tokens
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

DEFAULT_GAME_DIALOGUE = [
    "Guard: Halt! Who goes there? State your business in the royal city.",
    "Guard: I used to be an adventurer like you, then I took an arrow in the knee.",
    "Guard: Keep your hands where I can see them, traveler.",
    "Guard: No lollygagging within the city walls.",
    "Guard: Trouble is brewing in the lower district. Watch your coin purse.",
    "Guard: The high king has ordered the city gates locked at nightfall.",
    "Guard: Disrespect the law and you will spend the night in the dungeon.",
    "Guard: What is it, citizen? Walk along unless you have urgent business.",
    "Guard: Stay sharp. Bandits were spotted near the western crossroads.",
    "Guard: A storm is gathering over the dragon mountains.",
    
    "Merchant: Welcome stranger! Looking for weapons, potions, or rare armor?",
    "Merchant: Take a look at my finest wares, fresh from the eastern caravan.",
    "Merchant: Everything has a price, friend. Even secrets and ancient maps.",
    "Merchant: I have healing potions that can cure any poison or battle wound.",
    "Merchant: Come back when your coin pouch is heavier, adventurer.",
    "Merchant: That enchanted blade was forged in the fires of Mount Caldera.",
    "Merchant: Best prices in the realm! You won't find better steel anywhere.",
    "Merchant: Trade has been slow since the monsters blocked the mountain pass.",
    "Merchant: Safe travels, and remember to visit my shop on your return.",
    "Merchant: If you find rare crystals in the caves, bring them straight to me.",
    
    "Wizard: The arcane energies are restless tonight. A dark omen approaches.",
    "Wizard: Magic is not a mere parlor trick, apprentice. Speak with reverence.",
    "Wizard: I have deciphered the forgotten runes of the elder kings.",
    "Wizard: Beware the shadow realm; its whispers deceive even the wisest minds.",
    "Wizard: The ancient seal beneath the citadel is beginning to crack.",
    "Wizard: Seek the staff of starlight hidden inside the enchanted sanctuary.",
    "Wizard: Knowledge is the greatest weapon against the creeping darkness.",
    "Wizard: When the blood moon rises, the spectral gate will open.",
    "Wizard: Destiny is not written in stone, but shaped by the courageous.",
    "Wizard: Focus your mind, young spellcaster, and feel the weave of magic.",
    
    "Innkeeper: Pull up a chair by the warm hearth and have a mug of cold ale.",
    "Innkeeper: A weary traveler always finds comfortable lodging and good stew here.",
    "Innkeeper: Have you heard the tavern rumors? The forgotten catacombs have stirred.",
    "Innkeeper: Many brave warriors entered the misty marsh, but few ever returned.",
    "Innkeeper: Drink up, friends! Tonight we sing of legendary heroes and lost treasure.",
    "Innkeeper: Mind your manners, stranger. We don't tolerate tavern brawls here.",
    "Innkeeper: If you are looking for work, check the bounty board by the door.",
    "Innkeeper: Rest your weary bones tonight. Tomorrow's road will be long and perilous.",
    
    "Hero: I will not let this kingdom fall to darkness as long as I draw breath.",
    "Hero: Stand together, allies! We fight for honor, freedom, and our people.",
    "Hero: The path ahead is treacherous, but courage will guide our steps.",
    "Hero: Draw your sword and face your destiny without fear.",
    "Hero: Every scar tells a tale of survival against impossible odds.",
    
    "Villain: You foolish mortals believe you can halt the inevitable eclipse.",
    "Villain: Kneel before me and witness the true power of the void.",
    "Villain: Your courage is commendable, but your defeat was written long ago.",
    "Villain: Hope is merely an illusion that blinds you to reality.",
    "Villain: Soon, shadows will swallow every corner of this fragile kingdom.",
    
    "Elder: In the first age of heroes, the dragons guarded the balance of the world.",
    "Elder: Remember the lessons of our ancestors, for history repeats itself.",
    "Elder: The sacred tree is withering because darkness has poisoned its roots.",
    "Elder: Take this blessed amulet; it will guide you through the dense fog.",
    "Elder: May the ancestral spirits watch over you in your quest.",
    
    "Thief: Keep quiet and step lightly. Shadows are our best friends.",
    "Thief: Gold shines brightest when taken from the pockets of greedy nobles.",
    "Thief: Pick the lock swiftly before the palace guards make their rounds.",
    "Thief: Everyone has something to hide, and that makes my job easier.",
    "Thief: If you need to enter the underground vault, I know a secret tunnel.",
]


def clean_dialogue_line(line: str) -> str:
    """Strip speaker tag if present and clean up whitespace."""
    line = line.strip()
    if ":" in line:
        parts = line.split(":", 1)
        # Keep dialogue content after speaker tag
        line = parts[1].strip()
    return line


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into words and punctuation marks."""
    if lowercase:
        text = text.lower()
    # Match words (including contractions like don't) and punctuation marks
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    tokens = re.findall(pattern, text)
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Convert token list back into readable natural text with correct spacing and punctuation."""
    if not tokens:
        return ""
    # Filter out special boundary tokens
    clean = [t for t in tokens if t not in (BOS, EOS, UNK)]
    if not clean:
        return ""

    out: list[str] = []
    attach_left = {".", ",", "!", "?", ";", ":", "'", "’", "n't", "')", "]", "}"}
    attach_right = {"(", "[", "{"}
    
    for i, tok in enumerate(clean):
        if not out:
            out.append(tok.capitalize())
            continue
        
        prev = out[-1]
        if tok in attach_left or tok.startswith("'"):
            out[-1] = prev + tok
        elif prev in attach_right:
            out[-1] = prev + tok
        else:
            # Capitalize if previous token ended a sentence
            if prev.endswith((".", "!", "?")):
                out.append(tok.capitalize())
            else:
                out.append(tok)
                
    result = " ".join(out)
    # Ensure final capitalization and punctuation
    if result and not result[-1] in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else ""


def pad_sequence(tokens: list[str], n: int) -> list[str]:
    """Pad token list with (n - 1) BOS tokens at start and 1 EOS token at end."""
    return [BOS] * (n - 1) + tokens + [EOS]


def preprocess_corpus(corpus_input: str | list[str] | None = None, n: int = 3, lowercase: bool = True) -> list[list[str]]:
    """Preprocess dialogue lines into tokenized, padded sequences."""
    if corpus_input is None:
        lines = DEFAULT_GAME_DIALOGUE
    elif isinstance(corpus_input, str):
        lines = [line for line in corpus_input.splitlines() if line.strip()]
    else:
        lines = list(corpus_input)

    processed: list[list[str]] = []
    for line in lines:
        cleaned = clean_dialogue_line(line)
        if not cleaned:
            continue
        toks = tokenize(cleaned, lowercase=lowercase)
        if toks:
            padded = pad_sequence(toks, n=n)
            processed.append(padded)
    return processed


def build_vocabulary(tokenized_corpus: list[list[str]], min_freq: int = 1) -> set[str]:
    """Build vocabulary from tokenized corpus with minimum frequency threshold."""
    counts = Counter(token for seq in tokenized_corpus for token in seq)
    vocab = {token for token, count in counts.items() if count >= min_freq}
    vocab.add(BOS)
    vocab.add(EOS)
    vocab.add(UNK)
    return vocab


class Corpus:
    """Manages loading, cleaning, and tokenizing a game dialogue corpus."""

    def __init__(self, raw_lines: list[str] | None = None, n: int = 3, min_freq: int = 1):
        self.raw_lines = raw_lines if raw_lines is not None else list(DEFAULT_GAME_DIALOGUE)
        self.n = n
        self.min_freq = min_freq
        self.tokenized_sentences = preprocess_corpus(self.raw_lines, n=self.n)
        self.vocab = build_vocabulary(self.tokenized_sentences, min_freq=self.min_freq)

    @classmethod
    def from_file(cls, path: str | Path, n: int = 3, min_freq: int = 1) -> Corpus:
        text = Path(path).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return cls(raw_lines=lines, n=n, min_freq=min_freq)

    def stats(self) -> dict:
        total_tokens = sum(len(seq) for seq in self.tokenized_sentences)
        return {
            "num_dialogues": len(self.raw_lines),
            "num_sentences": len(self.tokenized_sentences),
            "total_tokens": total_tokens,
            "vocab_size": len(self.vocab),
        }


# --- Milestone 1: Collect and preprocess a corpus of game dialogues.
# Hint: Corpus preprocessing and tokenization.
def milestone_1(corpus_input: str | list[str] | None = None, n: int = 3) -> dict:
    """Milestone 1: Collect and preprocess game dialogue corpus.
    
    Returns tokenized corpus, vocabulary, and summary statistics.
    """
    corpus = Corpus(raw_lines=corpus_input if isinstance(corpus_input, list) else None, n=n)
    if isinstance(corpus_input, str):
        corpus = Corpus(raw_lines=[line for line in corpus_input.splitlines() if line.strip()], n=n)
    
    return {
        "corpus": corpus,
        "tokenized_sentences": corpus.tokenized_sentences,
        "vocab": corpus.vocab,
        "stats": corpus.stats(),
    }


# --- Milestone 2: Build the n-gram language model with smoothing.
# Hint: N-gram counting and probability calculation with smoothing.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Implement dialogue generation logic using the model.
# Hint: Probabilistic word sampling for sequence generation.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
# Hint: Command-line argument parsing and output formatting.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
