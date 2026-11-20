"""Game Dialogue Assistant - Core functionality for n-gram language modeling and dialogue generation."""
from __future__ import annotations

import json
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
        line = parts[1].strip()
    return line


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into words and punctuation marks."""
    if lowercase:
        text = text.lower()
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    tokens = re.findall(pattern, text)
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Convert token list back into readable natural text with correct spacing and punctuation."""
    if not tokens:
        return ""
    clean = [t for t in tokens if t not in (BOS, EOS, UNK)]
    if not clean:
        return ""

    out: list[str] = []
    attach_left = {".", ",", "!", "?", ";", ":", "'", "’", "n't", "')", "]", "}"}
    attach_right = {"(", "[", "{"}
    
    for tok in clean:
        if not out:
            out.append(tok.capitalize())
            continue
        
        prev = out[-1]
        if tok in attach_left or tok.startswith("'"):
            out[-1] = prev + tok
        elif prev in attach_right:
            out[-1] = prev + tok
        else:
            if prev.endswith((".", "!", "?")):
                out.append(tok.capitalize())
            else:
                out.append(tok)
                
    result = " ".join(out)
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

class NGramModel:
    """N-gram language model supporting orders 1..N with Add-k (Laplace/Lidstone) smoothing and backoff."""

    def __init__(self, n: int = 3, k: float = 1.0, backoff: bool = True):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.k = k
        self.backoff = backoff
        self.vocab: set[str] = set()
        self.target_vocab: list[str] = []
        # counts[order][context] -> Counter(word -> count)
        # order ranges from 1 to n.
        # For order 1 (unigram), context is empty tuple ()
        self.counts: dict[int, dict[tuple[str, ...], Counter]] = {i: defaultdict(Counter) for i in range(1, n + 1)}
        self.context_totals: dict[int, dict[tuple[str, ...], int]] = {i: defaultdict(int) for i in range(1, n + 1)}
        self.total_words: int = 0

    def train(self, tokenized_sentences: list[list[str]], vocab: set[str] | None = None) -> NGramModel:
        """Count n-grams across all sentences in the corpus."""
        if vocab is not None:
            self.vocab = set(vocab)
        else:
            self.vocab = build_vocabulary(tokenized_sentences)
        
        # Target vocabulary: tokens that can be predicted (excludes BOS)
        self.target_vocab = sorted(t for t in self.vocab if t != BOS)

        # Count n-grams for every order from 1 up to self.n
        for sentence in tokenized_sentences:
            # Map tokens outside vocab to UNK
            clean_seq = [tok if tok in self.vocab else UNK for tok in sentence]
            for i, target in enumerate(clean_seq):
                if target == BOS:
                    continue  # BOS is never a predicted target
                self.total_words += 1
                # Record counts for unigram, bigram, ..., n-gram
                for order in range(1, self.n + 1):
                    ctx_len = order - 1
                    if i >= ctx_len:
                        context = tuple(clean_seq[i - ctx_len : i])
                        self.counts[order][context][target] += 1
                        self.context_totals[order][context] += 1
        return self

    def _resolve_context(self, context: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
        """Trim context to at most (n - 1) tokens, and back off to shorter context if unseen."""
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        
        if not self.backoff:
            order = len(ctx) + 1
            return order, ctx

        # If backoff is enabled, find highest order where context was seen
        while len(ctx) > 0:
            order = len(ctx) + 1
            if self.context_totals[order].get(ctx, 0) > 0:
                return order, ctx
            ctx = ctx[1:]  # Back off by removing oldest token

        return 1, ()

    def get_probability(self, context: tuple[str, ...], word: str) -> float:
        """Calculate P(word | context) using Add-k smoothing on resolved context."""
        target = word if word in self.vocab else UNK
        if target == BOS:
            return 0.0

        order, resolved_ctx = self._resolve_context(context)
        v_size = len(self.target_vocab)
        if v_size == 0:
            return 0.0

        ctx_count = self.context_totals[order].get(resolved_ctx, 0)
        word_count = self.counts[order][resolved_ctx][target]

        # Add-k smoothing formula: P = (C(ctx, w) + k) / (C(ctx) + k * |V|)
        prob = (word_count + self.k) / (ctx_count + self.k * v_size)
        return prob

    def get_next_word_probabilities(self, context: tuple[str, ...]) -> dict[str, float]:
        """Compute full probability distribution over target vocabulary for given context."""
        order, resolved_ctx = self._resolve_context(context)
        v_size = len(self.target_vocab)
        ctx_count = self.context_totals[order].get(resolved_ctx, 0)
        denom = ctx_count + self.k * v_size

        probs: dict[str, float] = {}
        for word in self.target_vocab:
            word_count = self.counts[order][resolved_ctx][word]
            probs[word] = (word_count + self.k) / denom
        return probs

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Calculate model perplexity on a list of test sentences."""
        log_prob_sum = 0.0
        word_count = 0

        for sentence in sentences:
            clean_seq = [tok if tok in self.vocab else UNK for tok in sentence]
            for i, target in enumerate(clean_seq):
                if target == BOS:
                    continue
                ctx = tuple(clean_seq[max(0, i - self.n + 1) : i])
                prob = self.get_probability(ctx, target)
                if prob <= 0.0:
                    prob = 1e-12
                log_prob_sum += math.log(prob)
                word_count += 1

        if word_count == 0:
            return float("inf")
        return math.exp(-log_prob_sum / word_count)

    def to_dict(self) -> dict:
        """Serialize model data to dictionary."""
        serializable_counts: dict[str, dict[str, dict[str, int]]] = {}
        serializable_totals: dict[str, dict[str, int]] = {}
        for order in self.counts:
            serializable_counts[str(order)] = {
                " ".join(ctx): dict(word_counts) for ctx, word_counts in self.counts[order].items()
            }
            serializable_totals[str(order)] = {
                " ".join(ctx): total for ctx, total in self.context_totals[order].items()
            }
        return {
            "n": self.n,
            "k": self.k,
            "backoff": self.backoff,
            "vocab": sorted(self.vocab),
            "target_vocab": self.target_vocab,
            "counts": serializable_counts,
            "context_totals": serializable_totals,
            "total_words": self.total_words,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NGramModel:
        """Deserialize model from dictionary."""
        model = cls(n=data["n"], k=data["k"], backoff=data.get("backoff", True))
        model.vocab = set(data["vocab"])
        model.target_vocab = data["target_vocab"]
        model.total_words = data.get("total_words", 0)
        for order_str, ctx_map in data["counts"].items():
            order = int(order_str)
            for ctx_str, word_counts in ctx_map.items():
                ctx = tuple(ctx_str.split()) if ctx_str else ()
                model.counts[order][ctx] = Counter(word_counts)
        for order_str, totals_map in data["context_totals"].items():
            order = int(order_str)
            for ctx_str, total in totals_map.items():
                ctx = tuple(ctx_str.split()) if ctx_str else ()
                model.context_totals[order][ctx] = total
        return model

    def save(self, filepath: str | Path) -> None:
        """Save model JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: str | Path) -> NGramModel:
        """Load model from JSON file."""
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def milestone_2(corpus_input: str | list[str] | None = None, n: int = 3, k: float = 1.0) -> dict:
    """Milestone 2: Build the n-gram language model with smoothing.
    
    Trains an n-gram model and returns model instance, perplexity, and probability checks.
    """
    m1 = milestone_1(corpus_input=corpus_input, n=n)
    model = NGramModel(n=n, k=k)
    model.train(m1["tokenized_sentences"], vocab=m1["vocab"])
    
    # Verify probability distribution sums to 1.0
    sample_context = (BOS, BOS) if n >= 3 else (BOS,)
    probs = model.get_next_word_probabilities(sample_context)
    prob_sum = sum(probs.values())
    
    ppl = model.perplexity(m1["tokenized_sentences"])
    
    return {
        "model": model,
        "n": n,
        "k": k,
        "perplexity": ppl,
        "prob_sum": prob_sum,
        "vocab_size": len(model.vocab),
    }


# --- Milestone 3: Implement dialogue generation logic using the model.
# Hint: Probabilistic word sampling for sequence generation.

def sample_next_token(
    probs: dict[str, float],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    disallow_eos: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Sample the next token from probabilities with temperature, top-k, and top-p filtering."""
    if rng is None:
        rng = random

    filtered = dict(probs)
    if disallow_eos and EOS in filtered:
        filtered.pop(EOS)

    # Disallow UNK from generation if possible
    if UNK in filtered and len(filtered) > 1:
        filtered.pop(UNK)

    if not filtered:
        return EOS

    words = list(filtered.keys())
    raw_probs = [filtered[w] for w in words]

    # Greedy choice if temperature is 0
    if temperature <= 0.0:
        max_idx = max(range(len(raw_probs)), key=lambda i: raw_probs[i])
        return words[max_idx]

    # Temperature scaling: P_new ~ P^(1/T)
    inv_t = 1.0 / max(temperature, 1e-4)
    scaled = [p ** inv_t for p in raw_probs]
    total_scaled = sum(scaled)
    if total_scaled <= 0:
        return rng.choice(words)
    p_norm = [s / total_scaled for s in scaled]

    # Combine words and normalized probabilities and sort descending
    sorted_pairs = sorted(zip(words, p_norm), key=lambda x: x[1], reverse=True)

    # Apply Top-k filtering
    if top_k is not None and top_k > 0:
        sorted_pairs = sorted_pairs[:top_k]

    # Apply Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        cumulative = 0.0
        cutoff_idx = 0
        for idx, (_, prob) in enumerate(sorted_pairs):
            cumulative += prob
            cutoff_idx = idx
            if cumulative >= top_p:
                break
        sorted_pairs = sorted_pairs[: cutoff_idx + 1]

    candidates = [pair[0] for pair in sorted_pairs]
    weights = [pair[1] for pair in sorted_pairs]
    total_w = sum(weights)
    if total_w <= 0:
        return candidates[0]
    normalized_weights = [w / total_w for w in weights]

    return rng.choices(candidates, weights=normalized_weights, k=1)[0]


def generate_sequence(
    model: NGramModel,
    prompt: str = "",
    max_length: int = 25,
    min_length: int = 3,
    temperature: float = 0.8,
    top_k: int | None = 20,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> list[str]:
    """Autoregressively generate a sequence of tokens from a prompt using the n-gram model."""
    rng = random.Random(seed) if seed is not None else random.Random()
    
    prompt_tokens = tokenize(prompt) if prompt.strip() else []
    # Seed history with BOS padding
    history = [BOS] * (model.n - 1) + prompt_tokens
    generated_tokens = list(prompt_tokens)

    for step in range(max_length):
        context = tuple(history[-(model.n - 1):]) if model.n > 1 else ()
        probs = model.get_next_word_probabilities(context)
        
        disallow_eos = len(generated_tokens) < min_length
        next_token = sample_next_token(
            probs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            disallow_eos=disallow_eos,
            rng=rng,
        )

        if next_token == EOS:
            break

        generated_tokens.append(next_token)
        history.append(next_token)

        # Stop if sentence has ended with punctuation and reached min_length
        if len(generated_tokens) >= min_length and next_token in (".", "!", "?"):
            break

    return generated_tokens


def generate_dialogue(
    model: NGramModel,
    prompt: str = "",
    max_length: int = 25,
    min_length: int = 3,
    temperature: float = 0.7,
    top_k: int | None = 10,
    top_p: float | None = 0.9,
    num_samples: int = 1,
    seed: int | None = None,
    save_path: str | Path | None = None,
) -> str | list[str]:
    """Generate coherent game dialogue snippet(s) for NPCs.
    
    Supports:
    - Custom prompt phrase (with optional speaker tag, e.g. 'Guard: Halt')
    - Configurable length (max_length, min_length)
    - Multiple alternative snippets (num_samples > 1)
    - Saving generated dialogue to a text file (save_path)
    """
    speaker_prefix = ""
    prompt_dialogue = prompt.strip()
    if ":" in prompt_dialogue:
        prefix, rest = prompt_dialogue.split(":", 1)
        if rest.strip():
            speaker_prefix = f"{prefix.strip()}: "
            prompt_dialogue = rest.strip()

    snippets: list[str] = []
    base_seed = seed if seed is not None else random.randint(0, 1000000)

    for i in range(max(1, num_samples)):
        sample_seed = base_seed + i if seed is not None else None
        tokens = generate_sequence(
            model=model,
            prompt=prompt_dialogue,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=sample_seed,
        )
        text = detokenize(tokens)
        if speaker_prefix:
            text = f"{speaker_prefix}{text}"
        snippets.append(text)

    # Save to file if path provided
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        content = "\n\n".join(
            f"Snippet {idx + 1}:\n{s}" if len(snippets) > 1 else s
            for idx, s in enumerate(snippets)
        )
        out_file.write_text(content, encoding="utf-8")

    return snippets if num_samples > 1 else snippets[0]


def milestone_3(
    model: NGramModel | None = None,
    prompt: str = "Halt",
    max_length: int = 20,
    num_samples: int = 3,
    seed: int = 42,
) -> dict:
    """Milestone 3: Implement dialogue generation logic using the model.
    
    Returns generated single snippet and multiple alternative snippets.
    """
    if model is None:
        m2 = milestone_2(n=3, k=0.5)
        model = m2["model"]

    # Generate single snippet
    single_snippet = generate_dialogue(
        model=model,
        prompt=prompt,
        max_length=max_length,
        num_samples=1,
        seed=seed,
    )

    # Generate multiple alternative snippets (Stretch idea)
    alternatives = generate_dialogue(
        model=model,
        prompt=prompt,
        max_length=max_length,
        num_samples=num_samples,
        seed=seed,
    )

    return {
        "model": model,
        "prompt": prompt,
        "single_snippet": single_snippet,
        "alternatives": alternatives,
        "num_alternatives": len(alternatives) if isinstance(alternatives, list) else 1,
    }


# --- Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
# Hint: Command-line argument parsing and output formatting.

def milestone_4(
    prompt: str = "Halt",
    length: int = 25,
    samples: int = 2,
    corpus_file: str | None = None,
    save_file: str | None = None,
    temperature: float = 0.7,
    seed: int | None = 42,
) -> dict:
    """Milestone 4: CLI pipeline for dialogue generation.
    
    Generates dialogue from a starting phrase with options for length, alternative
    samples, and saving to text files.
    """
    if corpus_file and Path(corpus_file).exists():
        corpus = Corpus.from_file(corpus_file, n=3)
    else:
        corpus = Corpus(n=3)

    model = NGramModel(n=3, k=0.05)
    model.train(corpus.tokenized_sentences, vocab=corpus.vocab)

    snippets = generate_dialogue(
        model=model,
        prompt=prompt,
        max_length=length,
        num_samples=samples,
        temperature=temperature,
        seed=seed,
        save_path=save_file,
    )

    formatted_output = format_cli_output(
        prompt=prompt,
        snippets=snippets if isinstance(snippets, list) else [snippets],
        save_path=save_file,
    )

    return {
        "prompt": prompt,
        "snippets": snippets,
        "formatted_output": formatted_output,
        "saved_to": save_file,
    }


def format_cli_output(prompt: str, snippets: list[str], save_path: str | Path | None = None) -> str:
    """Format dialogue snippets for command-line presentation."""
    lines = [
        "=" * 50,
        "GAME DIALOGUE ASSISTANT",
        "=" * 50,
        f'Starting Prompt: "{prompt}"',
        "-" * 50,
    ]
    if len(snippets) == 1:
        lines.append(f"Generated Dialogue:\n  {snippets[0]}")
    else:
        lines.append(f"Generated Dialogue ({len(snippets)} alternatives):")
        for idx, snippet in enumerate(snippets, start=1):
            lines.append(f"  [{idx}] {snippet}")
    
    lines.append("-" * 50)
    if save_path:
        lines.append(f"Saved to: {save_path}")
    return "\n".join(lines)
