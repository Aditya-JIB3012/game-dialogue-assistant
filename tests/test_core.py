"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app import core


# Milestone 1: Collect and preprocess a corpus of game dialogues.
def test_milestone_1():
    raw_line = "Guard: Halt! Who goes there? State your business."
    cleaned = core.clean_dialogue_line(raw_line)
    assert cleaned == "Halt! Who goes there? State your business."
    
    tokens = core.tokenize(cleaned)
    assert "halt" in tokens
    assert "!" in tokens
    assert "who" in tokens
    
    padded = core.pad_sequence(tokens, n=3)
    assert padded[:2] == [core.BOS, core.BOS]
    assert padded[-1] == core.EOS

    reconstructed = core.detokenize(["halt", "!", "who", "goes", "there", "?"])
    assert reconstructed == "Halt! Who goes there?"

    res = core.milestone_1()
    assert isinstance(res, dict)
    assert "tokenized_sentences" in res
    assert "vocab" in res
    assert "stats" in res
    assert res["stats"]["num_dialogues"] >= 40
    assert res["stats"]["vocab_size"] > 50
    assert core.BOS in res["vocab"]
    assert core.EOS in res["vocab"]
    assert core.UNK in res["vocab"]


# Milestone 2: Build the n-gram language model with smoothing.
def test_milestone_2(tmp_path):
    res = core.milestone_2(n=3, k=1.0)
    assert isinstance(res, dict)
    assert "model" in res
    model = res["model"]
    
    # 1. Probabilities sum to 1.0
    assert res["prob_sum"] == pytest.approx(1.0, abs=1e-5)
    
    # 2. Probability sum on unseen context must also sum to 1.0
    unseen_context = ("completely", "unseen")
    unseen_probs = model.get_next_word_probabilities(unseen_context)
    assert sum(unseen_probs.values()) == pytest.approx(1.0, abs=1e-5)
    
    # 3. Add-k smoothing ensures non-zero probability for all vocabulary words
    for word in model.target_vocab:
        assert unseen_probs[word] > 0.0

    # 4. Frequent transitions have higher probability than unseen ones
    seen_prob = model.get_probability(("who", "goes"), "there")
    random_prob = model.get_probability(("who", "goes"), "potions")
    assert seen_prob > random_prob

    # 5. Perplexity is positive and finite
    assert 1.0 < res["perplexity"] < 500.0

    # 6. Model serialization and deserialization
    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded_model = core.NGramModel.load(save_path)
    assert loaded_model.n == model.n
    assert loaded_model.k == model.k
    assert loaded_model.get_probability(("who", "goes"), "there") == pytest.approx(seen_prob)


# Milestone 3: Implement dialogue generation logic using the model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
