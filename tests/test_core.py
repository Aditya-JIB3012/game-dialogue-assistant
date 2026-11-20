"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app import core


# Milestone 1: Collect and preprocess a corpus of game dialogues.
def test_milestone_1():
    # Test tokenization and cleaning
    raw_line = "Guard: Halt! Who goes there? State your business."
    cleaned = core.clean_dialogue_line(raw_line)
    assert cleaned == "Halt! Who goes there? State your business."
    
    tokens = core.tokenize(cleaned)
    assert "halt" in tokens
    assert "!" in tokens
    assert "who" in tokens
    
    # Test padding
    padded = core.pad_sequence(tokens, n=3)
    assert padded[:2] == [core.BOS, core.BOS]
    assert padded[-1] == core.EOS

    # Test detokenization
    reconstructed = core.detokenize(["halt", "!", "who", "goes", "there", "?"])
    assert reconstructed == "Halt! Who goes there?"

    # Test running milestone_1()
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement dialogue generation logic using the model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
