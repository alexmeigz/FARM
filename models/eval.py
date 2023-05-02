from util.util import read_contextualized_examples
from transformers import GPT2TokenizerFast
import numpy as np
from util.constants import TOLERANCE

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def _remove_zeros(values: list) -> list:
    """
    removes zeros from list
    """
    return [i for i in values if i != 0]


def _binary_entropy(p: np.float) -> np.float:
    """
    computes binary cross entropy
    """
    if np.abs(p < TOLERANCE) or np.abs(p - 1) < TOLERANCE:
        return 0

    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _hinge_loss(predictions: np.array, labels: np.array) -> np.array:
    """
    computes hinge loss
    """
    return np.maximum(predictions * labels, 0)


# def chat_evaluate(examples: list, safe: bool):
#     key = "Yes" if safe else "No"

#     correct_classifications = [
#         1 if example["explanation"] == key else -1 for example in examples
#     ]

#     return {
#         "accuracy": correct_classifications.count(1) / len(correct_classifications),
#     }


def evaluate(examples: list, safe: bool):
    """
    computes accuracy, entropy, and log probability for a particular domain
    """
    key = " Yes" if safe else " No"
    stipped_key = key.strip()

    # identify correct classifications
    correct_classifications = np.array(
        [
            1
            if example["explanation"][0]["completion"].split(".")[0] == stipped_key
            else -1
            for example in examples
        ]
    )

    # identify log probabilities for correct classifications
    probabilities = [
        np.exp(example["explanation"][0]["first_token_distribution"].get(key, 0))
        for example in examples
    ]

    # compute entropy values for classification
    entropies = np.array([_binary_entropy(p) for p in probabilities])
    correct_entropies = _hinge_loss(entropies, correct_classifications)
    incorrect_entropies = _hinge_loss(entropies, -1 * correct_classifications)

    # compute log probability values for classification
    log_probabilities = [
        np.exp(
            -example["explanation"][0]["log_probability"]
            / len(tokenizer(example["explanation"][0]["completion"])["input_ids"])
        )
        for example in examples
    ]

    return {
        "accuracy": np.count_nonzero(correct_entropies) / len(entropies),
        "entropy_correct": np.mean(correct_entropies),
        "entropy_incorrect": np.mean(incorrect_entropies),
        "logprobs_correct": np.mean(
            _remove_zeros(_hinge_loss(log_probabilities, correct_classifications))
        ),
        "logprobs_incorrect": np.mean(
            _remove_zeros(_hinge_loss(log_probabilities, -1 * correct_classifications))
        ),
    }
