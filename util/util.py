import json
from enum import Enum
import random

random.seed(69)

from models.gpt import check_success, gpt_completion_request
from util.constants import AttributionSource, Domain


# one-time load global constants
FEW_SHOT_FOVEATIONS = None
FEW_SHOT_EXPLANATIONS = None
CHAT_FEW_SHOT_EXPLANATIONS = None
FEW_SHOT_AUGMENTATIONS = None


def safety_conditional(safe: bool) -> str:
    return "safe" if safe else "unsafe"


def _rephrase_advice(advice: str) -> str:
    """
    rephrases advice to start with an action verb
    (i.e., rephrases "do not" or "don't")
    """
    return advice.replace("don't", "not").replace("do not", "not")


def _extract_source(url: str) -> str:
    """
    extracts website source from a google snippet url
    (i.e., returns scubadiverlife.com from https://scubadiverlife.com/prevent-vertigo-scuba-diving/)
    """
    return url.replace("https://", "").replace("http://", "").split("/")[0]


def _read_few_shot_foveations(k: int = 16) -> str:
    """
    returns k-shot foveations
    """
    examples = json.load(open(f"./data/few_shot/foveation.json", "r"))
    return (
        "\n\n".join(
            [
                f"""{context_prompt(prompt=examples[i]["prompt"], advice=examples[i]["advice"], few_shot=False)} {examples[i]["foveation"][0]["completion"]}"""
                for i in range(k)
            ]
        )
        + "\n\n"
    )


def _read_few_shot_explanations(
    augment: bool,
    num_sources: int = None,
    num_examples: int = 8,
) -> str:
    """
    return k-shot explanations
    """
    examples = json.load(open(f"./data/few_shot/rationalization.json", "r"))

    snippets = None

    if augment:
        assert num_sources, f"num_sources is None"

        snippets = [
            f"""{augment_snippets(snippets=examples[i]["attribution"], num_sources=num_sources)}\nQ: {base_scenario(prompt=examples[i]["prompt"], advice=examples[i]["advice"])}\nA: {examples[i]["explanation"][0]["completion"]}"""
            for i in range(num_examples)
        ]

    else:
        snippets = [
            f"""Q: {base_scenario(prompt=examples[i]["prompt"], advice=examples[i]["advice"])}\nA: {examples[i]["explanation"][0]["completion"]}"""
            for i in range(num_examples)
        ]

    return "\n\n".join(snippets) + "\n\n"


def _chat_few_shot_explanations(
    augment: bool,
    num_sources: int = None,
    num_examples: int = 2,
):
    """
    return k-shot explanations
    """
    examples = json.load(open(f"./data/few_shot/rationalization.json", "r"))
    messages = list()

    for i in range(num_examples):
        messages.append(
            {
                "role": "user",
                "content": f"""Q: {base_scenario(prompt=examples[i]["prompt"], advice=examples[i]["advice"])}""",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"""A: {examples[i]["explanation"][0]["completion"]}""",
            }
        )

    return messages


def read_base_examples(safe: bool, domain: Domain = None) -> list:
    """
    reads safe/unsafe base scenarios and returns parsed scenarios in list format.
    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    """
    examples = json.load(
        open(f"./data/safetext/{safety_conditional(safe)}_samples.json", "r")
    )

    if domain == Domain.ALL.value or domain is None:
        return examples

    # filter for domain specific examples
    filtered_examples = list()

    for example in examples:
        if example["domain"] == domain:
            filtered_examples.append(example)

    random.shuffle(filtered_examples)

    return filtered_examples


def read_foveated_examples(
    folder: str,
    model_class: str,
    model_variant: str,
    safe: bool,
) -> list:
    """
    reads safe/unsafe foveated scenarios and returns parsed scenarios in list format.
    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    """
    return json.load(
        open(
            f"./data/{folder}/foveation_{model_class}_{model_variant}_{safety_conditional(safe)}.json",
            "r",
        )
    )


def read_augmented_examples(
    folder: str,
    model_class: str,
    model_variant: str,
    safe: bool,
    attribution_source: str,
) -> list:
    """
    reads safe/unsafe augmented scenarios and returns parsed scenarios in list format.
    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    """
    return json.load(
        open(
            f"./data/{folder}/attribution_{model_class}_{model_variant}_{safety_conditional(safe)}_{attribution_source}.json",
            "r",
        )
    )


def read_contextualized_examples(
    folder: str,
    # domain: str,
    model_class: str,
    model_variant: str,
    safe: bool,
    baseline: bool,
    attribution_source: AttributionSource = None,
    num_sources: int = None,
) -> list:
    """
    reads safe/unsafe contextualized scenarios and returns parsed scenarios in list format.
    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    """
    return json.load(
        open(
            f"./data/{folder}/{'baseline' if baseline else 'rationalization'}_{model_class}_{model_variant}_{safety_conditional(safe)}{'_snippet' + str(num_sources) if num_sources else ''}{'_' + attribution_source if attribution_source else ''}.json",
            "r",
        )
    )


def base_scenario(prompt: str, advice: str) -> str:
    """
    returns formatted baseline question of interest
    f"{prompt}, should you {advice}?"
    """
    return f"""{prompt}, should you {_rephrase_advice(advice)}?"""


def zero_shot_reasoning_prompt(prompt: str, action: str, safe: bool) -> str:
    """
    generates the zero shot reasoning prompt given the prompt, action, and class of text.
    TODO
    """
    return f"""{base_scenario(prompt, action, safe)} Let's think step by step."""


def context_prompt(prompt: str, advice: str, few_shot: bool = True, k: int = 16) -> str:
    """
    generates a context asking prompt given the prompt, action, and class of text.
    TODO
    """
    global FEW_SHOT_FOVEATIONS
    if few_shot and not FEW_SHOT_FOVEATIONS:
        FEW_SHOT_FOVEATIONS = _read_few_shot_foveations(k)

    return f"""{FEW_SHOT_FOVEATIONS if few_shot else ""}Q: To answer, "{base_scenario(prompt, advice)}" what do we first need context about?\nA:"""


def augment_snippets(snippets: list, num_sources: int) -> str:
    """
    returns data augmentation of top k snippets
    applies summarization when summarize = True
    """
    return "; ".join(
        [
            f"""{_extract_source(s["source"])}: {s["content"][:600]}"""
            for s in snippets[:num_sources]
        ]
    )


def chat_reasoning_prompt(
    prompt: str, advice: str, few_shot=True, num_examples: int = 4
) -> str:
    """
    generates an explanation-asking prompt given the prompt, action, and type of text.
    """
    global CHAT_FEW_SHOT_EXPLANATIONS
    if few_shot and not CHAT_FEW_SHOT_EXPLANATIONS:
        CHAT_FEW_SHOT_EXPLANATIONS = _chat_few_shot_explanations(
            augment=False, num_examples=num_examples
        )

    return CHAT_FEW_SHOT_EXPLANATIONS + [
        {
            "role": "system",
            "content": f"""Q: {base_scenario(prompt=prompt, advice=advice)}""",
        }
    ]


def baseline_reasoning_prompt(
    prompt: str, advice: str, few_shot=True, num_examples: int = 4
) -> str:
    """
    generates an explanation-asking prompt given the prompt, action, and type of text.
    """
    global FEW_SHOT_EXPLANATIONS
    if few_shot and not FEW_SHOT_EXPLANATIONS:
        FEW_SHOT_EXPLANATIONS = _read_few_shot_explanations(
            augment=False, num_examples=num_examples
        )

    return f"""{FEW_SHOT_EXPLANATIONS if few_shot else ""}Q: {base_scenario(prompt=prompt, advice=advice)}\nA:"""


def contextualized_reasoning_prompt(
    prompt: str,
    advice: str,
    context: str,
    num_sources: int,
    few_shot: bool = True,
    num_examples: int = 16,
) -> str:
    """
    generates an explanation-asking prompt given the prompt, advice, foveation, context, and type of text.
    """
    global FEW_SHOT_AUGMENTATIONS
    if few_shot and not FEW_SHOT_AUGMENTATIONS:
        FEW_SHOT_AUGMENTATIONS = _read_few_shot_explanations(
            augment=True,
            num_examples=num_examples,
            num_sources=num_sources,
        )

    return f"""{FEW_SHOT_AUGMENTATIONS if few_shot else ""}{context}\nQ: {base_scenario(prompt=prompt, advice=advice)}\nA:"""
