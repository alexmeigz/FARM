from models.eval import evaluate
from models.google import query_google_snippet, query_google_credible
from models.wikipedia import query_wikipedia
from util.util import (
    augment_snippets,
    baseline_reasoning_prompt,
    contextualized_reasoning_prompt,
    read_base_examples,
    read_contextualized_examples,
    read_foveated_examples,
    read_augmented_examples,
    safety_conditional,
    context_prompt,
)
from util.constants import (
    INVALID_ATTRIBUTION_SOURCE_ERROR,
    INVALID_MODEL_ERROR,
    AttributionSource,
    Model,
    Phase,
    get_enum_values,
)
from models.gpt import check_success, gpt_completion_request
import json
import argparse


def _parse_model(model: str) -> tuple:
    """
    given an argument input model
    returns a 2-tuple (model class, model variant)
        model class: bloom, gpt
        model variant (gpt): ada-001, babbage-001, curie-001, davinci-002, davinci-003, codex, chat
        model variant (bloom): None
    """
    if "gpt" in model or "chat" in model:
        return model.split("_")

    return (model, None)


def _save_json(
    examples: dict,
    phase: str,
    folder: str,
    model_class: str,
    model_variant: str,
    safe: bool,
    num_sources: int = None,
    attribution_source: str = None,
    baseline: bool = False,
) -> None:
    """
    given pipeline arguments
    save json file into mapped file name
    """
    file = open(
        f"""./data/{folder}/{phase}_{model_class}_{model_variant}_{safety_conditional(safe)}{"_snippet" + str(num_sources) if num_sources else ""}{"_" + attribution_source if attribution_source else ""}{"_baseline" if baseline else ""}.json""",
        "w",
    )
    json.dump(examples, file, indent=2)
    file.close()


def _clean_foveation(foveation: str) -> str:
    """
    Given an input foveation,
    remove any newlines and trailing spaces
    """
    return foveation.replace("\n", "").strip(" ")


def baseline_process(
    folder: str,
    safe: bool,
    model_class: str,
    model_variant: str,
    test: bool,
    num_examples: int = 16,
) -> None:
    """
    Phase 0. Baseline rationale generation without leveraging external knowledge.

    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    use test=True to run this step on only a single example.
    outputs rationales in './data/{folder}/baseline_{model_class}_{model_variant}_(un)safe.json'.
    """
    examples = read_base_examples(safe)
    examples = examples[:1] if test else examples

    completion_request = gpt_completion_request

    for sample in examples:
        scenario = baseline_reasoning_prompt(
            prompt=sample["prompt"], advice=sample["advice"], num_examples=num_examples
        )

        sample["explanation"] = completion_request(
            scenario,
            model=f"text-{model_variant}",
            max_tokens=128,
            uncertainty=True,
            stop_tokens=["."],
        )

    _save_json(
        examples=examples,
        phase=Phase.BASELINE.value,
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
    )


def foveation_process(
    folder: str, safe: bool, model_class: str, model_variant: str, test: bool = False
) -> None:
    """
    Phase I. Foveation task. Apply few-shot prompting to foveate on what external knowledge to retreive.

    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    use test=True to run this step on only a single example.
    outputs foveations in './data/{folder}/foveation_{model_class}_{model_variant}_(un)safe.json'.
    """
    examples = read_base_examples(safe=safe)
    examples = examples[:1] if test else examples

    completion_request = (
        gpt_completion_request if model_class == "gpt" else bloom_completion_request
    )

    for sample in examples:
        scenario = context_prompt(
            prompt=sample["prompt"], advice=sample["advice"], few_shot=True
        )

        sample["foveation"] = completion_request(
            scenario,
            model=f"text-{model_variant}",
            max_tokens=256,
            stop_tokens=["Q:", "A:"],
            uncertainty=False,
        )

        # if completion request successful, clean the foveation
        # otherwise, need to manually check for any errors and rerun
        if check_success(sample["foveation"]):
            sample["foveation"][0]["completion"] = _clean_foveation(
                sample["foveation"][0]["completion"]
            )

    _save_json(
        examples=examples,
        phase=Phase.FOVEATION.value,
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
    )


def attribution_process(
    folder: str,
    safe: bool,
    model_class: str,
    model_variant: str,
    attribution_source: str,
    test: bool,
) -> None:
    """
    Phase II. Attribution task. Leverage foveations from step 1 to retreive external knowledge.

    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    use test=True to run this step on only a single example.
    outputs attributions in './data/{folder}/attribution_{model_class}_{model_variant}_(un)safe.json'.
    """
    examples = read_foveated_examples(
        folder=folder, model_class=model_class, model_variant=model_variant, safe=safe
    )
    examples = examples[:2] if test else examples

    # configure external knowledge source
    query_source = None
    if attribution_source == AttributionSource.GOOGLE_VANILLA.value:
        query_source = query_google_snippet
    elif attribution_source == AttributionSource.GOOGLE_CREDIBLE.value:
        query_source = query_google_credible
    elif attribution_source == AttributionSource.WIKIPEDIA.value:
        query_source = query_wikipedia
    else:
        raise INVALID_ATTRIBUTION_SOURCE_ERROR

    for sample in examples:
        sample["attribution"] = query_source(sample["foveation"])

    _save_json(
        examples=examples,
        phase=Phase.ATTRIBUTION.value,
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        attribution_source=attribution_source,
        safe=safe,
    )


def contextualized_reasoning_process(
    folder: str,
    safe: bool,
    model_class: str,
    model_variant: str,
    num_sources: int,
    test: bool,
    attribution_source: str,
    num_examples: int = 16,
) -> None:
    """
    Phase III. Rationalization task. Use augmented external knowledge for in-context inference.

    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    use test=True to run this step on only a single example.
    num_examples dictates the number of few shot examples to use.
    num_sources dictates the number of external sources to augment.
    outputs rationales in './data/{folder}/rationalization_{model_class}_{model_variant}_(un)safe.json'.
    """
    examples = read_augmented_examples(
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
        attribution_source=attribution_source,
    )
    examples = examples[:1] if test else examples

    completion_request = gpt_completion_request

    for sample in examples:
        try:
            sample["context"] = augment_snippets(
                sample["attribution"], num_sources=num_sources
            )
        except Exception as e:
            print(f"ERROR: {e}")
            sample["context"] = e.__str__()

        scenario = contextualized_reasoning_prompt(
            prompt=sample["prompt"],
            advice=sample["advice"],
            context=sample["context"],
            num_sources=num_sources,
            num_examples=num_examples,
        )

        sample["explanation"] = completion_request(
            scenario, model=f"text-{model_variant}", max_tokens=128, uncertainty=True
        )

    _save_json(
        examples=examples,
        phase=Phase.RATIONALIZATION.value,
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
        attribution_source=attribution_source,
        num_sources=num_sources,
    )


def evaluation_process(
    folder: str,
    safe: bool,
    # domain: str,
    model_class: str,
    model_variant: str,
    baseline: bool,
    attribution_source: AttributionSource = None,
    num_sources: int = None,
) -> None:
    """
    Phase IV. Evaluation metrics. Performs automatic evaluation where applicable on rationales.

    use safe=True for safe scenarios; safe=False for unsafe scenarios.
    outputs evaluation results in './data/{folder}/evaluation_{model_class}_{model_variant}_(un)safe.json'.
    """
    examples = read_contextualized_examples(
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
        attribution_source=attribution_source,
        baseline=baseline,
        num_sources=num_sources,
    )

    results = evaluate(examples, safe)

    _save_json(
        examples=results,
        phase=Phase.EVALUATION.value,
        folder=folder,
        model_class=model_class,
        model_variant=model_variant,
        safe=safe,
        attribution_source=attribution_source,
        baseline=baseline,
        num_sources=num_sources,
    )


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="folder name",
    )

    parser.add_argument(
        "-p",
        "--phase",
        type=int,
        choices={0, 1, 2, 3, 4},
        required=True,
        help="{0|1|2|3|4}.. use '0' to run baseline, '1' to run foveation, '2' to run augmentation, '3' to run rationalization, '4' to run evaluation",
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices={"safe", "unsafe", "all"},
        required=True,
        help="{safe|unsafe|all}.. use 'safe' to run safe examples, 'unsafe' to run unsafe examples, 'all' to run all examples",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=get_enum_values(Model),
        required=True,
        help=INVALID_MODEL_ERROR,
    )

    parser.add_argument(
        "-s",
        "--num_sources",
        type=int,
        choices=range(1, 10),
        metavar="[1-10]",
        required=False,
        help="{1..10}",
    )

    parser.add_argument(
        "-a",
        "--attribution_source",
        type=str,
        required=False,
        choices=get_enum_values(AttributionSource),
        help=INVALID_ATTRIBUTION_SOURCE_ERROR,
    )

    parser.add_argument(
        "-e",
        "--num_examples",
        type=int,
        choices=range(0, 17),
        metavar="[0-16]",
        required=False,
        default=16,
        help="{0..16}.",
    )

    parser.add_argument("--test", action="store_true")
    parser.set_defaults(test=False)

    parser.add_argument("--baseline", action="store_true")
    parser.set_defaults(baseline=False)

    args = parser.parse_args()
    model_class, model_variant = _parse_model(args.model)
    print("Arguments parsed correctly.")

    # ----- STAGE 0 -----
    if args.phase == 0:
        if args.type in ["unsafe", "all"]:
            baseline_process(
                folder=args.folder,
                safe=False,
                model_class=model_class,
                model_variant=model_variant,
                num_examples=args.num_examples,
                test=args.test,
            )
        if args.type in ["safe", "all"]:
            baseline_process(
                folder=args.folder,
                safe=True,
                model_class=model_class,
                model_variant=model_variant,
                num_examples=args.num_examples,
                test=args.test,
            )

    # ----- STAGE 1 -----
    elif args.phase == 1:
        if args.type in ["unsafe", "all"]:
            foveation_process(
                folder=args.folder,
                safe=False,
                model_class=model_class,
                model_variant=model_variant,
                test=args.test,
            )
        if args.type in ["safe", "all"]:
            foveation_process(
                folder=args.folder,
                safe=True,
                model_class=model_class,
                model_variant=model_variant,
                test=args.test,
            )

    # ----- STAGE 2 -----
    elif args.phase == 2:
        assert args.attribution_source, INVALID_ATTRIBUTION_SOURCE_ERROR

        if args.type in ["unsafe", "all"]:
            attribution_process(
                folder=args.folder,
                safe=False,
                model_class=model_class,
                model_variant=model_variant,
                attribution_source=args.attribution_source,
                test=args.test,
            )
        if args.type in ["safe", "all"]:
            attribution_process(
                folder=args.folder,
                safe=True,
                model_class=model_class,
                model_variant=model_variant,
                attribution_source=args.attribution_source,
                test=args.test,
            )

    # ----- STAGE 3 -----
    elif args.phase == 3:
        assert (
            args.num_sources
        ), f"Must set -n --num_sources flag for rationalization step"
        assert args.attribution_source, INVALID_ATTRIBUTION_SOURCE_ERROR

        if args.type in ["unsafe", "all"]:
            contextualized_reasoning_process(
                folder=args.folder,
                safe=False,
                model_class=model_class,
                model_variant=model_variant,
                num_sources=args.num_sources,
                num_examples=args.num_examples,
                attribution_source=args.attribution_source,
                test=args.test,
            )
        if args.type in ["safe", "all"]:
            contextualized_reasoning_process(
                folder=args.folder,
                safe=True,
                model_class=model_class,
                model_variant=model_variant,
                num_sources=args.num_sources,
                num_examples=args.num_examples,
                attribution_source=args.attribution_source,
                test=args.test,
            )

    # ----- STAGE 4 -----
    elif args.phase == 4:
        if args.type in ["unsafe", "all"]:
            evaluation_process(
                folder=args.folder,
                safe=False,
                model_class=model_class,
                model_variant=model_variant,
                num_sources=args.num_sources,
                attribution_source=args.attribution_source,
                baseline=args.baseline,
            )
        if args.type in ["safe", "all"]:
            evaluation_process(
                folder=args.folder,
                safe=True,
                model_class=model_class,
                model_variant=model_variant,
                num_sources=args.num_sources,
                attribution_source=args.attribution_source,
                baseline=args.baseline,
            )

    # ----- INVALID INPUT -----
    else:
        print("Argument parsing is set up incorrectly.")
