# Imports
import os
import openai
import numpy as np
from dotenv import load_dotenv

from util.constants import DEFAULT_GPT_MODEL

# Handle environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def check_success(response: any) -> bool:
    """
    checks whether a GPT response was successful
    """
    return type(response) == list and len(response) > 0


def gpt_completion_request(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0,
    top_p: float = 1,
    model: str = DEFAULT_GPT_MODEL,
    stop_tokens: list = None,
    uncertainty: bool = False,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    **kwargs,
) -> str:
    """
    given a prompt, query gpt-3 as completion task to generate a response and return list of responses.
    max_tokens denotes max response length.
    temperature denotes added randomness in abstractive generation.
    """
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            logprobs=5,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    except Exception as e:
        print(f"ERROR: {e}")

        return {
            "error": e.__str__(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "model": model,
            "stop_tokens": stop_tokens,
            "uncertainty": uncertainty,
        }

    # only return completion in base case
    if not uncertainty:
        return response["choices"][0]["text"].strip(" .")

    # return completion and uncertainty calculations
    result = list()
    for item in response["choices"]:
        stop_index = max_tokens

        # stop computation at the stop token if it exists
        try:
            stop_index = item["logprobs"]["tokens"].index("<|endoftext|>")
        except:
            pass

        result.append(
            {
                "completion": item["text"].strip(" ."),
                "log_probability": np.sum(
                    item["logprobs"]["token_logprobs"][:stop_index]
                ),
                "first_token_distribution": dict(item["logprobs"]["top_logprobs"][0]),
            }
        )

    return result
