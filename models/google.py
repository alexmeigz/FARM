import os
from dotenv import load_dotenv
import requests
from util.constants import (
    CREDIBLE_DOMAINS,
    SERP_ENDPOINT,
    SERP_PARAMS,
)
from util.attribution import format_query

# Handle environment
load_dotenv()


def serp_search(query: str):
    """
    queries Google for front page results using SERP API
    """
    response = requests.get(
        SERP_ENDPOINT,
        params={**SERP_PARAMS, "q": query, "api_key": os.getenv("SERP_API_KEY")},
    )

    # return text if html successful
    assert response.status_code == 200, f"Response status {response.status_code}"
    return response.json()


def query_google_snippet(foveation: str, credible: bool = False):
    """
    invokes google search with the input foveation as query
    returns a list of front page google snippets and associated sources
    """
    try:
        # format google search query
        query = format_query(f"""{CREDIBLE_DOMAINS if credible else ""} {foveation}""")

        # parse and return front page snippets from Google
        return parse_serp_snippets(serp_search(query))

    except Exception as e:
        # handle potential errors
        print(f"ERROR {e} for foveation: {foveation}")
        return {
            "error": e.__str__(),
            "google_query": query,
        }


def query_google_credible(foveation: str):
    """
    involves google search with added constaint that only uses credible domains as an attribution source
    returns a list of front page google snippets and associated sources
    """
    return query_google_snippet(foveation=foveation, credible=True)


def parse_serp_snippets(response: dict) -> list:
    """
    parse source and content of front page snippets
    """
    results = list()

    for item in response["organic_results"]:
        if "snippet" in item:
            results.append(
                {
                    "source": item["link"],
                    "content": f"{item['title']}. {item['snippet']}",
                }
            )

    return results


def parse_serp_wikipages(response: dict) -> list:
    """
    parse source and title of front page wikipedia results on Google
    """
    return [
        {
            "source": item["link"],
            "title": item["title"],
        }
        for item in response["organic_results"]
    ]
