import requests
from models.google import serp_search, parse_serp_wikipages
from util.attribution import format_query

from util.constants import WIKIPEDIA_DOMAIN, WIKIPEDIA_ENDPOINT, WIKIPEDIA_HEADERS


def _clean_title(title: str) -> str:
    """
    removes extrenous information from the title
    """
    return title.split("-")[0].strip().split("–")[0].strip()


def query_wikipedia(foveation: str):
    """
    invokes google search with the input foveation as query using only Wikipedia articles
    returns a list of wikipedia articles and associated abstracts
    """
    # find wikipedia sources with relevant matches
    sources = get_wikipedia_sources(foveation)

    # for each page, extract the abstract and return attributions
    if isinstance(sources, list):
        results = list()
        for source in sources:
            abstract = get_wikipedia_abstract(_clean_title(source["title"]))
            if isinstance(abstract, str):
                results.append(
                    {
                        "source": source["source"],
                        "content": abstract,
                    }
                )
        return results

    # return raised error if not list
    return sources


def get_wikipedia_sources(foveation: str) -> list:
    """
    search for wikipedia sources via Google
    """
    try:
        # format google search query
        query = format_query(f"""{WIKIPEDIA_DOMAIN} {foveation}""")

        # parse and return front page Wikipedia sources from Google
        result = parse_serp_wikipages(serp_search(query))
        return result

    except Exception as e:
        # handle potential errors
        print(f"ERROR {e} for foveation: {foveation}")
        return {
            "error": e.__str__(),
            "google_query": query,
        }


def get_wikipedia_abstract(title: str) -> str:
    """
    returns abstract of a given wikipedia page title
    """
    try:
        # query wikipedia endpoint for response
        response = requests.get(
            WIKIPEDIA_ENDPOINT, params={**WIKIPEDIA_HEADERS, "titles": title}
        ).json()

        # output possible warnings
        if "warnings" in response:
            print(response["warnings"])

        # return first abstract if exists
        return next(iter(response["query"]["pages"].values()))["extract"]

    except Exception as e:
        # handle potential errors
        print(f"ERROR {e} for query: {title}")
        return {
            "error": e.__str__(),
            "wikipedia_query": title,
        }
