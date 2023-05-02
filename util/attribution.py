def format_query(foveation: str) -> str:
    """
    formats the query correctly for google search
    """
    return foveation.replace('"', "").replace("'", "")
