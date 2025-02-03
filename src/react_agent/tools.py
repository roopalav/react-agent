"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, Tool
from typing_extensions import Annotated
from react_agent.configuration import Configuration
from react_agent.retriever import vector_store_manager
import os
import requests
from typing import Any, Callable, List, Optional, cast
from react_agent.utils import load_cache, save_cache
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

structured_prompt = PromptTemplate.from_template(
    """
You are a structured assistant. Format the following response according to its type (table, list, or paragraph).

### {title}

{content}

If the response contains tabular data, present it as a table with proper column headers and values.

If the response contains multiple points, present them as bullet points or a numbered list, depending on context.

Ensure the response is **clear, concise, and easy to read**.

Examples:
1. For tabular data:                                                  
    +------------+----------------+---------------+---------------+---------------+
    | Date       | Condition      | Max Temp (°C) | Min Temp (°C) | Rainfall (mm) |
    +------------+----------------+---------------+---------------+---------------+
    | 2025-02-02 | Mist/Haze      | 32            | 22            | 0.0           |
    +------------+----------------+---------------+---------------+---------------+
    | 2025-02-03 | Partly Cloudy  | 32            | 22            | 0.0           |
    +------------+----------------+---------------+---------------+---------------+
   
2. For a list:
   - Point 1
   - Point 2
   - Point 3

After the output, include the source link(s) where the information was obtained.

**Source(s):**
{sources}

If the results include tweets, format them as follows:
1. For **multiple tweets**, present them in a list format (as shown in the list example).
2. For **a single tweet**, summarize the tweet’s content in a paragraph, highlighting key details such as author, tweet content, and location.

Example output with tweets:

- **Tweet 1**: Author ID: 123456789, Tweet: "Sample tweet", Location: Madhurai, Date: 2025-02-02
- **Tweet 2**: Author ID: 987654321, Tweet: "Another tweet", Location: Chennai, Tamil Nadu, Date: 2025-02-03                                                                                                  
"""
)


def format_response(input_data: dict) -> str:
    """Formats responses into structured headings and bullet points."""
    return structured_prompt.format(**input_data)


format_tool = Tool(
    name="format_response_tool",
    func=format_response,
    description="Formats responses into structured headings and bullet points for better readability.",
)


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results from specific trusted sites like IMD,INCOIS,Tnsmart

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    params = {
        "query": query,
        "search_depth": "advanced",
        "include_domains": [
            "mausam.imd.gov.in",
            "aws.imd.gov.in",
            "beta-tnsmart.rimes.int",
        ],
        "exclude_domains": [
            "weatherapi.com",
            "weathertab.com",
            "weather2travel.com",
            "world-weather.info",
            "weather-atlas.com",
            "weather25",
            "en.climate-data.org",
            "wisemeteo.com",
            "easeweather.com",
        ],
    }
    result = await wrapped.ainvoke(params)
    # result = await wrapped.ainvoke({"query": query,"include_domains": ["mausam.imd.gov.in","aws.imd.gov.in","beta-tnsmart.rimes.int"]})
    return cast(list[dict[str, Any]], result)


async def retrieve(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Retrieve semantically similar documents from the vector store.

    This tool is designed to fetch relevant documents based on semantic similarity,
    useful for answering domain-specific or context-aware questions.
    """
    retriever = vector_store_manager.get_retriever()
    results = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in results)


async def twitter_search_tool(query: str) -> str:
    """Fetches recent tweets based on the query.

    Uses the Twitter API to search for recent tweets and caches them to avoid duplication.
    """
    # configuration = Configuration.from_runnable_config(config)
    load_dotenv()
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return "Twitter API credentials are missing."

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": 10,
        "tweet.fields": "id,created_at,author_id,text,geo",
        "expansions": "geo.place_id",
        "place.fields": "full_name,country",
        "user.fields": "location",
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.json()}"

    tweets = response.json().get("data", [])
    places = {p["id"]: p for p in response.json().get("includes", {}).get("places", [])}
    users = {
        user["id"]: user
        for user in response.json().get("includes", {}).get("users", [])
    }
    if not tweets:
        return "No tweets found."

    # Load and check cache
    cache = load_cache()
    existing_tweets = cache.get("twitter", {})
    existing_tweet_ids = {tweet["id"] for tweet in cache.get("twitter", {}).values()}
    new_tweets = [tweet for tweet in tweets if tweet["id"] not in existing_tweet_ids]

    # Update cache if there are new tweets
    if new_tweets:
        cache.setdefault("twitter", {}).update(
            {tweet["id"]: tweet for tweet in new_tweets}
        )
        save_cache(cache)

    all_tweets = list(existing_tweets.values()) + new_tweets

    formatted_tweets = []
    for tweet in all_tweets:
        place_info = "Location: Unknown"
        user_location = users.get(tweet["author_id"], {}).get("location", "Unknown")

        # If tweet geo information is available, use that
        if tweet.get("geo"):
            place_id = tweet["geo"]["place_id"]
            place_info = f"Location: {places[place_id]['full_name']}, {places[place_id]['country']}"
        # Otherwise, use user profile location if available
        elif user_location != "Unknown":
            place_info = f"User's Location: {user_location}"

        formatted_tweets.append(
            f"Author ID: {tweet['author_id']}\nTweet: {tweet['text']}\n{place_info}\nDate: {tweet['created_at']}"
        )

    return "\n\n".join(formatted_tweets)


TOOLS: List[Callable[..., Any]] = [search, retrieve, format_tool]
