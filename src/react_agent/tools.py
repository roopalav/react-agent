"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg,Tool
from typing_extensions import Annotated
from react_agent.configuration import Configuration
from react_agent.retriever import vector_store_manager
import os
import requests
from typing import Any, Callable, List, Optional, cast
from react_agent.utils import load_cache, save_cache
from langchain_core.prompts import PromptTemplate


structured_prompt = PromptTemplate.from_template("""
You are a structured assistant. Format the following response using proper headings and bullet points.

### {title}

#### {section1_title}
- {section1_bullet1}
- {section1_bullet2}
- {section1_bullet3}

#### {section2_title}
- {section2_bullet1}
- {section2_bullet2}
- {section2_bullet3}

#### {section3_title}
- {section3_bullet1}
- {section3_bullet2}

Ensure the response is **clear, concise, and easy to read**.
""")


def format_response(input_data: dict) -> str:
    """Formats responses into structured headings and bullet points."""
    return structured_prompt.format(**input_data)

format_tool = Tool(
    name="format_response_tool",
    func=format_response,
    description="Formats responses into structured headings and bullet points for better readability."
)

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query,"include_domains": ["mausam.imd.gov.in","aws.imd.gov.in","beta-tnsmart.rimes.int"]})
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

async def twitter_search_tool(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Fetches recent tweets based on the query.

    Uses the Twitter API to search for recent tweets and caches them to avoid duplication.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return "Twitter API credentials are missing."

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": 10,
        "tweet.fields": "id,created_at,author_id,text"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.json()}"

    tweets = response.json().get("data", [])
    if not tweets:
        return "No tweets found."

    # Load and check cache
    cache = load_cache()
    existing_tweets = cache.get("twitter", {})
    existing_tweet_ids = {tweet["id"] for tweet in cache.get("twitter", {}).values()}
    new_tweets = [tweet for tweet in tweets if tweet["id"] not in existing_tweet_ids]

    # Update cache if there are new tweets
    if new_tweets:
        cache.setdefault("twitter", {}).update({tweet["id"]: tweet for tweet in new_tweets})
        save_cache(cache)

    all_tweets = list(existing_tweets.values()) + new_tweets    

    # Format and return new tweets
    return "\n\n".join(
        f"Author ID: {tweet['author_id']}\nTweet: {tweet['text']}\nDate: {tweet['created_at']}"
        for tweet in all_tweets
    ) if all_tweets else "No tweets found."

TOOLS: List[Callable[..., Any]] = [search,retrieve,format_tool]