"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from vector_store_manager import vector_store_manager


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
    result = await wrapped.ainvoke({"query": query})
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


TOOLS: List[Callable[..., Any]] = [search,retrieve]
