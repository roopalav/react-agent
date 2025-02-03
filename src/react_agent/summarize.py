import os
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from react_agent.utils import load_chat_model
from react_agent.state import InputState, State
from react_agent.tools import twitter_search_tool


# Step 1: Fetch tweets using Twitter tool
async def fetch_tweets(state: State) -> dict:
    """Fetches relevant tweets before summarization."""
    last_message = state.messages[-1].content  # User query

    # Call the async Twitter search tool
    tweets = await twitter_search_tool(last_message)

    return {"tweets": tweets}


# Step 2: Summarize the fetched tweets
async def summarize_tweets(state: State) -> dict:
    """Summarizes fetched tweets in a structured format."""
    model = load_chat_model("gpt-4-turbo")  # Use your preferred model
    tweets = state.tweets  # Get tweets from the previous step

    if not tweets or tweets == "No tweets found.":
        return {
            "messages": [AIMessage(content="No relevant tweets found to summarize.")]
        }

    structured_prompt = f"""
You are an AI assistant summarizing recent tweets related to the topic.
Extract key insights and trends while ensuring accuracy.

📌 **Summary of Recent Tweets**
🔹 **Topic:** Identify the main theme.
🔹 **Total Tweets Analyzed:** {len(tweets)}

🚀 **Key Highlights:**
- What are the most important takeaways?
- Any significant hashtags or mentions?
- Are there common concerns or requests?

📍 **Geographical Insights:**
- Where are most tweets coming from?
- Mention specific locations if available.

📊 **Trends & Patterns:**
- Most frequently mentioned words?
- Any emergency alerts or critical updates?

📢 **Urgent Alerts:**
- Are there any actionable tweets (e.g., rescue requests, warnings)?
- If yes, summarize them.

Now, summarize the following tweets:
{tweets}
"""

    response = await model.ainvoke(structured_prompt)

    return {"messages": [AIMessage(content=response)]}


# Build the summarization graph
builder = StateGraph(State, input=InputState)

# Add nodes
builder.add_node("fetch_tweets", fetch_tweets)
builder.add_node("summarizer", summarize_tweets)

# Define flow
builder.add_edge("__start__", "fetch_tweets")
builder.add_edge("fetch_tweets", "summarizer")
builder.add_edge("summarizer", "__end__")

# Compile Summarization Graph
summarization_graph = builder.compile()
summarization_graph.name = "Twitter Summarization Agent"
