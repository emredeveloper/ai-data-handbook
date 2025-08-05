import asyncio
from pathlib import Path
import shutil

from openai import AsyncOpenAI
from agents import (
    Agent,
    ItemHelpers,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    function_tool,
)
from agents.mcp import MCPServerStdio
from pydantic import BaseModel


class WeatherParams(BaseModel):
    location: str


async def prompt_user(question: str) -> str:
    """Async input prompt function"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, question)


async def main():
    # Set up OpenAI client for local server (e.g., Ollama)
    openai_client = AsyncOpenAI(
        api_key="local",
        base_url="http://localhost:11434/v1",
        timeout=30.0,  # Add timeout for requests
    )

    # Get current working directory
    samples_dir = str(Path.cwd())

    # Create MCP server for filesystem operations
    mcp_server = MCPServerStdio(
        name="Filesystem MCP Server, via npx",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                samples_dir,
            ],
        },
    )

    # Connect to MCP server with error handling
    try:
        print("Connecting to MCP server...")
        await mcp_server.connect()
        print("MCP server connected successfully!")
    except Exception as e:
        print(f"Error initializing MCP server: {e}")
        print("Continuing without MCP server...")
        mcp_server = None

    # Configure agents SDK
    set_tracing_disabled(True)
    set_default_openai_client(openai_client)
    set_default_openai_api("chat_completions")

    # Define weather tool
    @function_tool
    async def search_tool(location: str) -> str:
        return f"The weather in {location} is sunny."

    # Create agent
    mcp_servers = [mcp_server] if mcp_server else []
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant.",
        tools=[search_tool],
        model="gpt-oss:20b",
        mcp_servers=mcp_servers,
    )

    # Get user input
    user_input = await prompt_user("> ")
    print(f"User input received: {user_input}")

    # Test Ollama connection first
    print("Testing Ollama connection...")
    try:
        test_response = await openai_client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        print("Ollama is responding!")
    except Exception as e:
        print(f"Ollama connection test failed: {e}")
        return

    # Run agent with streaming
    print("Starting agent...")
    result = Runner.run_streamed(agent, user_input)

    # Process streaming results
    try:
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                elif event.item.type == "message_output_item":
                    print(
                        f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                    )
                else:
                    print(f"-- Unknown event type: {event.item.type}")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("=== Run complete ===")


if __name__ == "__main__":

    if not shutil.which("npx"):
        raise RuntimeError(
            "npx is not installed. Please install it with `npm install -g npx`."
        )
    asyncio.run(main())