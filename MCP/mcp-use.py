import asyncio
import json
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

async def main():
    # browser_mcp.json dosyasından config'i oku
    with open("browser_mcp.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    client = MCPClient.from_dict(config)
    
    # Initialize local Ollama model
    llm = ChatOllama(
        model="qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.7
    )
    
    # Create agent
    agent = MCPAgent(llm=llm, client=client)
    
    # Use the agent
    result = await agent.run("List the files in the current directory")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())