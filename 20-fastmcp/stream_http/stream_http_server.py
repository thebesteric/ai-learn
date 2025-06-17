from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(
    name="HelpfulAssistant",
    instructions="""
    This server provides some tools.
    """,
)


@mcp.tool(
    name="multiply",
    description="Multiplies two numbers together."
)
def multiply(a: Annotated[float, Field(description="the argument named a")],
             b: Annotated[float, Field(description="another argument named b")]) -> float:
    return a * b


@mcp.tool(
    name="greet",
    description="Greet a user by name."
)
def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    # This runs the server, defaulting to STDIO transport
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        log_level="debug",
        path="/mcp",
    )
