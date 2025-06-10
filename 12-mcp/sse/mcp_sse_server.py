import argparse
import json
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from mcp.server import FastMCP, Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount

load_dotenv()

mcp = FastMCP("WeatherServer")

# Open Weather API 配置
OPEN_WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPEN_WEATHER_API_KEY = "57ae333b23774c9bb9b82273213d7d47"
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 Open Weather API 获取天气信息
    :param city: 城市名称（需要使用英文，如 "Shanghai"）
    :return: 天气数据字典，若出错返回包含 error 信息的字典
    """
    params = {
        "q": city,
        "appid": OPEN_WEATHER_API_KEY,
        "lang": "zh_cn",
        "units": "metric",
    }
    headers = {
        "User-Agent": USER_AGENT,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPEN_WEATHER_BASE_URL, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}


def format_weather(weather_data: dict[str, Any] | str) -> str:
    """
    格式化天气数据为易读文本
    :param weather_data: 天气数据（可以是字典或 JSON 格式字符串）
    :return: 格式化后的天气信息字符串
    """
    if isinstance(weather_data, str):
        try:
            weather_data = json.loads(weather_data)
        except Exception as e:
            return f"无法解析天气数据：{e}"

    if "error" in weather_data:
        return f"天气查询出错：{weather_data['error']}"

    # 提取数据
    city = weather_data.get("name", "未知")
    country = weather_data.get("sys", {}).get("country", "未知")
    temperature = weather_data.get("main", {}).get("temp", "N/A")
    humidity = weather_data.get("main", {}).get("humidity", "N/A")
    wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
    weather_list = weather_data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")

    return (
        f"城市：{city} ({country})\n"
        f"天气：{description}\n"
        f"温度：{temperature}°C\n"
        f"湿度：{humidity}%\n"
        f"风速：{wind_speed} m/s"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果
    :param city: 城市名称（需要使用英文，如 "Shanghai"）
    :return: 格式化后的天气信息
    """
    weather_data = await fetch_weather(city)
    return format_weather(weather_data)


@mcp.tool()
async def query_current_time() -> str:
    """
    返回当前时间
    :return: 当前时间字符串
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_starlette_app(server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            # 定义 /sse 路径，当客户端访问此路径时将触发 handle_sse 函数处理 SSE 连接
            Route("/sse", endpoint=handle_sse),
            # 将 /messages/ 路径挂载到 sse.handle_post_message 应用上，用于处理通过 POST 请求发送的消息，实现与 SSE 长连接的消息传递功能。
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=18080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
