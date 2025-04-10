import asyncio
import json

from mcp.server import FastMCP

mcp = FastMCP("PythonServer")
USER_AGENT = "PythonServer-app/1.0"

@mcp.tool()
async def python_inter(python_code):
    """
    运行用户提供的 Python 代码，并返回执行结果
    :param python_code: 字符串形式的 Python 代码
    :return: 代码运行的最终结果
    """
    g = globals()

    try:
        # 若是表达式，直接运行并返回结果，如"1+1"，"max([1,2,3])"
        result = eval(python_code, g)
        return json.dumps(str(result), ensure_ascii=False)
    except Exception:
        # 记录执行前的全局变量
        global_vars_before = set(g.keys())
        try:
            # 处理完整的 Python 代码，如"a=1;b=2;print(a+b)"
            exec(python_code, g)
        except Exception as e:
            return json.dumps(f"代码执行出错：{str(e)}", ensure_ascii=False)

        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before

        if new_vars:
            safe_result = {}
            for var in new_vars:
                try:
                    # 尝试序列化，确保可以转换为
                    json.dumps(g[var]  )
                    safe_result[var] = g[var]
                except (TypeError, OverflowError):
                    # 如果无法序列化，则转换为字符串
                    safe_result[var] = str(g[var])
            return json.dumps(safe_result, ensure_ascii=False)
        else:
            return json.dumps("代码执行成功", ensure_ascii=False)

async def test():
    print(await python_inter("a=10\nb=20\nc=a+b"))

if __name__ == "__main__":
    # asyncio.run(test())
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")

