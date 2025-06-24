import asyncio
import os

import browser_cookie3
from browser_use.agent.service import Agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
llm = ChatDeepSeek(base_url=os.getenv("DEEPSEEK_BASE_URL"), model='deepseek-chat', api_key=os.getenv("DEEPSEEK_API_KEY"))

task = """
1. 访问：https://passport.ctrip.com/user/login?BackUrl=https%3A%2F%2Fwww.ctrip.com%2F#ctm_ref=c_ph_login_buttom
2. 填写用户名：13966660426
3. 填写密码：P@ssw0rd
4. 勾选：阅读并同意携程的服务协议和个人信息保护政策
5. 点击：登录按钮
6. 如果出现验证码弹窗，则点击：获取验证码，并等待用户输入验证码后继续
7. 等待登录成功后，点击主页的搜索按钮
"""


async def main():
    agent = Agent(
        task=task,
        llm=llm,
    )
    result = await agent.run()
    print('result:', result)


def get_cookies(url: str, cookie_name: str = None):
    """
    获取制定域名下的制定 cookie 的值
    :param url: url
    :param cookie_name: cookie 名称
    :return:
    """
    # 创建 Chrome cookie 加载器
    cj = browser_cookie3.chrome()
    # 获取指定域名的 cookies
    cookies = {cookie.name: cookie.value for cookie in cj if cookie.domain in url}
    if cookie_name:
        return cookies.get(cookie_name)
    return cookies


if __name__ == "__main__":
    asyncio.run(main())
    cticket = get_cookies("https://hotels.ctrip.com/hotels/list", "cticket")
    print(cticket)
