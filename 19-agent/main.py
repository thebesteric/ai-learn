from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from .agent.re_act import ReActAgent
from .models.factory import ChatModelFactory
from .tools.python_tools import ExcelAnalyser
from .tools.tools import document_qa_tool, document_generation_tool, email_tool, excel_inspection_tool, directory_inspection_tool, finish_placeholder

_ = load_dotenv(find_dotenv())


def launch_agent(agent: ReActAgent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    chat_history = ChatMessageHistory()
    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}: {reply}\n")


def main():
    llm = ChatModelFactory.get_model("deepseek")
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(llm=llm, prompt_file="./prompts/tools/excel_analyser.txt", verbose=True).as_tool()
    ]
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main.txt",
        max_thought_steps=20
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
