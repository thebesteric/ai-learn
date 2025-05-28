from typing import List, Callable
import inspect

def auto_functions(func_list: List[Callable]) -> List[dict]:
    tools_description = []

    for func in func_list:
        # 获取函数的签名信息
        sig = inspect.signature(func)
        func_params = sig.parameters

        # 函数的参数描述
        parameters = {
            'type': 'object',
            'properties': {},
            'required': []
        }

        for param_name, param in func_params.items():
            # 添加参数描述和类型
            parameters['properties'][param_name] = {
                'description': param.annotation.__doc__ if param.annotation is not inspect._empty else "",
                'type': str(param.annotation) if param.annotation != param.empty else 'Any'
            }
            # 如果参数有默认值，那么它不是必须的
            if param.default != param.empty:
                parameters['required'].append(param_name)

        # 函数描述字典
        func_dict = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__.strip(),
                "parameters": parameters
            }
        }

        tools_description.append(func_dict)

    return tools_description

def machine_learning_1():
    """
    解释机器学习是什么
    """

    answer = """机器学习是人工智能的一个分支，研究计算机如何自动从数据中学习，提升性能并做出预测。\
    它通过算法让计算机提炼知识，优化任务执行，而无需明确编程。"""

    return answer


def check_tick(date, start, end):
    """
    查询是否可以订票
    :param date: 日期
    :param start: 出发站
    :param end: 终点站
    :return: 可以预定的车次信息
    """
    content = "xxxx"
    return content

functions_list = [machine_learning_1, check_tick]
tools = auto_functions(functions_list)
print(tools)