def python_inter(py_code, g='globals()'):
    """
    专门用于执行用户提供的 Python 代码，并返回执行结果
    :param py_code: 字符串形式的 Python 代码
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: 代码运行的最终结果
    """

    print("正在调用 python_inter 函数")
    try:
        # 若是表达式，直接运行并返回结果，如"1+1"，"max([1,2,3])"
        result = eval(py_code, g)
        return str(result)
    except Exception as e:
        # 记录执行前的全局变量
        global_vars_before = set(g.keys())
        try:
            # 处理完整的 Python 代码，如"a=1;b=2;print(a+b)"
            exec(py_code, g)
        except Exception as e:
            return f"代码执行出错：{str(e)}"

        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before

        print("正在进行最后的整理...")

        # 如果有新变量，则返回这些变量的值
        if new_vars:
            result = {var: g[var] for var in new_vars}
            return str(result)
        else:
            return "代码执行成功"


example_python_inter_args = '{"py_code": "import numpy as np\\narr = np.array([1, 2, 3])\\nsum_arr = np.sum(arr)"}'
python_inter_tool = {
    "type": "function",
    "function": {
        "name": "python_inter",
        "description": f"当用户需要编写 python 程序并执行时，请调用此函数。该函数可以执行一段 Python 代码，并返回执行结果。\n"
                       f"需要注意，本函数只能执行非绘图类的代码，若是绘图类的代码，则需要调用 fig_inter 函数运行。\n"
                       f"同时需要注意，编写外部函数的参数消息时，必须是满足 json 格式的字符串。\n\n"
                       f"例如：如下形式的字符串就是合规的字符串：{example_python_inter_args}",
        "parameters": {
            "type": "object",
            "properties": {
                "py_code": {
                    "type": "string",
                    "description": "The Python code to execute. It can be an expression or a complete Python code."
                },
                "g": {
                    "type": "string",
                    "description": "Global environment variables, default to globals().",
                    "default": "globals()"
                }
            },
            "required": ["py_code"]
        }
    }
}


def fig_inter(py_code, fname, g='globals()'):
    """
    专门用于执行用户提供的绘图代码，并返回执行结果
    :param py_code: 字符串形式的 Python 代码
    :param fname: 图片的名称
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: 代码运行的最终结果
    """
    print("正在调用 fig_inter 函数")
    import matplotlib
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from IPython.display import display, Image

    # 切换位无交互式后端
    current_backend = matplotlib.get_backend()
    matplotlib.use('Agg')

    # 用于执行代码的本地变量
    local_vars = {"plt": plt, "pd": pd, "sns": sns}

    # 相对路径保存目录
    pics_dir = "./assets/pics"
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)

    try:
        # 执行用户代码
        exec(py_code, g, local_vars)
        g.update(local_vars)

        # 获取图像对象
        fig = local_vars.get(fname, None)
        if fig:
            rel_path = os.path.join(pics_dir, f"{fname}.png")
            fig.savefig(rel_path, bbox_inches='tight')
            display(Image(filename=rel_path))
            print(f"代码已顺利执行，正在进行结果梳理...")
            return f"✅ 图像已保存到 {rel_path}"
        else:
            print(f"代码执行失败，尝试修复中...")
            return "⚠️ 代码执行成功，但未找到图像对象，请确保有 'fig = ...'"
    except Exception as e:
        # 如果执行代码时出错，则返回错误信息
        return f"❌ 代码执行出错：{str(e)}"
    finally:
        # 恢复原始交互式后端
        matplotlib.use(current_backend)


fig_inter_tool = {
    "type": "function",
    "function": {
        "name": "fig_inter",
        "description": "当用户需要使用 Python 进行可视化绘图任务时，请调用此函数。\n"
                       "该函数会执行用户提供的 Python 绘图代码，并自动将生成的图像对象保存在图片文件并进行展示。\n\n"
                       "调用该函数时，请传入以下参数：\n"
                       "1. `py_code`: 字符串形式的 Python 代码，**必须是完整、可独立运行的脚本**，代码必须创建并返回一个命名为 `fname` 的 matplotlib 图像对象；\n"
                       "2. `fname`: 图像对象的变量名，字符串形式，默认为 `fig`；\n"
                       "3. `g`: 全局变量环境，默认保持为 'globals()' 即可。\n\n"
                       "📌 请确保绘图代码满足以下要求：\n"
                       "- 包含所有必要的 import 语句，例如 `import matplotlib.pyplot as plt`，`import seaborn as sns` 等；\n"
                       "- 必须包含数据定义，如：`df = pd.DataFrame(...)`，不要依赖外部变量；\n"
                       "- 推荐使用 `ax` 对象进行绘图操作，例如 `sns.lineplot(..., ax=ax)`；\n"
                       "- 最后明确图像对象保存为 `fname` 变量，例如 `fig = plt.gcf()`。\n\n"
                       "📌 不需要自己保存图像，函数会自动保存并展示。\n\n"
                       "✅ 合规的示例代码：\n"
                       "```python\n"
                       "import matplotlib.pyplot as plt\n"
                       "import seaborn as sns\n"
                       "import pandas as pd\n" 
                       "df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})\n"
                       "fig, ax = plt.subplots()\n"
                       "sns.lineplot(x='x', y='y', data=df, ax=ax)\n"
                       "plt.title('Line Plot')\n"
                       "plt.xlabel('X Axis')\n"
                       "plt.ylabel('Y Axis')\n"
                       "fig = plt.gcf() # 一定要赋值给 fname 指定的变量名\n"
                       "```\n\n"
                       "⚠️ 注意事项：\n"
                       "需要注意，本函数只能执行绘图类的代码，若是非绘图类的代码，则需要调用 python_inter 函数运行。",
        "parameters": {
            "type": "object",
            "properties": {
                "py_code": {
                    "type": "string",
                    "description": "需要执行的 Python 绘图代码（字符串形式），代码必须创建一个 matplotlib 图像对象并赋值给 `fname` 所指定的变量名。"
                },
                "fname": {
                    "type": "string",
                    "description": "图像对象的变量名，例如：'fig'，代码中必须使用这个变量名来保存图像对象。",
                },
                "g": {
                    "type": "string",
                    "description": "运行环境变量，默认保持为 'globals()' 即可。",
                    "default": "globals()"
                }
            },
            "required": ["py_code", "fname"]
        }
    }
}


if __name__ == "__main__":
    print(example_python_inter_args)
    result = python_inter("import numpy as np\narr = np.array([1, 2, 3])\nsum_arr = np.sum(arr)", g=globals())
    print(result)

    fig_inter(py_code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 25, 30, 40]})

# 创建一个散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=df, s=100, color='blue')
plt.title('Scatter Plot Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 保存图像到变量 fig
fig = plt.gcf()
""", fname="fig", g=globals())
