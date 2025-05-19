from typing import TypedDict, List, Dict, Any

"""
### 规约器：归约器是理解如何将节点的更新应用到 State 的关键
如果没有明确指定归约器函数，则假定对该键的所有更新都应覆盖它

在这个示例中，没有为任何键指定归约器函数。假设图的输入是 {"foo": 1, "bar": ["hi"]}。然后假设第一个 Node 返回 {"foo": 2}。这被视为对状态的更新。
请注意，Node 不需要返回整个 State 模式，只需要返回一个更新即可。
应用此更新后，State 将变为 {"foo": 2, "bar": ["hi"]}。
如果第二个节点返回 {"bar": ["bye"]}，则 State 将变为 {"foo": 2, "bar": ["bye"]}
"""


class State(TypedDict):
    foo: int
    bar: List[str]


# 模拟默认规约器的实现
def update_state(current_state: State, updates: Dict[str, Any]) -> State:
    # 创建一个新的状态字典
    new_state = current_state.copy()
    # 更新状态字典中的值
    new_state.update(updates)
    return new_state


# 初始状态
state: State = {"foo": 1, "bar": ["hi"]}

# 第一个节点返回的更新
node1_update = {"foo": 2}
state = update_state(state, node1_update)
print(state)  # 输出: {'foo': 2, 'bar': ['hi']}

# 第二个节点返回的更新
node2_update = {"bar": ["bye"]}
state = update_state(state, node2_update)
print(state)  # 输出: {'foo': 2, 'bar': ['bye']}
