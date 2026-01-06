import json
from typing import Any, List, Sequence

from termcolor import cprint


def reorder_init_states(init_states: Sequence[Any], mapping_json_path: str) -> List[Any]:
    """
    根据 mapping.json 中的映射，把 init_states 中的位置重排：
      new_states[i] = init_states[mapping[i]]
    其余位置不在映射中时，则保持原始顺序。

    Args:
        init_states: 原始的状态列表（如 list 或其他支持索引的序列）。
        mapping_json_path: JSON 文件路径，内容格式为 {"i": "j", ...}，表示第 i 个 demo 对应原始第 j 个 demo。

    Returns:
        new_states: 重排后的 init_states 列表。
    """
    # 载入并解析映射
    try:
        with open(mapping_json_path, "r") as f:
            raw_map = json.load(f)
    except FileNotFoundError:
        cprint(
            f"[Eval]: Error: {mapping_json_path} doesn't exist, \n\tassuming the head of the dataset is training data",
            "red",
        )
        return init_states
    mapping = {int(i): int(j) for i, j in raw_map.items()}

    N = len(init_states)
    # 2. 检查映射值是否合法
    for tgt, src in mapping.items():
        if src < 0 or src >= N:
            raise IndexError(f"mapping[{tgt}] = {src} 越界 (len={N})")

    # 3. 前半部分：按 target 索引升序填入映射项
    new_states: List[Any] = []
    for tgt in sorted(mapping.keys()):
        print(f"Adding Init State[{mapping[tgt]}] to new_states[{tgt}]")
        new_states.append(init_states[mapping[tgt]])

    # 4. 后半部分：追加所有未映射的原列表条目
    mapped_srcs = set(mapping.values())
    for idx in range(N):
        if idx not in mapped_srcs:
            new_states.append(init_states[idx])
    return new_states
