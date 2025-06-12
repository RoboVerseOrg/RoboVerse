def record_number_positions(lst):
    reverse_idex = []
    for i in range(len(lst)):
        j = 0
        while j < len(lst):
            if lst[j] == i:
                reverse_idex.append(j)
                break
            j += 1
    return reverse_idex


def reverse_joint_index_2(lst: list[int]) -> list[int]:
    reverse_index = [0] * len(lst)
    for i, val in enumerate(lst):
        reverse_index[val] = i
    return reverse_index


# 示例用法
numbers = [4, 14, 2, 1, 3, 0, 11, 12, 13, 9, 18, 7, 6, 5, 8, 15, 16, 17, 10]
output = record_number_positions(numbers)
output2 = reverse_joint_index_2(numbers)
print(output)
print(output2)
