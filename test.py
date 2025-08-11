d = dict()
print(d.values())  # dict_values([]) 空的

d["hip"] = "a"
d["knee"] = "b"

print(list(d.values()))  # [actuator1, actuator2]

# for a in d.values():           # 遍历“值”
#     a.reset()
