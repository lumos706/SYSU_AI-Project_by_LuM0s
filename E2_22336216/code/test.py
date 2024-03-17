input_str = "KB = {(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}"

import re
# 去除外部大括号
input_str = input_str[5:].strip("{}")
# 按逗号分割每个元组，并处理每个元组
result = []
matches = input_str.split('),(')
for match in matches:
    match = match.strip('(').strip(',')
    result.append(match)

KB = []
for element in result:
    matches = (re.findall(r'~?\w+\(\w+,*\w*\)', element))
    KB.append(matches)

for i in range(len(KB)):
    for j in range(len(KB[i])):
        KB[i][j] = KB[i][j].replace('(', ",").replace(')', '').split(',')

print()
for item in KB:
    print(item)
print()
n = len(KB)
print(n)