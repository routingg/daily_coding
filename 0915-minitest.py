# 1. 점수 리스트에서 짝수만 제곱해서 리스트 만들기

data = [0, 1, 2, 3, 4, 5]

res = []

for x in data:
    if x % 2 == 0:
        res.append(x*x)
print(res)

print("-------------------------------")

# pythonic


list = [x**2 for x in data if x%2 == 0]
print(list)

print("==============================")


# 2. 문자열 리스트에서 길이가 3 이상인 단어만 필터링

data = ["hello", "my", "name", "is", "yujeong"]

res = []

for x in data:
    if len(x) >= 3:
        res.append(x)

print(res)

print("-------------------------------")

# pythonic

list = [x for x in data if len(x) >=3 ]
print(list)

#변수 이름을 list로 쓰지 않는 게 좋아요.
# list는 파이썬 내장 타입 이름이라 덮어쓰면 혼란이 생깁니다.
# 대신 res, words, filtered 같은 이름이 더 좋아요.

print("==============================")

# 3. 숫자 리스트에서 평균보다 큰 값만 추리기
data = [1, 2, 3, 4, 5]

avg = sum(data) / len(data)
res = []
for x in data:
    if x > avg:
        res.append(x)
print(res)

print("-------------------------------")

# pythonic

res = [x for x in data if x>avg]
print(res)

