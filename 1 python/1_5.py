def q1(x):
    # 1.找到 2000~3200之间 7 的除数，不能除尽5 的数
    def helper(n):
        if n % 7==0 and n%5 != 0:
            return True
        else:
            return False
    ans = []
    for n in x:
        if helper(n):
            ans.append(n)
    return ans


# ans = q1(range(2000,3200))
# print(ans)


def q2(n):
    # 2. 输入一个数n，输出n 的阶乘
    if n==0:
        return 1
    return n*q2(n-1)

# x=int(input("Enter number: "))
# print("%d!=%d"%(x,q2(x)))


def q3(n):
    # 3.输入一个整数n，输出字典{n:n^2}
    if n<1:
        return "请输入大于等于1的整数"
    ans = {}
    for i in range(1,n+1):
        ans[i] = i**2
    return ans

x=int(input("Q3: "))
print("ans: ",q3(x))
