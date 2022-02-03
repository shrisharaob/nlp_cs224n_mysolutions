def climbStairs(n: int) -> int:
    l, r = 1, 1
    tmp = 0
    i = n - 1
    while i > 0:
        tmp = l
        l = l + r
        r = tmp
        i -= 1
    return l

out = climbStairs(2)

print(out)