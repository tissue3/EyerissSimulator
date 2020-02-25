def findAllFactors(num):
    lt = []
    for i in range(1, num+1):
        if num % i == 0:
            lt.append(i)
    print(lt)
