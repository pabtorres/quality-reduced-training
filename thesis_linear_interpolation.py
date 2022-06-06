def interpolacion(a_1, a_0, CICLOS):
    k = a_0 + (a_1-a_0)/(CICLOS-1)
    l = [a_0]
    for i in range(1, CICLOS):
        l.append(a_0 + (a_1-a_0)/(CICLOS-1)*i)
    l.reverse()
    return l