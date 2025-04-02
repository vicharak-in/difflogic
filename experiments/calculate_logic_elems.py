def gates(depth):
    return 2 ** depth - 1

def conv_gates(dims, depth):
    g = gates(depth)
    print(g)
    return dims[0] * dims[1] * dims[2]*g

def or_gates(dims):
    return dims[0] * dims[1] * dims[2]

def cal_gates(net, depth):
    g = 0
    for i in net:
        if i[0] == 'conv':
            g += conv_gates(i[1], depth)
        elif i[0] == 'or':
            g += or_gates(i[1])
        elif i[0] == 'fc':
            g += i[1][0]
    return g

k =32
net = [
       ("conv", [k/8, 32, 32]),
       ("or", [k/8, 16, 16]),
       ("conv", [k/2, 16, 16]),
       ("or", [k/2, 8, 8]),
       ("conv", [k*2, 8, 8]),
       ("or", [k*2, 4, 4]),
       ("conv", [k*4, 4, 4]),
       ("or", [k*4, 2, 2]),
       ("fc", [k*128]),
       ("fc", [k*1280]),
       ("fc", [k*640])
       ]

print(cal_gates(net, 3))
