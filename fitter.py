import math
import random

unary_list = ['sin','cos','tan']
binary_list = ['+','*']
def evalstr(fnstring, constant_list, x):
    fnstring = fnstring.replace(" ","")
    tmp = fnstring
    out = ''
    cnt = 0
    c = 0
    while cnt < len(fnstring):
        prog = 0
        for i in range(len(unary_list)):
            if tmp.startswith(unary_list[i]+'('):
                cnt += len(unary_list[i])
                out = out + 'math.' + unary_list[i]
                tmp = tmp[len(unary_list[i]):]
                prog = 1
                break
        if prog == 1 :
            continue
        if tmp.startswith('x'):
            out = out + str(x)
            tmp = tmp[1:]
            cnt += 1
            continue
        elif tmp.startswith('c'):
            out = out + str(constant_list[c])
            tmp = tmp[1:]
            cnt += 1
            c += 1
            continue
        else :
            out = out + fnstring[cnt]
            tmp = tmp[1:]
            cnt += 1
    #print eval(out)
    return eval(out)

def numc(fnstring):
    tmp = fnstring
    cnt = 0
    c = 0
    while cnt < len(fnstring):
        prog = 0
        for i in range(len(unary_list)):
            if tmp.startswith(unary_list[i]+'('):
                cnt += len(unary_list[i])
                tmp = tmp[len(unary_list[i]):]
                prog = 1
                break
        if prog == 1 :
            continue
        if tmp.startswith('x'):
            tmp = tmp[1:]
            cnt += 1
            continue
        elif tmp.startswith('c'):
            tmp = tmp[1:]
            cnt += 1
            c += 1
            continue
        else :
            tmp = tmp[1:]
            cnt += 1
    return c

def errorsum(fnstring,constant_list,x,y):
    err = 0.0
    for i in range(len(x)):
        err += (evalstr(fnstring,constant_list,x[i])-y[i])**2
    return err

def regression(fnstring, x, y, tol=0.01, eta = 0.001, dx = 0.001):
    constant_list = []
    for i in range(numc(fnstring)):
        constant_list.append(0)
    err = errorsum(fnstring, constant_list,x,y)
    c = numc(fnstring)
    print c
    while err > tol :
        for i in range(c):
            tmp_list = list(constant_list)
            tmp_list[i] += dx
            grad = (errorsum(fnstring, tmp_list,x,y) - err)/dx
            constant_list[i] -= eta*grad
            err = errorsum(fnstring, constant_list,x,y)
        print err
    return err, constant_list

x = []
y = []
for i in range(100):
    x.append(i*math.pi/100)
    y.append(math.sin(x[i]*0.45)+math.cos(x[i]))
err, constant_list = regression('sin(x*c)+cos(x*c)',x,y)
print err, constant_list
#print evalstr('sin(x+1)+tan(c)',[1],1)
#print numc('sin(c)+tan(c)+cos(c)')
