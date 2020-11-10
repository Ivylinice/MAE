'''
The main code
@author: Lin(Copy Shan)
'''

import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from domain import Domain

d2q9_e = numpy.array([
    [ 0,  0],
    [ 1,  0],
    [ 0,  1],
    [-1,  0],
    [ 0, -1],
    [ 1,  1],
    [-1,  1],
    [-1, -1],
    [ 1, -1]
])

d2q9_w = numpy.concatenate(([4./9.], 4 * [1./9.], 4 * [1./36]))#方括号表示这是一个list. numpy.concatenate表示把后面的连成一个.

def shear_layer(x, y): #剪切流
    vx = u0 * math.tanh((abs(y - 0.5) - 0.25) / thickness) #thickness的物理意义不清楚.
    vy = delta * u0 * math.sin(2. * math.pi * (x - .25))
    return 1., numpy.array([vx, vy]) #返回密度和速度.定死了密度为1，而且在整个过程中的密度都是1.

Reynolds  = 10000.
Mach      = 0.1   #近不可压的运算.
L         = [100, 100]
thickness = 0.0125
delta     = 0.05 #初始扰动

u0 = Mach / math.sqrt(3.)
nu = u0 * L[1] / Reynolds   #这个特征尺度为什么要去L[1]?
tau = 3. * nu + .5
#
#setup wave numbers for Fourler transform
#
LH=int( L[0] / 2)
scale=2j * numpy.pi / L[0] 

kx = scale * (numpy.arange(0, L[0]) - LH + 1) #numpy.arange 创建等距间隔的数组，包含起始数不包括终止数。
ky = scale * numpy.arange(0, LH + 1) 

kx=numpy.roll(kx, LH + 1)
#
#setup computational domain
#
domain = Domain(L, d2q9_e, d2q9_w, tau)
domain.initialize(shear_layer)
domain.advance(1)
'''
for i in range(2):
    print(i)
    domain.advance(10)
    
    state = domain.state()
    
    ux=numpy.fft.rfft2(state[:,:,1])
    uy=numpy.fft.rfft2(state[:,:,2])
    
    vorticity=numpy.fft.irfft2(ux * ky - (uy.T * kx).T)
    
    plt.imshow(vorticity.T, cmap =cm.viridis,origin='lower',interpolation='bilinear')
    plt.savefig("w%d.png" % (i))
'''

state = domain.state()
'''
【总结】乘运算：
1.c=a@  (有顺序)
2.b=numpy.dot(a,a)
''' 