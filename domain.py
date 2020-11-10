# -*- coding: utf-8 -*-
"""
A computer domain of LBM D2Q9
@author: Lin(Copy Shan)
"""

import numpy

class Domain:
    def __init__(self, shape, e, w, tau):

        self.dim = len(shape)
        self.d = len(e)
        self.e = e
        self.w = w
        self.omega = 1. / tau

        shape.append(self.d)
        self.f = numpy.empty(shape, dtype=numpy.float64) # dtype是对参数类型进行声明.

    def _state(self, f):
        density = sum(f)
        velocity = (f @ self.e) / density
        return density, velocity

    def _equilibrium(self, rho, u):
        ux = self.e @ u #点乘运算，但是单老师说e*u和是不一样的，不知道为什么不一样。
        return rho * self.w * (1 + 3. * ux + 4.5 * ux * ux - 1.5 * numpy.dot(u, u))

    def _stream(self):  
        for i, ei in enumerate(self.e):
            self.f[..., i] = numpy.roll(self.f[..., i], ei, range(self.dim))

    def _collide(self):
        fs = self.f.reshape(-1, self.d)
        for f in fs:
            rho, u = self._state(f)
            feq = self._equilibrium(rho, u)
            f -= self.omega * (f - feq)

    def initialize(self, initializer):
        shape = self.f.shape  #f.shape 是f的一个type.
        for i in range(shape[0]):
            x = float(i) / shape[0]
            for j in range(shape[1]):
                y = float(j) / shape[1]
                rho, u = initializer(x, y) #initializer函数：给x,y值，并返回此网格节点上的rho和u值
                self.f[i, j, :] = self._equilibrium(rho, u)

    def advance(self, steps):
        for i in range(steps):
            self._stream()
            self._collide()

    def state(self):
        shape = list(self.f.shape)
        shape[-1] = self.dim + 1
        state = numpy.empty(shape, dtype=numpy.float64)
        ss = state.reshape(-1, self.dim+1)
        fs = self.f.reshape(-1, self.d)
        for i, f in enumerate(fs):
            rho, u = self._state(f)
            ss[i, 0] = rho
            ss[i, 1:] = u

        return state