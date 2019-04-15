lam = 20
l = [x+1 for x in range(lam)]
f = lambda i: np.max([0, np.log(lam/2+1)-np.log(i)])
t = np.array([f(x) for x in l])
divisor = np.sum(t)
final = t/divisor - 1/lam