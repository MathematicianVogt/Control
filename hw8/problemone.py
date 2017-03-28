import numpy as np 
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pylab as plt
import scipy.optimize as op
import math

def make_cons(parameter_guess):
	cons=()
	for i in range(0,len(parameter_guess)):
		constraint = {'type': 'ineq', 'fun': lambda x:  -math.fabs(x[i]) + 1 }
		cons +=(constraint,)
	# print cons
	#cons=({'type': 'ineq', 'fun': lambda parameter_guess:  -parameter_guess+ 1 })
	return cons


def bnds(parameter_guess):
	bnds=()
	for i in range(0,len(parameter_guess)):
		bnds +=((-1.0,1.0),)
	print bnds
	return bnds
def problem(N,IC):
	t=np.linspace(0,5,1000)
	tt=np.linspace(0,5,N+1)
	parameter_guess = .5*np.ones(len(tt))
	res=op.minimize(cost_function, parameter_guess, args=(t,tt,IC), method='SLSQP',bounds=bnds(parameter_guess))
	true_param= res.x
	print res.message
	print true_param
	generate_state_and_control(true_param,t,tt,IC)


def cost_function(parameter_guess,t,tt,IC):
	#print parameter_guess
	f_p = interpolate.interp1d(tt, parameter_guess)
	sol = integrate.odeint(f, [IC[0],IC[1],0], t, args=(f_p,))
	cost_sol = sol[:,2]
	cost=cost_sol[-1]
	print 'cost ' + str(cost) 
	return cost


def f(y,t,f_p):
	if t<5.0:
		dydt=[-y[0] +2*y[1] , y[0] -.2*y[1] + f_p(t), .5*(y[0]**2 + 2*y[1]**2 + 3*f_p(t)**2)]
	else: 
		dydt=[-y[0] +2*y[1] , y[0] -.2*y[1] + f_p(5), .5*(y[0]**2 + 2*y[1]**2 + 3*f_p(5)**2)]
	return dydt


def generate_state_and_control(parameters,t,tt,IC):
	f_p = interpolate.interp1d(tt, parameters)
	sol = integrate.odeint(f, [IC[0],IC[1],0], t, args=(f_p,))
	control=f_p(t)
	position=sol[:,0]
	velocity=sol[:,1]
	cost_sol = sol[:,2]
	cost=cost_sol[-1]
	print 'cost ' + str(cost) 
	print parameters
	plt.plot(tt,parameters,label='Control')
	plt.xlabel('time')
	plt.ylabel('u')
	plt.title('Control')
	plt.show()
	plt.clf()
	plt.plot(position,velocity,label='Velocity vs Position')
	plt.xlabel('Position')
	plt.ylabel('Velocity')
	plt.title('Velocity vs Position')
	plt.show()
	plt.clf()


problem(5,[.1,.1])
problem(5,[3,6])
problem(15,[.1,.1])
problem(15,[3,6])