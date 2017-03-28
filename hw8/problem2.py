import numpy as np 
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pylab as plt
import scipy.optimize as op
import math
import random

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
		if (i==len(parameter_guess)-1):
			bnds +=((0.0,None),)
		else:
			bnds +=((-1.0,1.0),)
	print bnds
	return bnds
def problem(N,IC,FC):
	#part a
	# tt=np.linspace(0,1,N+1)
	# param_guess=[]
	# for i in range(0,N+1):
	# 	param_guess.append(random.uniform(-1.0,1.0))

	# print param_guess
	# parameter_guess=np.array(param_guess)
	# #parameter_guess = .5*np.ones(len(tt))
	# parameter_guess= np.append(parameter_guess, [5.1])
	# print parameter_guess
	# Tguess= parameter_guess[-1]
	# t=np.linspace(0,1,1000)
	
	# res=op.minimize(cost_function, parameter_guess, args=(t,tt,IC,FC), method='SLSQP',bounds=bnds(parameter_guess))
	# true_param= res.x
	# print res.message
	# print true_param
	# generate_state_and_control(true_param,t,tt,IC)
	#part b
	tt=np.linspace(0,1,N+1)
	a=double_generate(tt)
	parameter_guess=generate_parameters(a)
	Tguess= parameter_guess[-1]
	t=np.linspace(0,1,1000)
	
	res=op.minimize(cost_function, parameter_guess, args=(t,tt,IC,FC), method='L-BFGS-B',bounds=bnds(parameter_guess))
	true_param= res.x
	print res.message
	print true_param
	generate_state_and_control(true_param,t,tt,IC)


def double_generate(array):
	newarray=[]
	array=list(array)
	for i in range(0,len(array)):
		if i!=0 and i!=len(array)-1:
			newarray.append(array[i])
			newarray.append(array[i])
		else:
			newarray.append(array[i])
	return np.array(newarray)
def generate_parameters(array):
	newarray=[]
	array=list(array)
	for i in range(0,len(array)):
		newarray.append(random.uniform(-1,1))
	newarray.append(5.1)
	return np.array(newarray)
def cost_function(parameter_guess,t,tt,IC,FC):
	#print parameter_guess
	#part a
	# n=11
	# Tmax=parameter_guess[-1]
	# tauTmax=np.linspace(0,Tmax,1000)
	# tau=np.linspace(0,1,1000)
	# taup=np.linspace(0,1,n)
	# parameter_guess=parameter_guess[0:len(parameter_guess)-1]
	# f_p = interpolate.interp1d(taup, parameter_guess)
	# sol = integrate.odeint(f1, [IC[0],IC[1],0], tau, args=(f_p,Tmax))
	# v=sol[:,1]
	# w=sol[:,2]
	
	# taup= np.linspace(0,Tmax,n)
	# f_v = interpolate.interp1d(tauTmax,v)
	# f_w = interpolate.interp1d(tauTmax,w)
	# f_p=interpolate.interp1d(taup, parameter_guess)
	# sol = integrate.odeint(f2, [IC[0],IC[1],0], tau, args=(f_p,f_w,f_v,Tmax))
	# v=sol[:,1]
	# w=sol[:,1]
	# cost=Tmax
	# print 'cost ' + str(cost) 

	#part b
	n=6
	Tmax=parameter_guess[-1]
	tauTmax=np.linspace(0,Tmax,1000)
	tau=np.linspace(0,1,1000)
	taup=np.linspace(0,1,n)
	taup=double_generate(taup)
	parameter_guess=generate_parameters(taup)
	print parameter_guess
	parameter_guess=parameter_guess[0:len(parameter_guess)-1]
	f_p = interpolate.interp1d(taup, parameter_guess)
	sol = integrate.odeint(f1, [IC[0],IC[1],0], tau, args=(f_p,Tmax))
	v=sol[:,0]
	w=sol[:,1]
	
	taup= np.linspace(0,Tmax,n)
	taup=double_generate(taup)
	f_v = interpolate.interp1d(tauTmax,v)
	f_w = interpolate.interp1d(tauTmax,w)
	f_p=interpolate.interp1d(taup, parameter_guess)
	sol = integrate.odeint(f2, [IC[0],IC[1],0], tau, args=(f_p,f_w,f_v,Tmax))
	v=sol[:,0]
	w=sol[:,1]
	cost=Tmax
	print 'cost ' + str(cost) 

	
	return (Tmax+ 5*v[-1]**2 + 5*w[-1]**2)/10000 


def f2(y,t,f_p,f_w,f_v,Tmax):
	if t<Tmax:
		dydt=[ Tmax*f_w(t), Tmax*(-4*f_v(t) +2*f_p(t)), Tmax]
	else: 
		dydt=[Tmax*f_w(Tmax), Tmax*(-4*f_v(Tmax) +2*f_p(Tmax)), Tmax]
	return dydt

def f1(y,t,f_p,Tmax):
	if t<1.0:
		dydt=[ Tmax*y[1], Tmax*(-4*y[0] +2*f_p(t)), Tmax]
	else: 
		dydt=[Tmax*y[1], Tmax*(-4*y[0] +2*f_p(1)), Tmax]
	return dydt


def generate_state_and_control(parameters,t,tt,IC):
	#part a
	# n=11
	# parameter_guess=parameters
	# Tmax=parameter_guess[-1]
	# tauTmax=np.linspace(0,Tmax,1000)
	# tau=np.linspace(0,1,1000)
	# taup=np.linspace(0,1,n)
	# parameter_guess=parameter_guess[0:len(parameter_guess)-1]
	# f_p = interpolate.interp1d(taup, parameter_guess)
	# sol = integrate.odeint(f1, [IC[0],IC[1],0], tau, args=(f_p,Tmax))
	# v=sol[:,1]
	# w=sol[:,2]
	
	# taup= np.linspace(0,Tmax,n)
	# f_v = interpolate.interp1d(tauTmax,v)
	# f_w = interpolate.interp1d(tauTmax,w)
	# f_p=interpolate.interp1d(taup, parameter_guess)
	# sol = integrate.odeint(f2, [IC[0],IC[1],0], tauTmax, args=(f_p,f_w,f_v,Tmax))
	# position=sol[:,1]
	# velocity=sol[:,2]
	# print parameters
	# parameters=parameters[0:len(parameters)-1]
	# plt.plot(taup,parameters,label='Control')
	# plt.xlabel('time')
	# plt.ylabel('u')
	# plt.title('Control')
	# plt.show()
	# plt.clf()
	# plt.plot(position,velocity,label='Velocity vs Position')
	# plt.xlabel('Position')
	# plt.ylabel('Velocity')
	# plt.title('Velocity vs Position')
	# plt.show()
	# plt.clf()
	#part b
	n=6
	parameter_guess=parameters
	Tmax=parameter_guess[-1]
	tauTmax=np.linspace(0,Tmax,1000)
	tau=np.linspace(0,1,1000)
	taup=np.linspace(0,1,n)
	parameter_guess=parameter_guess[0:len(parameter_guess)-1]
	taup=double_generate(taup)
	f_p = interpolate.interp1d(taup, parameter_guess)
	sol = integrate.odeint(f1, [IC[0],IC[1],0], tau, args=(f_p,Tmax))
	v=sol[:,0]
	w=sol[:,1]
	
	taup= np.linspace(0,Tmax,n)
	taup=double_generate(taup)
	f_v = interpolate.interp1d(tauTmax,v)
	f_w = interpolate.interp1d(tauTmax,w)
	f_p=interpolate.interp1d(taup, parameter_guess)
	sol = integrate.odeint(f2, [IC[0],IC[1],0], tauTmax, args=(f_p,f_w,f_v,Tmax))
	position=sol[:,0]
	velocity=sol[:,1]
	print parameters
	parameters=parameters[0:len(parameters)-1]
	plt.plot(taup,parameters,label='Control')
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
	print position[0]
	print velocity[0]


problem(5,[3.3,1.1],[0,0])