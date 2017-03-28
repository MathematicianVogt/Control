import numpy as np 
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pylab as plt
import scipy.optimize as op
import math
import random
from pde import *
import time


def flux(u):
	#print (u**2)/2.0
	return (u**2)/2.0

def control_cost(parameter_guess,a,b,x,dx,t,dt,esp):
	#print parameter_guess
	parameter_guess=parameter_guess.tolist()
	print parameter_guess
	#time.sleep(5)
	sol = gudunov(parameter_guess,dx,dt,x,t,flux,esp)
	final_sol=sol[-1]
	integrand = np.multiply(np.subtract(np.array(final_sol) - tracking_sol_spactial(x)),np.subtract(np.array(final_sol) - tracking_sol_spactial(x)))
	cost = integrate.trapz(integrand,x)
	#print cost
	return cost


def cost_min(u):
	return flux(u)

def cost_max(u):
	return -flux(u)

def gudunov_flux(ul,ur):
	if(ul<=ur):
		res=op.minimize(cost_min, (ul+ur)/2.0, method='TNC',bounds=((ul,ur),))
		return res.fun
	elif(ul>ur):
		res=op.minimize(cost_max, (ul+ur)/2.0, method='TNC',bounds=((ur,ul),))
		return res.fun


def g(t):
	return math.sin(2*math.pi*t)

def tracking_sol_spactial(x):
	return np.exp(-x**2)

def gudunov(initial_cond,dx,dt,x,t,flux,esp):
	burger_solution=[]
	#print initial_cond
	burger_solution.append(initial_cond)
	# print initial_cond
	# print type(initial_cond)
	# print type(burger_solution)
	# print burger_solution[-1]
	x=x.tolist()
	t=t.tolist()
	
	for n in range(1,len(t)):
		last_sol=burger_solution[-1]
		current_time_step_sol=[]
		#print last_sol
		for i in range(0,len(x)):
			if(i==0):
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i+1]) - gudunov_flux(2*dx*g(t[n]) + last_sol[i+1],last_sol[i])) + (esp/dx**2)*(2*dx*g(t[n]) + last_sol[i+1] -2*last_sol[i] +last_sol[i+1])
				unext=np.asscalar(unext)
				print type(unext)
				current_time_step_sol.append(unext)

			elif(i==len(x)-1):
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i-1]) - gudunov_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( 2*last_sol[i-1] -2*last_sol[i])
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

			else:
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i+1]) - gudunov_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( last_sol[i-1] -2*last_sol[i] + last_sol[i+1])
				print unext
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

		print current_time_step_sol
		burger_solution.append(current_time_step_sol)

	return burger_solution


def initial_guess(a,b,spatial_number):
	(x,dx)=np.linspace(a,b,spatial_number,retstep=True)
	return (list(np.exp(-x**2)),dx)



def burger_problem(a,b,tmax,spatial_number,time_number,esp):
	(x,dx) = np.linspace(a,b,spatial_number,retstep=True)
	(t,dt) = np.linspace(0,tmax,time_number,retstep=True)
	print dt/dx
	(parameter_guess,dx) = initial_guess(a,b,spatial_number)
	#print parameter_guess
	res=op.minimize(control_cost, parameter_guess, args=(a,b,x,dx,t,dt,esp), method='SLSQP')
	plt.plot(t,res.x)
	makeMovie(x, burger_solution,t,"Burgesp=" +str(esp),"space(x)","State")



burger_problem(-10,10,5,100,1000,1)