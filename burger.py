import numpy as np 
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pylab as plt
import scipy.optimize as op
import math
import random
from pde import *
import time


temp=None
def flux(u):
	#print (u**2)/2.0
	return (u**2)/2.0

def control_cost_god(parameter_guess,a,b,x,dx,t,dt,esp,tracking_sol):
	#print parameter_guess
	parameter_guess=parameter_guess.tolist()
	#plt.plot(x,parameter_guess)
	#plt.show()
	#print parameter_guess
	#time.sleep(5)
	sol = gudunov(parameter_guess,dx,dt,x,t,flux,esp)
	final_sol=sol[-1]
	a=np.subtract(np.array(final_sol),tracking_sol)
	integrand = np.multiply(a,a)
	cost = integrate.trapz(integrand,x)
	#print cost
	return cost

def control_cost_eo(parameter_guess,a,b,x,dx,t,dt,esp,tracking_sol):
	#print parameter_guess
	parameter_guess=parameter_guess.tolist()
	#plt.plot(x,parameter_guess)
	#plt.show()
	#print parameter_guess
	#time.sleep(5)
	sol = eo(parameter_guess,dx,dt,x,t,flux,esp)
	final_sol=sol[-1]
	#plt.plot(x,final_sol)
	#plt.show()

	a=np.subtract(np.array(final_sol),tracking_sol)
	integrand = np.multiply(a,a)
	cost = integrate.trapz(integrand,x)
	print cost
	return cost


def cost_min(u):
	return flux(u)

def cost_max(u):
	return -flux(u)

def eointegrand(u):
	h=.01
	#return math.fabs((flux(u+h) - flux(u))/h)
	return math.fabs(u)
def eo_flux(ul,ur):
	
	integral ,err = integrate.quad(eointegrand, ul, ur) 
	return .5*(flux(ul) + flux(ur)) - .5*integral


def gudunov_flux(ul,ur):
	print (ul,ur)
	if(ul<=ur):
		res=op.minimize(cost_min, (ul+ur)/2.0, method='CG',bounds=((ul,ur),))
		#print res.fun
		return res.fun
	elif(ur<=ul):
		res=op.minimize(cost_max, (ul+ur)/2.0, method='CG',bounds=((ur,ul),))
		#print res.fun
		return res.fun


def g(t):
	#return math.sin(2*math.pi*t)
	return 0
def tracking_sol_spactial(x):
	#return np.exp(-x**2)
	pass
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
	#print dt/dx
	for n in range(1,len(t)):
		last_sol=burger_solution[-1]
		current_time_step_sol=[]
		plt.plot(x,last_sol)
		plt.show()
		for i in range(0,len(x)):
			if(i==0):
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i+1]) - gudunov_flux(2*dx*g(t[n]) + last_sol[i+1],last_sol[i])) + (esp/dx**2)*(2*dx*g(t[n]) + last_sol[i+1] -2*last_sol[i] +last_sol[i+1])
				unext=np.asscalar(unext)
				#print type(unext)
				current_time_step_sol.append(unext)

			elif(i==len(x)-1):
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i-1]) - gudunov_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( 2*last_sol[i-1] -2*last_sol[i])
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

			else:
				unext = last_sol[i] -(dt/dx)*(gudunov_flux(last_sol[i],last_sol[i+1]) - gudunov_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( last_sol[i-1] -2*last_sol[i] + last_sol[i+1])
				#print unext
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

		#print current_time_step_sol
		burger_solution.append(current_time_step_sol)

	global temp
	temp=burger_solution
	return burger_solution




def eo(initial_cond,dx,dt,x,t,flux,esp):
	
	burger_solution=[]
	#print initial_cond
	burger_solution.append(initial_cond)
	#plt.plot(x,initial_cond)
	#plt.show()
	# print initial_cond
	# print type(initial_cond)
	# print type(burger_solution)
	# print burger_solution[-1]
	x=x.tolist()
	t=t.tolist()
	#print dt/(dx**2)
	#time.sleep(10)
	
	for n in range(1,len(t)):
		last_sol=burger_solution[-1]
		current_time_step_sol=[]
		#print last_sol
		#plt.plot(x,last_sol)
		#plt.show()
		#print t[n]
		for i in range(0,len(x)):
			if(i==0):
				unext = last_sol[i] -(dt/dx)*(eo_flux(last_sol[i],last_sol[i+1]) - eo_flux(2*dx*g(t[n]) + last_sol[i+1],last_sol[i])) + (esp/dx**2)*(2*dx*g(t[n]) + last_sol[i+1] -2*last_sol[i] +last_sol[i+1])
				unext=np.asscalar(unext)
				#print type(unext)
				current_time_step_sol.append(unext)

			elif(i==len(x)-1):
				#no net flux
				unext = last_sol[i] -(dt/dx)*(eo_flux(last_sol[i],last_sol[i-1]) - eo_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( 2*last_sol[i-1] -2*last_sol[i])
				
				#reflective
				#unext = last_sol[i] -(dt/dx)*(eo_flux(last_sol[i],-last_sol[i]*2*dx + last_sol[i-1]) - eo_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( last_sol[i-1] -2*last_sol[i] + -last_sol[i]*2*dx + last_sol[i-1])
				
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

			else:
				unext = last_sol[i] -(dt/dx)*(eo_flux(last_sol[i],last_sol[i+1]) - eo_flux(last_sol[i-1],last_sol[i])) + (esp/dx**2)*( last_sol[i-1] -2*last_sol[i] + last_sol[i+1])
				#print unext
				unext=np.asscalar(unext)
				current_time_step_sol.append(unext)

		#print current_time_step_sol
		burger_solution.append(current_time_step_sol)
	global temp
	temp=burger_solution	
	
	return burger_solution

def initial_guess(a,b,spatial_number):
	(x,dx)=np.linspace(a,b,spatial_number,retstep=True)
	#return (list(np.exp(-x**2)),dx)
	return (list(np.sin((2.0/10.0)*math.pi*x)),dx)


def burger_problem(a,b,tmax,spatial_number,time_number,esp):
	(x,dx) = np.linspace(a,b,spatial_number,retstep=True)
	(t,dt) = np.linspace(0,tmax,time_number,retstep=True)
	print dt/dx
	(parameter_guess,dx) = initial_guess(a,b,spatial_number)
	initial_cond=list(np.sin((2.0/10.0)*math.pi*x))

	sol=eo(initial_cond,dx,dt,x,t,flux,esp)
	print "done making tracking"
	tracking_sol=sol[-1]

	res=op.minimize(control_cost_god, parameter_guess, args=(a,b,x,dx,t,dt,esp,tracking_sol), method='SLSQP')
	plt.plot(x,res.x)
	plt.xlabel('Space(x)')
	plt.ylabel('Control(u)')
	plt.title('Initial Condition')
	burger_solution=temp
	plt.savefig('controlBurgers')
	makeMovie(x, burger_solution,t,"Burgesp=" +str(esp),"space(x)","State")



def eo_problem(a,b,tmax,spatial_number,time_number,esp):
	(x,dx) = np.linspace(a,b,spatial_number,retstep=True)
	(t,dt) = np.linspace(0,tmax,time_number,retstep=True)
	print dt/dx
	(parameter_guess,dx) = initial_guess(a,b,spatial_number)
	initial_cond=list(np.sin((2.0/10.0)*math.pi*x))

	sol=eo(initial_cond,dx,dt,x,t,flux,esp)
	print "done making tracking"
	tracking_sol=sol[-1]
	#print parameter_guess
	res=op.minimize(control_cost_eo, parameter_guess, args=(a,b,x,dx,t,dt,esp,tracking_sol), method='CG')
	plt.plot(x,res.x)
	plt.xlabel('Space(x)')
	plt.ylabel('Control(u)')
	plt.title('Initial Condition')
	burger_solution=temp
	plt.savefig('control')
	
	makeMovie(x, burger_solution,t,"eop=" +str(esp),"space(x)","State")



#print res1.x
#print res2.x


burger_problem(-10,10,5,100,1000,.00001)