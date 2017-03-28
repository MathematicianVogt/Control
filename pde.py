import matplotlib.pyplot as plt
import time
import matplotlib.animation as manimation
import numpy as np
import matplotlib
import math
'''


['Spectral', 'summer', 'RdBu', 'Set1', 'Set2', 'Set3', 'brg_r', 'Dark2', 'hot', 'PuOr_r', 'afmhot_r', 'terrain_r', 'PuBuGn_r', 'RdPu', 'gist_ncar_r', 'gist_yarg_r', 'Dark2_r', 'YlGnBu', 'RdYlBu', 'hot_r', 'gist_rainbow_r', 'gist_stern', 'gnuplot_r', 'cool_r', 'cool', 'gray', 'copper_r', 'Greens_r', 'GnBu', 'gist_ncar', 'spring_r', 'gist_rainbow', 'RdYlBu_r', 'gist_heat_r', 'OrRd_r', 'bone', 'gist_stern_r', 'RdYlGn', 'Pastel2_r', 'spring', 'terrain', 'YlOrRd_r', 'Set2_r', 'winter_r', 'PuBu', 'RdGy_r', 'spectral', 'flag_r', 'jet_r', 'RdPu_r', 'Purples_r', 'gist_yarg', 'BuGn', 'Paired_r', 'hsv_r', 'bwr', 'YlOrRd', 'Greens', 'PRGn', 'gist_heat', 'spectral_r', 'Paired', 'hsv', 'Oranges_r', 'prism_r', 'Pastel2', 'Pastel1_r', 'Pastel1', 'gray_r', 'PuRd_r', 'Spectral_r', 'gnuplot2_r', 'BuPu', 'YlGnBu_r', 'copper', 'gist_earth_r', 'Set3_r', 'OrRd', 'PuBu_r', 'ocean_r', 'brg', 'gnuplot2', 'jet', 'bone_r', 'gist_earth', 'Oranges', 'RdYlGn_r', 'PiYG', 'YlGn', 'binary_r', 'gist_gray_r', 'Accent', 'BuPu_r', 'gist_gray', 'flag', 'seismic_r', 'RdBu_r', 'BrBG', 'Reds', 'BuGn_r', 'summer_r', 'GnBu_r', 'BrBG_r', 'Reds_r', 'RdGy', 'PuRd', 'Accent_r', 'Blues', 'Greys', 'autumn', 'PRGn_r', 'Greys_r', 'pink', 'binary', 'winter', 'gnuplot', 'pink_r', 'prism', 'YlOrBr', 'rainbow_r', 'rainbow', 'PiYG_r', 'YlGn_r', 'Blues_r', 'YlOrBr_r', 'seismic', 'Purples', 'bwr_r', 'autumn_r', 'ocean', 'Set1_r', 'PuOr', 'PuBuGn', 'afmhot']
'''
def onedfunc(f,interval):
	output=[]
	for i in interval:
		output.append(f(i))
	return output

def onedinterval(a,b,n):
	interval = []
	h=(b-a)/float(n)
	for i in range(0,n+1):
		interval.append(a+i*h)
	return interval
def derv(f,x,h):
	return (f(x+h)-f(x-h))/(2*h)



def makeMovie(xList, solList,tList,title,xlab,ylab):
		FFMpegWriter = manimation.writers['ffmpeg']
		metadata = dict(title=title, artist='Matplotlib',
		                comment='Movie support!')
		writer = FFMpegWriter(fps=1, metadata=metadata)

		fig = plt.figure()
		l, = plt.plot([], [], 'k-o')

		plt.ylim(0, 2)
		plt.xlim(xList[0]-1,xList[len(xList)-1]+1)

		with writer.saving(fig, title+ ".mp4", 100):
			for i in range(0,len(tList)):

				x0 = xList
				y0 =solList[i]

				#plt.xlim([-15,15])
				#plt.ylim([-2,2])
				plt.plot(x0,y0)
				plt.title(title + " Time = " + str(tList[i]))
				plt.xlabel(xlab)
				plt.ylabel(ylab)
				writer.grab_frame()
				plt.clf()
def makeMovieMax(xList,yList,tList, solList,title,xlab,ylab):
		FFMpegWriter = manimation.writers['ffmpeg']
		metadata = dict(title=title, artist='Matplotlib',
		                comment='Movie support!')
		writer = FFMpegWriter(fps=1, metadata=metadata)

		fig = plt.figure()
		l, = plt.plot([], [], 'k-o')

		plt.ylim(0, 1)
		plt.xlim(0,1)
		xx,yy=np.meshgrid(xList,yList)
		with writer.saving(fig, title+ ".mp4", 100):
			for i in range(0,len(tList)):

				im = plt.imshow(np.flipud(solList[i]),extent=[0,1,0,1], cmap='gist_rainbow')
				plt.colorbar(im, orientation='horizontal')
				plt.title(title + " Time = " + str(tList[i]))
				plt.xlabel(xlab)
				plt.ylabel(ylab)
				writer.grab_frame()
				plt.clf()

def generate_spline_2d(xval,yval):
	tck, u = inter.splprep([xval, yval], s=0)
	out = inter.splev(u, tck)
	return (out[0],out[1])

def generate_spline_2d_tangent(xval,yval):
	tck, u = inter.splprep([xval, yval], s=0)
	derv = inter.splev(u, tck,der=1)
	return (derv[0],derv[1])

def generate_spline_2d_normal(xval,yval):
	tck, u = inter.splprep([xval, yval], s=0)
	derv = inter.splev(u, tck,der=1)
	return (-derv[1],derv[0])
