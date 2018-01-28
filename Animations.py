from Simple import SimpleTuner
import matplotlib.pyplot as plotter
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy
import math
import time
import random
import heapq
import matplotlib.animation as animations
from matplotlib import cm
from matplotlib.patches import Polygon, Circle

from bayes_opt import BayesianOptimization
import cma


def objective(vector):#(x, y):
	x = vector[0]
	y = vector[1]
	value = 1 + 1/math.exp((x+3)**2 + (y-0)**2) + 0.5/math.exp((x+1)**2 + (y+2)**2)
	value += 0.1 * random.random()
	return value

def griewank(vector):
	part1 = 0
	for i in range(len(vector)):
		part1 += vector[i]**2
		part2 = 1
	for i in range(len(vector)):
		part2 *= math.cos(vector[i] / math.sqrt(i+1))
	return 5 - (1 + (part1/4000.0) - part2)

steepness = 2
shekelBeta = numpy.array([1 , 2 , 2, 4, 5]) / steepness
shekelC = numpy.array([[4, 1, 8, 6, 3], [4, 1, 8, 6, 7]]).T
def shekel(vector):
	vector = numpy.array(vector)
	difference = shekelC - vector
	difference *= difference
	difference = numpy.sum(difference, axis=1) + shekelBeta
	difference **= -1
	return numpy.sum(difference)


def michaelewicz(vector):
	steepness = 1
	functionSum = 0.4
	corrections = [0, 0]
	for i in range(len(corrections)):
		xi = vector[i] + corrections[i]
		functionSum += math.sin(xi) * (math.sin(i * xi * xi / math.pi) ** (2 * steepness))
	return functionSum


def getLoggingObjective(objectiveFunction, objectiveLog, timeLog, sampledCoords):
	startTime = time.time()
	def loggingObjective(vector):
		print(vector)
		objectiveValue = objectiveFunction(vector)
		objectiveLog.append(objectiveValue)
		timeLog.append(time.time() - startTime)
		sampledCoords.append(vector)
		return objectiveValue
	return loggingObjective

def renderAlgorithm(iterations=50):
	tuner = SimpleTuner([[-2.0, -1.0], [5.3, -1.0], [2.0, 7.3]], griewank, 0.6)
	tuner.optimize(2)

	figure = plotter.figure()
	axes1 = plotter.subplot2grid((5, 6), (0, 0), colspan=3, rowspan=3)
	axes2 = plotter.subplot2grid((5, 6), (0, 3), colspan=3, rowspan=3)
	axes3 = plotter.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=2)
	axes4 = plotter.subplot2grid((5, 6), (3, 2), colspan=3, rowspan=2)
	plotter.suptitle('Simple(x) Global Optimization', fontsize=18, fontweight='bold')

	figure.set_size_inches(9, 7)
	figure.tight_layout(rect=[0, 0.02, 1, 0.93])

	xDomain = numpy.arange(-4, 8, 0.02)
	yDomain = numpy.arange(-3, 9, 0.02)
	dim = len(xDomain)
	trueObjective = numpy.zeros((dim, dim))
	for xIndex in range(dim):
		for yIndex in range(dim):
			x = xDomain[xIndex]
			y = yDomain[yIndex]
			trueObjective[yIndex, xIndex] = griewank((x, y))

	maxValues = [tuner.maxValue]
	needsColorbars = [True]
	domainMin = trueObjective.min()
	domainMax = trueObjective.max()

	minAcquisitions = []
	maxAcquisitions = []

	def animate(iteration):
		heapq.heapify(tuner.queue)
		tuner.optimize(1)
		maxValues.append(tuner.maxValue)
		matrix = tuner.testPoints[0:tuner.iterations, :]
		x = matrix[:,0].flat
		y = matrix[:,1].flat
		z = matrix[:,2].flat
		coords = []
	
				

		acquisitions = []
		pendingX = []
		pendingY = []
		interpolatedValues = []
		currentDifference = tuner.maxValue - tuner.minValue
		for interpolation in tuner.queue:
			interpolation.update(currentDifference)
			acquisitions.append(-interpolation.acquisitionValue)
			pendingX.append(interpolation.testCoords[0])
			pendingY.append(interpolation.testCoords[1])
			coords.append(interpolation.pointIndices)

			vertexMatrix = tuner.testPoints[interpolation.pointIndices]
			coordMatrix = vertexMatrix[:, 0:-1]
			vertexValues = vertexMatrix[:,-1]
			for b1 in numpy.arange(0.02, .98, 0.02):
				for b2 in numpy.arange(0.02, 1 - b1, 0.02):
					b3 = 1 - (b1 + b2)
					barycentricCoords = numpy.array([b1, b2, b3])
					euclideanCoords = vertexMatrix.T.dot(barycentricCoords)
					differences = coordMatrix - euclideanCoords[0:-1]
					distances = numpy.sqrt(numpy.sum(differences * differences, axis=1))
					inverseDistances = 1 / distances
					inverseSum = numpy.sum(inverseDistances)
					interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum
					euclideanCoords[2] = interpolatedValue
					interpolatedValues.append(euclideanCoords)
		interpolatedMatrix = numpy.append(matrix, interpolatedValues, axis=0)
		interpolatedX = interpolatedMatrix[:,0]
		interpolatedY = interpolatedMatrix[:,1]
		interpolatedZ = interpolatedMatrix[:,2]

		minAcquisitions.append(min(acquisitions))
		maxAcquisitions.append(max(acquisitions))
	

		axes1.clear()
		axes1.set_title('Estimated Objective Value')
		axes1.tricontourf(interpolatedX, interpolatedY, interpolatedZ, 100, cmap=cm.viridis, vmin=domainMin, vmax=domainMax)
		axes1.triplot(x, y, coords, color='white', lw=0.5)
		axes1.scatter(x, y, color='black', zorder=10, s=5)

		axes2.clear()
		axes2.set_title('Simplex Acquisition Values')
		trianglePlot = axes2.tripcolor(x, y, coords, acquisitions, cmap=cm.Blues_r, vmin=1.7, vmax=4.1)
		axes2.triplot(x, y, coords, color='white', lw=0.5)
		samples = axes2.scatter(x, y, color='black', zorder=10, s=5)
		candidates = axes2.scatter(pendingX, pendingY, color='black', facecolors='none', s=10, zorder=9)
		

		axes3.clear()
		axes3.set_title('Objective Domain')
		contour = axes3.contourf(xDomain, yDomain, trueObjective, 100, cmap=cm.viridis)
		axes3.triplot(x[0:3], y[0:3], color='white', lw=1)
		axes3.scatter(x, y, color='black', zorder=10, s=5)

		axes4.clear()
		axes4.set_title('Optimization Progress')
		axes4.set_xlabel('Iteration Number')
		axes4.set_ylabel('Objective Value')
		length = len(z)
		peakObjective, = axes4.plot(range(1, length), maxValues)
		objectiveScatter = axes4.scatter(range(length), z, s=8)
		axes4.text(15, 5.08, 'global optimum', color='gray')
		axes4.plot((-5, 55), (5, 5), color='gray', linestyle='dashed', linewidth=1)
		axes4.set_xlim(-5, 55)
		axes4.set_ylim(2.5, 5.5)
		axes4.legend((samples, candidates, peakObjective, objectiveScatter), ('Sample\nLocation', 'Candidate\nSample \nLocation', 'Best\nSample', 'Sample\nValues'), loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

		if needsColorbars[0]:
			colorbarRange = numpy.linspace(domainMin, domainMax, 100)
			objectiveColorBar = figure.colorbar(contour, ax=axes1)
			figure.colorbar(trianglePlot, ax=axes2)
			needsColorbars[0] = False

		allAxes = (axes1, axes2, axes3, axes4)
		for axis in allAxes:
			axis.tick_params('both', direction='in')

		return allAxes


	animation = animations.FuncAnimation(figure, animate, iterations, interval=400, repeat_delay=5000)
	animation.save('animation.gif', writer='imagemagick')








def renderComparison(iterations=50):
	boValues = []
	boTimes = []
	boCoords = []
	logger = getLoggingObjective(shekel, boValues, boTimes, boCoords)
	optimizer = BayesianOptimization(lambda x,y: logger([x, y]), {'x': (0, 10), 'y': (0,10)})
	optimizer.maximize(n_iter=(iterations - 5))
	boCoords = numpy.array(boCoords)
	boX = boCoords[:, 0]
	boY = boCoords[:, 1]


	simpleValues = []
	simpleTimes = []
	simpleCoords = []
	logger = getLoggingObjective(shekel, simpleValues, simpleTimes, simpleCoords)
	simpleDomain = [[-0.2, -3.9], [-.2, 10.2], [13.9, 10.2]]
	tuner = SimpleTuner(simpleDomain, logger)
	tuner.optimize(iterations)
	simpleCoords = numpy.array(simpleCoords)
	simpleX = simpleCoords[:, 0]
	simpleY = simpleCoords[:, 1]

	figure = plotter.figure()
	axes1 = plotter.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
	axes2 = plotter.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=1)
	axes3 = plotter.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1)

	colors = plotter.rcParams['axes.prop_cycle'].by_key()['color']
	blue = colors[0]
	yellow = colors[1]

	figure.set_size_inches(9, 4.5)
	figure.tight_layout(rect=[0, 0.02, 1, 0.91], w_pad=2.5, h_pad = 2)
	plotter.suptitle('Simple vs. Bayesian Optimization*', fontsize=18, fontweight='bold')


	xDomain = numpy.arange(-1, 14.5, 0.1)
	yDomain = numpy.arange(-4.5, 11, 0.1)
	dim = len(xDomain)
	trueObjective = numpy.zeros((dim, dim))
	for xIndex in range(dim):
		for yIndex in range(dim):
			x = xDomain[xIndex]
			y = yDomain[yIndex]
			trueObjective[yIndex, xIndex] = shekel((x, y))

	globalMaximum = shekel([4, 4])

	boMaxValues = []
	simpleMaxValues = []

	needsColorbar = [True]

	def animate(iteration):
		bound = len(boMaxValues) + 1
		iterationsRange = range(bound)

		simpleValuesSection = simpleValues[0:bound]
		boValuesSection = boValues[0:bound]
		simpleMaxValues.append(max(simpleValuesSection))
		boMaxValues.append(max(boValuesSection))
	
		simpleTimesSection = simpleTimes[0:bound]
		boTimesSection = boTimes[0:bound]

		simpleXSection = simpleX[0:bound]
		simpleYSection = simpleY[0:bound]
		boXSection = boX[0:bound]
		boYSection = boY[0:bound]

		boDomain = numpy.array([[0, 0], [0, 10], [10, 10], [10, 0]])
		simpleDomainValues = numpy.array(simpleDomain)[:, 0:-1]
		axes1.clear()
		contour = axes1.contourf(xDomain, yDomain, trueObjective, 100, cmap=cm.viridis)
		axes1.add_patch(Polygon(simpleDomainValues, fill=False, color=blue, linestyle='dashed'))
		axes1.add_patch(Polygon(boDomain, fill=False, color=yellow, linestyle='dashed'))
		simpleScatter = axes1.scatter(simpleXSection, simpleYSection, s=5)
		boScatter = axes1.scatter(boXSection, boYSection, s=5)
		axes1.legend((simpleScatter, boScatter), ('Simple', 'Bayesian Optimization*'))
		#boSampleLocations = axes1.scatter([], [])
		axes1.set_title('Sample Selection Comparison')
		axes1.text(-3, -6.5, '*https://github.com/fmfn/BayesianOptimization (default params)')
	

		axes2.clear()
		axes2.plot(iterationsRange, simpleTimesSection, iterationsRange, boTimesSection)
		axes2.set_yscale('log')
		axes2.set_xlim(-5, 55)
		axes2.set_ylim(5E-5, 2E3)
		#axes2.legend(['Simple', 'Bayesian Optimization'])
		axes2.set_title('Runtime Performance Comparison')
		axes2.set_xlabel('Iteration Number', labelpad=-1)
		axes2.set_ylabel('Elapsed Time\n(seconds)', labelpad=-1)


		axes3.clear()
		axes3.plot(iterationsRange, simpleMaxValues, iterationsRange, boMaxValues)
		axes3.scatter(iterationsRange, simpleValuesSection, s=5)
		axes3.scatter(iterationsRange, boValuesSection, s=5)
		#boSamples = axes3.scatter([], [])
		#axes3.legend(['Simple', 'Bayesian Optimization'])
		axes3.set_title('Sample Efficiency Comparison')
		axes3.set_xlabel('Iteration Number', labelpad=-1)
		axes3.set_ylabel('Objective Value\n(arbitrary units)', labelpad=-1)
		axes3.text(15, globalMaximum + .08, 'global optimum', color='gray')
		axes3.plot((-5, 55), (globalMaximum, globalMaximum), color='gray', linestyle='dashed', linewidth=1)
		axes3.set_xlim(-5, 55)
		axes3.set_ylim(top=globalMaximum + 0.5)
	
		if needsColorbar[0]:
			figure.colorbar(contour, ax=axes1)
			needsColorbar[0] = False

		allAxes = (axes1, axes2, axes3)
		for axis in allAxes:
			axis.tick_params('both', direction='in')

		return allAxes

		return allAxes



	animation = animations.FuncAnimation(figure, animate, frames=(iterations-1), interval=400, repeat_delay=5000)
	animation.save('comparison.gif', writer='imagemagick')










def renderCmaesComparison(iterations=50):
	cmaValues = []
	cmaTimes = []
	cmaCoords = []
	logger = getLoggingObjective(shekel, cmaValues, cmaTimes, cmaCoords)
	#optimizer = BayesianOptimization(lambda x,y: logger([x, y]), {'x': (0, 10), 'y': (0,10)})
	#optimizer.maximize(n_iter=(iterations - 5))
	cmaOrigin = (7, 7)
	cmaRadius = 5.6 # 3 sigma
	sigma = cmaRadius / 3
	cma.fmin(lambda vector: -1 * logger(vector), cmaOrigin, sigma)
	cmaCoords = numpy.array(cmaCoords)
	cmaX = cmaCoords[:, 0]
	cmaY = cmaCoords[:, 1]

	adjustedCmaTimes = numpy.array(cmaTimes) - cmaTimes[0]

	simpleValues = []
	simpleTimes = []
	simpleCoords = []
	logger = getLoggingObjective(shekel, simpleValues, simpleTimes, simpleCoords)
	simpleDomain = [[-0.2, -3.9], [-.2, 10.2], [13.9, 10.2]]
	tuner = SimpleTuner(simpleDomain, logger)
	tuner.optimize(iterations)
	simpleCoords = numpy.array(simpleCoords)
	simpleX = simpleCoords[:, 0]
	simpleY = simpleCoords[:, 1]

	figure = plotter.figure()
	axes1 = plotter.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
	axes2 = plotter.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=1)
	axes3 = plotter.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1)

	colors = plotter.rcParams['axes.prop_cycle'].by_key()['color']
	blue = colors[0]
	yellow = colors[1]

	figure.set_size_inches(9, 4.5)
	figure.tight_layout(rect=[0, 0.02, 1, 0.91], w_pad=2.5, h_pad = 2)
	plotter.suptitle('Simple vs. CMA-ES*', fontsize=18, fontweight='bold')


	xDomain = numpy.arange(-1, 14.5, 0.1)
	yDomain = numpy.arange(-4.5, 11, 0.1)
	dim = len(xDomain)
	trueObjective = numpy.zeros((dim, dim))
	for xIndex in range(dim):
		for yIndex in range(dim):
			x = xDomain[xIndex]
			y = yDomain[yIndex]
			trueObjective[yIndex, xIndex] = shekel((x, y))

	globalMaximum = shekel([4, 4])

	cmaMaxValues = []
	simpleMaxValues = []

	needsColorbar = [True]

	def animate(iteration):
		bound = len(cmaMaxValues) + 1
		iterationsRange = range(bound)

		simpleValuesSection = simpleValues[0:bound]
		cmaValuesSection = cmaValues[0:bound]
		simpleMaxValues.append(max(simpleValuesSection))
		cmaMaxValues.append(max(cmaValuesSection))
	
		simpleTimesSection = simpleTimes[0:bound]
		cmaTimesSection = cmaTimes[0:bound]
		adjustedCmaTimesSection = adjustedCmaTimes[0:bound]

		simpleXSection = simpleX[0:bound]
		simpleYSection = simpleY[0:bound]
		cmaXSection = cmaX[0:bound]
		cmaYSection = cmaY[0:bound]

		simpleDomainValues = numpy.array(simpleDomain)[:, 0:-1]
		axes1.clear()
		axes1.set_xlim(-1, 14.5 - 0.1)
		axes1.set_ylim(-4.5, 11 - 0.1)
		contour = axes1.contourf(xDomain, yDomain, trueObjective, 100, cmap=cm.viridis)
		axes1.add_patch(Polygon(simpleDomainValues, fill=False, color=blue, linestyle='dashed'))
		axes1.add_patch(Circle(cmaOrigin, cmaRadius, fill=False, color=yellow, linestyle='dashed'))
		simpleScatter = axes1.scatter(simpleXSection, simpleYSection, s=5)
		cmaScatter = axes1.scatter(cmaXSection, cmaYSection, s=5)
		axes1.legend((simpleScatter, cmaScatter), ('Simple', 'CMA-ES*'), loc='lower right')
		axes1.set_title('Sample Selection Comparison')
		axes1.text(-3, -6.5, '*https://github.com/CMA-ES/pycma (domain=3 sigma)')
	

		axes2.clear()
		axes2.plot(iterationsRange, simpleTimesSection, iterationsRange, cmaTimesSection)
		axes2.plot(iterationsRange, adjustedCmaTimesSection, linestyle='dashed', color=yellow)
		axes2.set_yscale('log')
		axes2.set_xlim(-5, 55)
		axes2.set_ylim(5E-5, 2E3)
		#axes2.legend(['Simple', 'Bayesian Optimization'])
		axes2.set_title('Runtime Performance Comparison')
		axes2.set_xlabel('Iteration Number', labelpad=-1)
		axes2.set_ylabel('Elapsed Time\n(seconds)', labelpad=-1)


		axes3.clear()
		axes3.plot(iterationsRange, simpleMaxValues, iterationsRange, cmaMaxValues)
		axes3.scatter(iterationsRange, simpleValuesSection, s=5)
		axes3.scatter(iterationsRange, cmaValuesSection, s=5)
		#boSamples = axes3.scatter([], [])
		#axes3.legend(['Simple', 'Bayesian Optimization'])
		axes3.set_title('Sample Efficiency Comparison')
		axes3.set_xlabel('Iteration Number', labelpad=-1)
		axes3.set_ylabel('Objective Value\n(arbitrary units)', labelpad=-1)
		axes3.text(15, globalMaximum + .08, 'global optimum', color='gray')
		axes3.plot((-5, 55), (globalMaximum, globalMaximum), color='gray', linestyle='dashed', linewidth=1)
		axes3.set_xlim(-5, 55)
		axes3.set_ylim(top=globalMaximum + 0.5)
	
		if needsColorbar[0]:
			figure.colorbar(contour, ax=axes1)
			needsColorbar[0] = False

		allAxes = (axes1, axes2, axes3)
		for axis in allAxes:
			axis.tick_params('both', direction='in')

		return allAxes

		return allAxes



	animation = animations.FuncAnimation(figure, animate, frames=(iterations-1), interval=400, repeat_delay=5000)
	animation.save('cma-comparison.gif', writer='imagemagick')
























