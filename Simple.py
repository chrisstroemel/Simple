from heapq import heappush, heappop, heappushpop
import numpy
import math
import time
import matplotlib.pyplot as plotter

CAPACITY_INCREMENT = 1000


class _Simplex:
	def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction, difference):
		self.pointIndices = pointIndices
		self.testCoords = testCoords
		self.contentFractions = contentFractions
		self.contentFraction = contentFraction
		self.__objectiveScore = objectiveScore
		self.__opportunityCost = opportunityCost
		self.update(difference)

	def update(self, difference):
		self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
		self.difference = difference

	def __eq__(self, other):
		return self.acquisitionValue == other.acquisitionValue

	def __lt__(self, other):
		return self.acquisitionValue < other.acquisitionValue

class SimpleTuner:
	def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
		self.__cornerPoints = cornerPoints
		self.__numberOfVertices = len(cornerPoints)
		self.queue = []
		self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
		self.testPoints = numpy.empty((self.capacity, self.__numberOfVertices))
		self.objective = objectiveFunction
		self.iterations = 0
		self.maxValue = None
		self.minValue = None
		self.bestCoords = []
		self.opportunityCostFactor = exploration_preference #/ self.__numberOfVertices
			

	def optimize(self, maxSteps=10):
		for step in range(maxSteps):
			#print(self.maxValue, self.iterations, self.bestCoords)
			if len(self.queue) > 0:
				targetSimplex = self.__getNextSimplex()
				newPointIndex = self.__testCoords(targetSimplex.testCoords)
				for i in range(0, self.__numberOfVertices):
					tempIndex = targetSimplex.pointIndices[i]
					targetSimplex.pointIndices[i] = newPointIndex
					newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
					newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
					heappush(self.queue, newSimplex)
					targetSimplex.pointIndices[i] = tempIndex
			else:
				testPoint = self.__cornerPoints[self.iterations]
				testPoint.append(0)
				testPoint = numpy.array(testPoint, dtype=numpy.float64)
				self.__testCoords(testPoint)
				if self.iterations == (self.__numberOfVertices - 1):
					initialSimplex = self.__makeSimplex(numpy.arange(self.__numberOfVertices, dtype=numpy.intp), 1)
					heappush(self.queue, initialSimplex)
			self.iterations += 1

	def get_best(self):
		return (self.maxValue, self.bestCoords[0:-1])

	def __getNextSimplex(self):
		targetSimplex = heappop(self.queue)
		currentDifference = self.maxValue - self.minValue
		while currentDifference > targetSimplex.difference:
			targetSimplex.update(currentDifference)
			# if greater than because heapq is in ascending order
			if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
				targetSimplex = heappushpop(self.queue, targetSimplex)
		return targetSimplex
		
	def __testCoords(self, testCoords):
		objectiveValue = self.objective(testCoords[0:-1])
		if self.maxValue == None or objectiveValue > self.maxValue: 
			self.maxValue = objectiveValue
			self.bestCoords = testCoords
			if self.minValue == None: self.minValue = objectiveValue
		elif objectiveValue < self.minValue:
			self.minValue = objectiveValue
		testCoords[-1] = objectiveValue
		if self.capacity == self.iterations:
			self.capacity += CAPACITY_INCREMENT
			self.testPoints.resize((self.capacity, self.__numberOfVertices))
		newPointIndex = self.iterations
		self.testPoints[newPointIndex] = testCoords
		return newPointIndex


	def __makeSimplex(self, pointIndices, contentFraction):
		vertexMatrix = self.testPoints[pointIndices]
		coordMatrix = vertexMatrix[:, 0:-1]
		barycenterLocation = numpy.sum(vertexMatrix, axis=0) / self.__numberOfVertices

		differences = coordMatrix - barycenterLocation[0:-1]
		distances = numpy.sqrt(numpy.sum(differences * differences, axis=1))
		totalDistance = numpy.sum(distances)
		barycentricTestCoords = distances / totalDistance

		euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)
		
		vertexValues = vertexMatrix[:,-1]

		testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
		testPointDistances = numpy.sqrt(numpy.sum(testpointDifferences * testpointDifferences, axis=1))



		inverseDistances = 1 / testPointDistances
		inverseSum = numpy.sum(inverseDistances)
		interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum


		currentDifference = self.maxValue - self.minValue
		opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

		return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue, opportunityCost, contentFraction, currentDifference)

	def plot(self):
		if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
		matrix = self.testPoints[0:self.iterations, :]

		x = matrix[:,0].flat
		y = matrix[:,1].flat
		z = matrix[:,2].flat

		coords = []
		acquisitions = []

		for triangle in self.queue:
			coords.append(triangle.pointIndices)
			acquisitions.append(-1 * triangle.acquisitionValue)


		plotter.figure()
		plotter.tricontourf(x, y, coords, z)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()


		plotter.figure()
		plotter.tripcolor(x, y, coords, acquisitions)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()

		plotter.show()





