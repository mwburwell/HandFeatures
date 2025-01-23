import numpy as np


import math

class Point():
    def __init__(self, point: tuple) -> None:
        self.x = point[0]
        self.y = point[1]
    
    def toCartesian(self):
        return (self.x, self.y)

class Feature():
    def __init__(self, point1, point2, color = 'r', isFinger= True) -> None:
        
        normalVector = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        
        magnitude = math.sqrt(normalVector[0] ** 2 + normalVector[1] ** 2)
        
        self.p1 = Point(point1)
        self.p2 = Point(point2)
        
        self.unitNVector = normalVector / magnitude
        self.unitDVector = np.array([self.unitNVector[1] * (-1), self.unitNVector[0]])
        self.isFinger = isFinger
        self.color = color
    
    def slope(self):
        return (self.p1.y - self.p2.y) / (self.p1.x - self.p2.x)
    
    def orthoganalSlope(self):
        return (-1 / self.slope())
    
class HandFeatures():
    def __init__(self, pDict: dict) -> None:
        self.features = {}
        self.features['F1'] = Feature(pDict['I'], pDict['J'], 'y', False)
        self.features['F2'] = Feature(pDict['K'], pDict['L'], 'r')
        self.features['F3'] = Feature(pDict['C'], pDict['F'], 'b')
        self.features['F4'] = Feature(pDict['A'], pDict['D'], 'g')
        self.features['F5'] = Feature(pDict['B'], pDict['E'], 'black')
        self.features['F6'] = Feature(pDict['G'], pDict['H'], 'r')
    
    def getFeatures(self) -> dict:
        return self.features
    
    def getFeature(self, idx) -> Feature:
        return self.features[idx]