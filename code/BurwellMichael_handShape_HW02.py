import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import math

from skimage import io
from skimage.color import rgb2gray
import scipy.ndimage as ndimage

from HandFeatures import HandFeatures
from HandFeatures import Feature
from HandFeatures import Point

def main():
    plainHands, gtHands, grayHands = loadHandImages()
    
    allFeatures = []

    figure2 = plt.figure()
    fig, axes= plt.subplots(2, 3)

    x_axis = 0
    y_axis = 0
    flagPlotOneSetOfFeatures = True
    processingOrder = 1
    for img_index in range(len(grayHands)):
        print(f'Processing image: {processingOrder}')
        processingOrder += 1
        
        featurePoints = getFeaturePointsFromGT(gtHands[img_index])
        features = clusterFeatures(featurePoints)
        grayCopy = grayHands[img_index].copy()

        if flagPlotOneSetOfFeatures:
            figure2 = plotEachFeature(plainHands[0], grayCopy, features)
            flagPlotOneSetOfFeatures = False
            
        featureDistances, featureStartToEndPoints = scanAllFeatures(features, grayCopy)
        
        axes[x_axis][y_axis] = showFeatures(axes[x_axis][y_axis], grayCopy, featureStartToEndPoints)
        
        if y_axis > 1:
            x_axis += 1
            y_axis = 0
        else:
            y_axis += 1
        
        allFeatures.append(featureDistances)
        
    euc_distances = np.zeros([len(allFeatures), len(allFeatures)])
    for i in range(len(allFeatures)):
        for ii in range(len(allFeatures)):
            if ii <= i:
                continue
            
            euc_distances[i][ii] = eucDistance(allFeatures[i], allFeatures[ii])     
            
    print(euc_distances)
    fig.tight_layout(pad=0.1)
    plt.show()
    
def eucDistance(array1, array2):
    total = 0
    for i in range(len(array1)):
        total += (array1[i] - array2[i]) ** 2
        
    return math.sqrt(total)

def clusterFeatures(featurePoints):
    variable = 65
    pointDict = {}
    for point in featurePoints:
        pointDict[chr(variable)] = (point[1], point[0])
        variable += 1
    
    features = HandFeatures(pointDict)
    
    return features


def loadHandImages(limit = 5):
    print('loading images')
    plainHands = []
    gtHands = []
    grayHands = []
    for i in range(1, limit + 1):
        plainIMG =  io.imread(f'hand/handSecondSet{i}.jpg')
        plainHands.append(plainIMG)
        
        gray_img= rgb2gray(plainIMG.copy())
        grayHands.append(gray_img)
        gtHands.append(plt.imread(f'handGT/handSecondSetGT{i}.png'))
        
    return plainHands, gtHands, grayHands



def getFeaturePointsFromGT(gt_image):
    print('Getting Feature points from ground-truthed image')
    featurePoints = []
    width = gt_image.shape[0]
    height = gt_image.shape[1]
    
    for x in range(width):
        print(f'completed: {round((x/width) * 100)}%', end='\r')
        for y in range(height):
            # looks for the reddest pixel, ground truthing was done with a red pen
            if(gt_image[x][y][0] >= .8 and gt_image[x][y][1] < .4 and gt_image[x][y][2] < .4):
                
                addPoint = True
                for point in featurePoints:
                    pfx = point[0]
                    pfy = point[1]
                    
                    summation = ((x - pfx) ** 2) + ((y - pfy) ** 2)
                    distance = math.sqrt(summation)
                    
                    if distance < 100:
                        addPoint = False
                        break
                
                if addPoint:
                    featurePoints.append((x, y))
                    
    return featurePoints

def plotEachFeature(plain_image, gray_image, features):
    print('Plotting features for first image')
    fig, axis = plt.subplots(2, 4)
    
    axis[1][0].set_xticks([])
    axis[1][0].set_yticks([])
    axis[1][0].set_title('Original')
    axis[1][0].imshow(plain_image)
    drawFeatureLines(gray_image, features, 'F6', axis[0][0])
    drawFeatureLines(gray_image, features, 'F5', axis[0][1])
    drawFeatureLines(gray_image, features, 'F4', axis[0][2])
    drawFeatureLines(gray_image, features, 'F3', axis[0][3])
    drawFeatureLines(gray_image, features, 'F2', axis[1][3])
    drawFeatureLines(gray_image, features, 'F1', axis[1][2])
    
    return fig

def drawFeatureLines(gray_image, features, featureLabel, ax):
    feature = features.features[featureLabel]
    ax.set_title(f'{featureLabel}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(gray_image, cmap=plt.cm.gray)
    ax.axline(feature.p1.toCartesian(), feature.p2.toCartesian(), color=feature.color, linewidth=1)
    ax.axline(feature.p1.toCartesian(), slope= feature.orthoganalSlope(), ls= ':', color = feature.color, linewidth=1)
    ax.axline(feature.p2.toCartesian(), slope= feature.orthoganalSlope(), ls= ':', color = feature.color, linewidth=1)

def scanFeatureR2L(hand_gray_img, point: Point, unitDVector, start=-250, end=250, scale=30, smoothing= 11, leftThreshold= 1500, rightThreshold= 1000):
    currentPoint = [point.x, point.y] + np.multiply(unitDVector, start)
    
    maxValues = []
    minValues = []
    maxIDXs = []
    minIDXs = []
    difference = []
    
    step = end
    
    maxStartDif = 0.0
    maxEndDif = 0.0
    beginFinger: tuple = (0, 0)
    endFinger: tuple = (0, 0)
    
    fingerStart = 0
    fingerEnd = 0
    while(step > start):
        
        currentPoint = [point.x, point.y] + np.multiply(unitDVector, step)
        windowArray = np.array(
            hand_gray_img[
                round(currentPoint[1]) - scale : round(currentPoint[1]) + scale,
                round(currentPoint[0]) - scale : round(currentPoint[0]) + scale])
        
        smoothWindow = ndimage.gaussian_filter(windowArray, sigma=smoothing)
        
        maxVal = np.max(smoothWindow)
        minVal = np.min(smoothWindow)
        
        maxIDX = np.argmax(smoothWindow)
        minIDX = np.argmin(smoothWindow)
        
        maxValues.append(maxVal)
        minValues.append(minVal)
        
        maxIDXs.append(maxIDX)
        minIDXs.append(minIDX)

        difference.append(maxVal - minVal)
        
        currentDifference = maxVal - minVal
        
        if minIDX - maxIDX > leftThreshold:
            if currentDifference > maxStartDif:
                maxStartDif = currentDifference
                fingerStart = step
                beginFinger = (round(currentPoint[0]), round(currentPoint[1]))
            
            
                
        if maxIDX - minIDX > rightThreshold:
            if currentDifference > maxEndDif:
                maxEndDif = currentDifference
                fingerEnd = step
                endFinger = (round(currentPoint[0]), round(currentPoint[1]))

    
        step -= 1
        
    
    # p1 = abs(fingerStart - end)
    # p2 = abs(fingerEnd - end)
    # plotValueGraph(maxValues, minValues, maxIDXs, minIDXs, difference, p1, p2)


        
    return beginFinger, endFinger

def scanFeatureL2R(hand_gray_img, point: Point, unitDVector, start=-250, end=250, scale=30, smoothing= 11, leftThreshold= 1500, rightThreshold= 1000):
    currentPoint = [point.x, point.y] + np.multiply(unitDVector, start)
    
    maxValues = []
    minValues = []
    maxIDXs = []
    minIDXs = []
    difference = []
    
    step = start
    
    maxStartDif = 0.0
    maxEndDif = 0.0
    beginFinger: tuple = (0, 0)
    endFinger: tuple = (0, 0)
    
    fingerStart = 0
    fingerEnd = 0
    while(step < end):
        currentPoint = [point.x, point.y] + np.multiply(unitDVector, step)
        windowArray = np.array(
            hand_gray_img[
                round(currentPoint[1]) - scale : round(currentPoint[1]) + scale,
                round(currentPoint[0]) - scale : round(currentPoint[0]) + scale])
        
        smoothWindow = ndimage.gaussian_filter(windowArray, sigma=smoothing)
        
        maxVal = np.max(smoothWindow)
        minVal = np.min(smoothWindow)
        
        maxIDX = np.argmax(smoothWindow)
        minIDX = np.argmin(smoothWindow)
        
        maxValues.append(maxVal)
        minValues.append(minVal)
        
        maxIDXs.append(maxIDX)
        minIDXs.append(minIDX)

        difference.append(maxVal - minVal)
        
        currentDifference = maxVal - minVal
        
        if minIDX - maxIDX > leftThreshold:
            if currentDifference > maxStartDif:
                maxStartDif = currentDifference
                fingerStart = step
                beginFinger = (round(currentPoint[0]), round(currentPoint[1]))
            
            
                
        if maxIDX - minIDX > rightThreshold:
            if currentDifference > maxEndDif:
                maxEndDif = currentDifference
                fingerEnd = step
                endFinger = (round(currentPoint[0]), round(currentPoint[1]))


    
        step += 1

    # p1 = abs(fingerStart - end)
    # p2 = abs(fingerEnd - end)
    # plotValueGraph(maxValues, minValues, maxIDXs, minIDXs, difference, p1, p2)

        
    return beginFinger, endFinger


def showFeatures(axes, gray_img, featurePoints):
    axes.imshow(gray_img, cmap='gray')
    axes.set_xticks([])
    axes.set_yticks([])
    for key in featurePoints:
        if key == 'F1':
            value = featurePoints[key]
            
            x_values = [value[0][0], value[1][0]]
            y_values = [value[1][1], value[1][1]]
            
            axes.plot(x_values , y_values, 'r--', linewidth =2)
            axes.plot(value[0][0], value[0][1], 'bo', markersize=1.5)
            axes.plot(value[1][0], value[1][1], 'bo', markersize=1.5)
        else:
            value = featurePoints[key]
            
            x_values1 = [value[0][0][0], value[0][1][0]]
            y_values1 = [value[0][0][1], value[0][1][1]]
            
            axes.plot(x_values1 , y_values1, 'r--', linewidth = 2)
            axes.plot(value[0][0][0], value[0][0][1], 'bo', markersize=1.5)
            axes.plot(value[0][1][0], value[0][1][1], 'bo', markersize=1.5)
            
            x_values2 = [value[1][0][0], value[1][1][0]]
            y_values2 = [value[1][0][1], value[1][1][1]]
            
            axes.plot(x_values2 , y_values2, 'r--', linewidth = 2)
            axes.plot(value[1][0][0], value[1][0][1], 'bo', markersize=1.5)
            axes.plot(value[1][1][0], value[1][1][1], 'bo', markersize=1.5)
                
    return axes

def scanAllFeatures(features, grayCopy):
    featureLabels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']

    featurePoints = {}
    featureDistances = []


    for label in featureLabels:
        print(f'Scanning feature: {label}', end='\r')
        currentFeature: Feature = features.getFeature(label)
        
        if label == 'F1':
            horizontalVector = [1, 0]
            scanSize = 44
            
            begin_hand, end_hand = scanFeatureR2L(
                grayCopy, 
                currentFeature.p1, 
                horizontalVector,  
                (-1) * (currentFeature.p1.x - scanSize - scanSize), 
                400, 
                scanSize, 13)

            distance = math.sqrt((begin_hand[0] - end_hand[0]) ** 2 + (begin_hand[1] - end_hand[1]) ** 2)
            
            featureDistances.append(distance)
            
            featurePoints[label] = (begin_hand, end_hand)
            
        elif label == 'F2':
            horizontalVector = [1, 0]
            scanSize = 33
            # plot only a horizontal line from the second point
            
            begin_finger_pt1, end_finger_pt1 = scanFeatureR2L(grayCopy, currentFeature.p1, currentFeature.unitDVector)
            
            
            begin_finger_pt2, end_finger_pt2 = scanFeatureL2R(
                grayCopy, 
                currentFeature.p2, 
                horizontalVector, 
                (-1) * (currentFeature.p2.x - scanSize - 400),
                400,
                scanSize, 
                13,
                1300,
                1500)

            distance1 = math.sqrt((begin_finger_pt1[0] - end_finger_pt1[0]) ** 2 + (begin_finger_pt1[1] - end_finger_pt1[1]) ** 2)
            distance2 = math.sqrt((begin_finger_pt2[0] - end_finger_pt2[0]) ** 2 + (begin_finger_pt2[1] - end_finger_pt2[1]) ** 2)
            
            featureDistances.append(distance1)
            featureDistances.append(distance2)

            
            featurePoints[label] = [(begin_finger_pt1, end_finger_pt1), (begin_finger_pt2, end_finger_pt2)]
            
            
        else:
            begin_finger_pt1, end_finger_pt1 = scanFeatureR2L(grayCopy, currentFeature.p1, currentFeature.unitDVector)
            begin_finger_pt2, end_finger_pt2 = scanFeatureR2L(grayCopy, currentFeature.p2, currentFeature.unitDVector, -250, 250, 28, 14, 1500, 1500)

            distance1 = math.sqrt((begin_finger_pt1[0] - end_finger_pt1[0]) ** 2 + (begin_finger_pt1[1] - end_finger_pt1[1]) ** 2)
            distance2 = math.sqrt((begin_finger_pt2[0] - end_finger_pt2[0]) ** 2 + (begin_finger_pt2[1] - end_finger_pt2[1]) ** 2)
            
            featureDistances.append(distance1)
            featureDistances.append(distance2)

            featurePoints[label] = [(begin_finger_pt1, end_finger_pt1), (begin_finger_pt2, end_finger_pt2)]
    
    print("Completed scanning labels")   
    return featureDistances, featurePoints
        

main()