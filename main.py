import os
import sys
import cv2
import argparse
import logging
import time
import numpy as np
import csv
from time import gmtime, strftime

from darkflow.net.build import TFNet
import yolo_darkflow as yolo
import tf_openpose_detector

"""
todo:
- Selecting best detector => done
- counting people in specific Area (instant count) ==> done
- tracking ==> done
- counting people passing through specific area (accumalative) and waiting time ==> done
- report results to csv file == done
"""

saveArea = savePoint = False
mouseX = mouseY = 0
parser   = argparse.ArgumentParser()
parser.add_argument('--record', action='store_true' ,help='')
parser.add_argument('--videoCam', type=str, default='',help='')
INF = 1e6


def saveEventsToCsv(fileName,row,startNewFile = False):
	if startNewFile:
		mode = 'w'
	else:
		mode = 'a'
	with open(fileName, mode) as csvfile:
		eventWriter = csv.writer(csvfile, delimiter=',')
		eventWriter.writerow(row)


class Person:
	untakenId = 1
	def __init__(self):
		self.id = 0
	def yoloData(self,box,conf):
		self.box = box
		self.conf = conf
		self.pos = (int((box[0][0]+box[1][0])/2), int((box[0][1]+box[1][1])/2))



def id2color(colorId,maxColorId):

	colorSet = ["#012C58", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",

	"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
	"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
	"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
	"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
	"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
	"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
	"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

	"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
	"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
	"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
	"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
	"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
	"#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
	"#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
	"#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#FFFFFF"]

	maxColorId = min(128,maxColorId)
	colorId = colorId % maxColorId
	if(colorId<0):
		colorId = 127;
	else:
		colorId = colorId * (128 / maxColorId);

	colorId = int(colorId)
	c1 = int(colorSet[colorId][1:3], 16)
	c2 = int(colorSet[colorId][3:5], 16)
	c3 = int(colorSet[colorId][5:7], 16)
	colorCmps = (c1,c2,c3)

	return colorCmps



def readRecordVideo(videoDir,record = False):
	if videoDir == '0':
		videoCapture = cv2.VideoCapture(0)
	else:
		videoCapture = cv2.VideoCapture(videoDir)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	isOpened = videoCapture.isOpened()
	width  = int( videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH ) )
	height = int( videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) )
	fps    = int( videoCapture.get(cv2.CAP_PROP_FPS) )
	
	if record:
		outVideo = cv2.VideoWriter('output.avi',fourcc, fps, (width,height))
	else:
		outVideo = None

	return videoCapture, outVideo, isOpened, fps


def visualizeDetections(frame,people):
	#frame = yolo.drawBoxes(yoloDetections, frame, ["person"])
	frame = drawBoxes(frame,people)
	cv2.imshow("output",frame)
	cv2.waitKey(100)
	return frame


def mouseEvent(event,x,y,flags,param):
	global mouseX,mouseY,savePoint,saveArea
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		savePoint = True
	elif event == cv2.EVENT_LBUTTONDBLCLK:
		saveArea = True


def drawAreas(frame, detectionAreas):
	frameCpy = frame.copy()
	for detectionArea in detectionAreas:
		if len(detectionArea)>1:
			cv2.fillPoly(frameCpy,[np.array(detectionArea, np.int32)],(0,255,0))
			cv2.polylines(frameCpy,[np.array(detectionArea, np.int32)],True,(0,0,255), thickness=3)
	return cv2.addWeighted(frame,0.7,frameCpy,0.3,0)


def readAreas(frame):
	global mouseX,mouseY,savePoint,saveArea
	windowTitle = 'Please locate detection areas'
	cv2.namedWindow(windowTitle)
	cv2.setMouseCallback(windowTitle,mouseEvent)
	detectionAreas = []
	detectionArea  = []

	while True:
		drawingFrame = frame.copy()
		drawingFrame = drawAreas(drawingFrame,detectionAreas)
		drawingFrame = drawAreas(drawingFrame,[detectionArea])		
		cv2.imshow(windowTitle,drawingFrame)
		c = cv2.waitKey(10)
		if c == 27:
			break
		if savePoint:
			savePoint = False
			detectionArea.append((mouseX,mouseY))
		if saveArea:
			saveArea = False
			detectionAreas.append(detectionArea)
			detectionArea = []

	return detectionAreas


def checkIfPointInsideArea(area,point):
	dis = cv2.pointPolygonTest(np.array(area, np.int32),point,True)
	isInside = dis>0
	return  isInside


def yoloDetections2people(yoloDetections):
	people = []
	for detection in yoloDetections:
		person = Person()
		box, conf, label = yolo.parseDetection(detection)
		if label == 'person':
			person.yoloData(box,conf)
			people.append(person)
	
	return people


def countPeopleInEachArea(areas,people):
	counts = []
	for area in areas:
		count = 0
		for person in people:
			if checkIfPointInsideArea(area,person.pos):
				count += 1
		counts.append(count)
	return counts


def drawBoxes(frame, people):

	for person in people:
		txtBoxW = 20
		txtBoxL = 150
		boxTl = person.box[0]
		boxBr = person.box[1]
		personColor = id2color(person.id,100)
		cv2.rectangle(frame, (boxTl[0],boxTl[1]-txtBoxW),(boxTl[0]+txtBoxL,boxTl[1]) , personColor, -1)
		cv2.rectangle(frame, (boxTl[0],boxTl[1]-txtBoxW),(boxTl[0]+txtBoxL,boxTl[1]) , personColor, 6)
		cv2.rectangle(frame, boxTl, boxBr, personColor, 6)
		cv2.rectangle(frame, person.pos, person.pos, (0, 0, 0), 12)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,"Id: "+ str(person.id)+ "  " + str(int(100*person.conf)) +'%',
			(boxTl[0],boxTl[1]), font, 0.8,(0,0,0),2,cv2.LINE_AA)

	return frame

def disTuple(pos1,pos2):
	dis = int(((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5)
	return dis


def tracker(lastDetected, currentDetected):
	minThresholdNewDetected = 0.6
	minThresholdTracked = 0.3
	minDisTh = 100
	approvedDetections = []
	reservedLastDetections = []
	currentDetectedAssigned2Tracks = []

	for cdIdx,cd in enumerate(currentDetected):
		minDis = INF
		bestLastDetectionIdx = 0
		for ldIdx,ld in enumerate(lastDetected):
			dis = disTuple(cd.pos,ld.pos)
			if ldIdx in reservedLastDetections:
				continue
			if dis < minDis and cd.conf > minThresholdTracked:
				minDis = dis
				bestLastDetectionIdx = ldIdx
		if minDis < minDisTh:
			cd.id = lastDetected[bestLastDetectionIdx].id
			approvedDetections.append(cd)
			currentDetectedAssigned2Tracks.append(cdIdx)
			reservedLastDetections.append(bestLastDetectionIdx)

	for cdIdx,cd in enumerate(currentDetected):
		if not(cdIdx in currentDetectedAssigned2Tracks) and (cd.conf > minThresholdNewDetected):
			cd.id = Person.untakenId
			Person.untakenId = Person.untakenId + 1
			approvedDetections.append(cd)


	return approvedDetections


def recordTimeSpent(trackedPeople, areas, fps, cameraId):
	# event: (areaIdx, personId) ==> spent time
	# update waiting time for events
	for trackedPerson in trackedPeople:
		for areaIdx,area in enumerate(areas):
			if checkIfPointInsideArea(area,trackedPerson.pos):
				pair = (areaIdx, trackedPerson.id)
				if pair in recordTimeSpent.history:
					recordTimeSpent.history[ pair ] = recordTimeSpent.history[ pair ] + 1
				else:
					recordTimeSpent.history[ pair ] = 1
	# save waiting time
	pairsToBeSaved = []

	for pair in recordTimeSpent.history:
		personId = pair[1]
		isPersonStillExist = False
		for trackedPerson in trackedPeople:
			if trackedPerson.id == personId:
				isPersonStillExist = True
		if not isPersonStillExist:
			pairsToBeSaved.append(pair)

	for pairToBeSaved in pairsToBeSaved:
		print(pairToBeSaved,recordTimeSpent.history[pairToBeSaved])
		time = recordTimeSpent.history[pairToBeSaved] / fps
		currentTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		event = [cameraId, pairToBeSaved[0], pairToBeSaved[1], time, currentTime]
		saveEventsToCsv('events.csv', event)
		del recordTimeSpent.history[pairToBeSaved]


recordTimeSpent.history = {}


def main(FLAGS):
	readAreasF = True
	videoCapture, outVideo, isOpened, fps = readRecordVideo(FLAGS.videoCam, FLAGS.record)
	yoloDetector = yolo.getYoloDetector(threshold = 0.1)
	peopleLastDetected = []
	saveEventsToCsv('events.csv',['Camera Id','Area Idx','Person Idx','waiting Time(s)','End time'],True)
	#tfOpenpose = tf_openpose_detector.tfOpenpose()

	while isOpened:
		ret, frame = videoCapture.read()
		#frame = cv2.resize(frame, (int(frame.shape[1]*3),int(frame.shape[0]*3)))
		c = cv2.waitKey(10)
		if c == 27:
			break

		if not ret:
			break
		if readAreasF:
			detectionAreas = readAreas(frame)
			readAreasF = False
		yoloDetections = yoloDetector.return_predict(frame)
		people = yoloDetections2people(yoloDetections)
		trackedPeople = tracker(peopleLastDetected, people)
		peopleLastDetected = trackedPeople
		areaCounts = countPeopleInEachArea(detectionAreas,trackedPeople)

		font = cv2.FONT_HERSHEY_SIMPLEX
		for areaIdx,areaCount in enumerate(areaCounts):
			cv2.putText(frame,"Area " + str(areaIdx)+" : " +str(areaCount),(10,30+areaIdx*25), 
				font, 0.8,(0,255,0),2,cv2.LINE_AA)

		recordTimeSpent(trackedPeople,detectionAreas,fps,0)
		frame = drawAreas(frame,detectionAreas)
		visualizeDetections(frame, trackedPeople)
		#tfOpenpose.detect(frame)

		if FLAGS.record:
			outVideo.write(frame)


	videoCapture.release()
	if FLAGS.record:	
		outVideo.release()



if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)