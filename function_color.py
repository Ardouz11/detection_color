#importing modules
def function(color):	
	import cv2   
	import numpy as np
	i=0
	j=0
	k=0
	m=0
	if color=="red":
		while(1):
			_, img = cap.read()
			#img=cv2.resize(img,(1300,700))   
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)

			hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			img=cv2.blur(img,(5,5))

	#definig the range of red color
			red_lower=np.array([136,87,111],np.uint8)
			red_upper=np.array([180,255,255],np.uint8)

	#finding the range of red,blue and yellow color in the image
			red=cv2.inRange(hsv, red_lower, red_upper)
	 	
			kernal = np.ones((6 ,6), "uint8")

			red=cv2.dilate(red, kernal)
			res=cv2.bitwise_and(img, img, mask = red)

	#Tracking the Red Color
			contours,hierarchy=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
			for pic, contour in enumerate(contours):
				area = cv2.contourArea(contour)
				if(area>300):
					i=i+1
					x,y,w,h = cv2.boundingRect(contour)	
					img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
					cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
					print(' red detected ')
	
			cv2.imshow("Color Tracking",img)
    	#cv2.imshow("red",res) 	
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				print('number of red detected is ',i)
				print('number of blue detected is ',j)
				print('number of yellow detected is ',k)
				print('number of green detected is ',m)
				break 
	elif color=="blue":
		while(1):
			_, img = cap.read()
			#img=cv2.resize(img,(1300,700))   
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)

			hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			img=cv2.blur(img,(5,5))

	#definig the range of red color
			blue_lower=np.array([98,109,150],np.uint8)
			blue_upper=np.array([112,255,255],np.uint8)

	#finding the range of red,blue and yellow color in the image
			blue=cv2.inRange(hsv,blue_lower,blue_upper)
	 	
			kernal = np.ones((6 ,6), "uint8")

			blue=cv2.dilate(blue,kernal)
			res1=cv2.bitwise_and(img, img, mask = blue)

	#Tracking the Red Color
			contours,hierarchy=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			for pic, contour in enumerate(contours):
				area = cv2.contourArea(contour)
				if(area>300):
					j=j+1
					x,y,w,h = cv2.boundingRect(contour)	
					img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
					cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
					print(' blue detected ')
	
			cv2.imshow("Color Tracking",img)
    	
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				print('number of red detected is ',i)
				print('number of blue detected is ',j)
				print('number of yellow detected is ',k)
				print('number of green detected is ',m)
				break 
	elif color=="green":
		while(1):
			_, img = cap.read()
			#img=cv2.resize(img,(1300,700))   
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)

			hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			img=cv2.blur(img,(5,5))

	#definig the range of red color
			green_lower=np.array([25, 189, 118],np.uint8)
			green_upper=np.array([95, 255, 198],np.uint8)

	#finding the range of red,blue and yellow color in the image
			green=cv2.inRange(hsv, green_lower, green_upper)
	 	
			kernal = np.ones((6 ,6), "uint8")

			green=cv2.dilate(green, kernal)
			res=cv2.bitwise_and(img, img, mask = green)

	#Tracking the Red Color
			contours,hierarchy=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
			for pic, contour in enumerate(contours):
				area = cv2.contourArea(contour)
				if(area>300):
					m=m+1
					x,y,w,h = cv2.boundingRect(contour)	
					img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.putText(img,"GREEN color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
					print(' Green detected ')
	
			cv2.imshow("Color Tracking",img)
    	#cv2.imshow("red",res) 	
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				print('number of red detected is ',i)
				print('number of blue detected is ',j)
				print('number of yellow detected is ',k)
				print('number of green detected is ',m)
				break 
	else:
		while(1):
			_, img = cap.read()
			#img=cv2.resize(img,(1300,700))   
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)

			hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			img=cv2.blur(img,(5,5))

	#definig the range of red color
			yellow_lower=np.array([20,190,20],np.uint8)
			yellow_upper=np.array([35,255,255],np.uint8)

	#finding the range of red,blue and yellow color in the image
			yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
	 	
			kernal = np.ones((6 ,6), "uint8")

			yellow=cv2.dilate(yellow,kernal)
			res2=cv2.bitwise_and(img, img, mask = yellow) 

	#Tracking the Red Color
			contours,hierarchy=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			for pic, contour in enumerate(contours):
				area = cv2.contourArea(contour)
				if(area>300):
					k=k+1
					x,y,w,h = cv2.boundingRect(contour)	
					img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
					cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))  
					print(' yellow detected ')
	
			cv2.imshow("Color Tracking",img)
    	#cv2.imshow("red",res) 	
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				print('number of red detected is ',i)
				print('number of blue detected is ',j)
				print('number of yellow detected is ',k)
				print('number of green detected is ',m)
				break 
import cv2   
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("color")
args=parser.parse_args()
#capturing video through webcam
cap=cv2.VideoCapture(0)

    	#cv2.imshow("red",res) 	
function(args.color)
	 
	
          


    

    
