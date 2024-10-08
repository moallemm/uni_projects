import cv2
import cvzone
import random
# cv2 and cvzone are open source computer vision librarys that are used for object detection and image proceccing
from cvzone.HandTrackingModule import HandDetector # pakage used for hand traking
import time # this is used for the timer of our game
cap = cv2.VideoCapture(0) #this will open up our camera 0 is the id of the our camera
cap.set(3,680)
cap.set(4,480)

detector=HandDetector(maxHands=1)#this will ditact hand movment of only one hand since maxhands is set to 1

# this is like a flag that will tell us if we have collected the result of theuser and the AI for comparison
timer=0#timer for game (intially set to 0)
state=False#intial state of the game
start=False # a flag that signals the start of the game
scores=[0,0]# [AI,Player]

while True:
    success, img= cap.read()#this will give us the image and success boolean 

    imgBG=cv2.imread("resources/BG.png")# reading the Background image from the resource file
    imagescale = cv2.resize(img,(0,0),None,0.875,0.875)#resizing the cam image to fit the BG
    imagescale= imagescale[:,80:480]# croping the the sides of the image box to fit BG

    #Finding hands (this is taken from the HandDetector source code ctrl+click)
    hands, img = detector.findHands(imagescale, draw=True, flipType=True)# we will use this to detect the hands in the images captured in imagescale
    if start:

        if state is False:#this if statments will only happen once
            timer= time.time() - intialTime
            cv2.putText(imgBG,str(int(timer)),(605,435),cv2.FONT_HERSHEY_PLAIN,6,(255,0,255),4)#placing a text on the presented screen on imgBG of the timer at (605,435) 
            if timer>3:
                state = True#after the timer hit 3 the AI and user results will be recorded
                timer=0#reset timer after taking results

                if hands:
                    playermove=None
                    hand= hands[0]#this will give us the hand that is detected
                    fingers=detector.fingersUp(hand)#checking how many fingers are up and returning it to the var fingers
                    print(fingers)

                    if fingers==[0,0,0,0,0]:#all 0s means rock 
                        playermove=0#0 for rock
                    if fingers==[1,1,1,1,1]:#all 1s means paper
                        playermove=1#1 for paper
                    if fingers==[0,1,1,0,0]:# 01100 means scissor
                        playermove=2#2 for scissor
                    randomnum= random.randint(0,2)
                    namearray=['rock','paper','scissors']
                    Aichoice=namearray[randomnum]
                    AIimg= cv2.imread(f'resources/{Aichoice}.png',cv2.IMREAD_UNCHANGED)# importing the unchanged img otherwise it won't read it
                    cvzone.overlayPNG(imgBG,AIimg,(149,310))#laying the png picture of the move over the imgBG

                    #player wins
                    if playermove==0 and randomnum==2 or playermove==1 and randomnum==0 or playermove==2 and randomnum==1:
                        scores[1] += 1
                    #AI wins
                    if playermove==2 and randomnum==0 or playermove==0 and randomnum==1 or playermove==1 and randomnum==2:
                        scores[0] += 1
                    #draw
                    if playermove==randomnum:
                        pass

                    print(playermove)

    imgBG[234:654,795:1195] =imagescale#placing the new camera image captured in the background image(BG) after choosing the scale or area we want to place it at
    #the first value(:) is hieght and the second is the width

    if state:
        cvzone.overlayPNG(imgBG,AIimg,(149,310))#this keeps the image inplace until next start


    cv2.putText(imgBG,str(scores[0]),(410,215),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),6)#placing a text on the presented screen on imgBG of the timer at (605,435) 
    cv2.putText(imgBG,str(scores[1]),(1112,215),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),6)#placing a text on the presented screen on imgBG of the timer at (605,435) 



    #cv2.imshow("Image",img)# showing the image(img) that we capture from the laptop camera
    cv2.imshow("BG", imgBG)#showing the BG
     
    #cv2.imshow("scaled", imagescale)#Showing the resized image of the cam

    key=cv2.waitKey(1)# wait time of 1 millisecond
    if key == ord('s'):#the game will start when we press the s key on our keyboard
        start=True
        intialTime=time.time()#the intial time is the time when the game starts
        state=False#reseting the state for repetition
