import cv2
import pyautogui
from time import time
from math import hypot
#Multidimensional Euclidean distance from the origin to a point.

import mediapipe as mp

import matplotlib.pyplot as plt

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# Initialize mediapipe pose class
mp_pose=mp.solutions.pose

#setup with image
pose_image=mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5,model_complexity=1)

#set up for video
pose_video=mp_pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

#for drawing the line
mp_drawing=mp.solutions.drawing_utils

#face dected 
mp_face_detection = mp.solutions.face_detection

# Function to check left and right eye position
def checkLeftRight_EYE(image, results, draw=False, display=False):
    # horizontal position (left, right, and center) of person
    horizontal_position_eye = None

    # get height and width of the image
    height, width, _ = image.shape
    output_image = image.copy()

    # Retrieve the x-coordinate of the right eye landmark.
    right_eye_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * width)

    # Retrieve the x-coordinate of the left eye landmark.
    left_eye_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * width)

    # if person at left then x-coordinate is less than or equal to the x-coordinate of the center of the image.
    if (right_eye_x <= width//2 and left_eye_x <= width//2):
        # Set the person's position to left.
        horizontal_position_eye = 'Left_tilt'

    # Same as right
    elif (right_eye_x >= width//2 and left_eye_x >= width//2):
        # Set the person's position to right.
        horizontal_position_eye = 'Right_tilt'
    
    # Check if the person is at the center that is when right shoulder landmark x-coordinate is greater than or equal to
    # and left shoulder landmark x-coordinate is less than or equal to the x-coordinate of the center of the image.
    elif (right_eye_x >= width//2 and left_eye_x <= width//2):
        # Set the person's position to center.
        horizontal_position_eye = 'Center'

    # check if the person's horizontal position and draw a line at the center of the image
    if draw:
        # Write the horizontal position of the person on the image
        cv2.putText(output_image, horizontal_position_eye, (5, height - 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
        # Draw a line at the center of the image.
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)

    if display:
        # Display the output image.
        pass
       # plt.figure(figsize=[10, 10])
        # plt.imshow(output_image[:,:,::-1])
        # plt.title("Output Image")
        # plt.axis('off')
    
    # Return the output image and the person's horizontal position.
    return output_image, horizontal_position_eye



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def detectPose(image,pose, draw=False,display=False):
    
    # copy of input image
    output_image=image.copy()

    #bgr to rgb
    imageRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # deceted the pose
    results =pose.process(imageRGB)  # ya rgb image sa coordinate nikal ga
     #or coordinate wapasw sa image ko dana hoga draw karna ka leya


    # draw the land mark
    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(
            image=output_image, landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS, # mp_pose uupar define ha
            landmark_drawing_spec=mp_drawing.DrawingSpec( color=(255,255,255),
                                                         thickness=3, circle_radius=3
                                                         ),    # or mp_drawing uupar definr ha#for drawing the linn
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                            thickness=2, 
                                                            circle_radius=2
                                                            )                                                        
                                )
    #for checking the original input image and  resultant
    if display:
        # Display the original input image and the resultant image.
        pass


        # plt.figure(figsize=[22,22])
        # plt.subplot(121)
        # plt.imshow(image[:,:,::-1])
        # plt.title("Original Image");plt.axis('off');
        # plt.subplot(122);
        # plt.imshow(output_image[:,:,::-1]);
        # plt.title("Output Image");plt.axis('off');


    else:
# Return the output image and the results of pose landmarks detection.
        return output_image , results







#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def checkHandsJoined(image,results,draw=False,display=False):
    height , width, _ =image.shape
    # copy of image
    output_image= image.copy()

    # getting lest wtrist landmark x and y coordinate
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    # Get the right wrist landmark x and y coordinates.
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    # ecular distance
    euclidean_distance=int(hypot
                           (
                               left_wrist_landmark[0]-right_wrist_landmark[0],
                            left_wrist_landmark[1]-right_wrist_landmark[1]
                            )
                            )
     # Compare the distance
    if euclidean_distance <130:
       
       #set hand join
       hand_status='Hands Joined'
       color=(0,255,0)

       #otherwise
    else:
        hand_status='Hands Not Joined'
        color=(0,255,0)
    #hands joined status and hand distance
    if draw:
        cv2.putText(output_image, hand_status,(10,30),cv2.FONT_HERSHEY_PLAIN, 2, color ,3)

        # write the equiler distance
        cv2.putText(
            output_image, f'Distance: {euclidean_distance}' , (10,70) ,
            cv2.FONT_HERSHEY_PLAIN , 2, color ,3
        )
    #cheak if the output image is specified to be display

    if display:
        #display the output image
        pass

        # plt.figure(figsize=[10,10])
        # plt.imshow(output_image[:,:,::-1])
        # plt.title("Output Image");plt.axis('off');
    #other wise
    else:
        return output_image , hand_status
    #side sa add karna para retyurn statement
    return output_image , hand_status





#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



def checkLeftRight(image, results, draw=False, display=False):
    # horizontal position (left ,right and center) of persion

    horizontal_position=None
    #get hight and width of image
    height, width,_=image.shape
    output_image=image.copy()
      # Retreive the x-coordinate of the left shoulder landmark.
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

    # Retreive the x-corrdinate of the right shoulder landmark.
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    # if persion at left then x-coordinate is less than than or equal to the x-corrdinate of the center of the image.
    if (right_x <= width//2 and left_x <= width//2):
            
        # Set the person's position to left.
        horizontal_position = 'Left'

    # Same as right
    elif (right_x >= width//2 and left_x >= width//2):
        
        # Set the person's position to right.
        horizontal_position = 'Right'
    
    # Check if the person is at center that is when right shoulder landmark x-corrdinate is greater than or equal to
    # and left shoulder landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
    elif (right_x >= width//2 and left_x <= width//2):
        
        # Set the person's position to center.
        horizontal_position = 'Center'

    #cheak if the person horizontal postion and line and ceter of image

    if draw:
        # Write the horizontal position of the person on the image
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
 # Draw a line at the center of the image.
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    if display:
    
        # Display the output image.
        pass
        # plt.figure(figsize=[10,10])
        # plt.imshow(output_image[:,:,::-1]);
        # plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the person's horizontal position.
        return output_image, horizontal_position


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




def checkJumpCrouch(image, results,MID_Y=250, draw=False, display=False):
    #get hight and width of images

    height, width , _ =image.shape
    # create a copy of image
    output_image=image.copy()
    # y-coordinate of the left shoulder landmark.
    left_y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    # y-coordinate of the right shoulder landmark.
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    # Calculate the y-coordinate of the mid-point of both shoulders.
    actual_mid_y = abs(right_y + left_y) // 2
    #calculating the upper and lower bond of the therosold
    lower_bound= MID_Y -20
    upper_bound = MID_Y + 100
       
    # Check if the person has jumped that is when the y-coordinate of the mid-point 
    # of both shoulders is less than the lower bound.
    if (actual_mid_y < lower_bound):
        
        # Set the posture to jumping.
        posture = 'Jumping'
    
    # Check if the person has crouched that is when the y-coordinate of the mid-point 
    # of both shoulders is greater than the upper bound.
    elif (actual_mid_y > upper_bound):
        
        # Set the posture to crouching.
        posture = 'Crouching'
    
    # Otherwise the person is standing and the y-coordinate of the mid-point 
    # of both shoulders is between the upper and lower bounds.    
    else:
        
        # Set the posture to Standing straight.
        posture = 'Standing'
    if draw:
    
        # Write the posture of the person on the image. 
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Draw a line at the intial center y-coordinate of the person (threshold).
        cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        pass
        # plt.figure(figsize=[10,10])
        # plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and posture indicating whether the person is standing straight or has jumped, or crouched.
        return output_image, posture
    #retur will add by me
    return output_image, posture


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Create named window for resizing purposes.
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
 
# Initialize a variable to store the time of the previous frame.
time1 = 0

# Initialize a variable to store the state of the game (started or not).
game_started = False   

# Initialize a variable to store the index of the current horizontal position of the person.
# At Start the character is at center so the index is 1 and it can move left (value 0) and right (value 2).
x_pos_index = 1

# Initialize a variable to store the index of the current vertical posture of the person.
# At Start the person is standing so the index is 1 and he can crouch (value 0) and jump (value 2).
y_pos_index = 1

# Declate a variable to store the intial y-coordinate of the mid-point of both shoulders of the person.
MID_Y = None

# Initialize a counter to store count of the number of consecutive frames with person's hands joined.
counter = 0

# Initialize the number of consecutive frames on which we want to check if person hands joined before starting the game.
num_of_frames = 5

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Perform the pose detection on the frame.
    frame, results = detectPose(frame, pose_video, draw=game_started)

   # cheaking the landmarks in frames
    if results.pose_landmarks:

        if game_started:
            #for horizontal command
            frame, horizontal_position =checkLeftRight(frame,results,draw=True)

            #move left from center and center from right
            if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):

                #press utokey
                ####################################################################
                pyautogui.press('left')

                #update horizontal
                x_pos_index -=1

                
            elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                
                # Press the right arrow key.
                ##################################################################
                pyautogui.press('right')
                
                # Update the horizontal position index of the character.
                x_pos_index += 1

                ##########################################################################################################
                # for eye movement

            frame,  horizontal_position_eye=checkLeftRight_EYE(frame, results, draw=True)
            
            if (horizontal_position_eye=='Left_tilt'):
                pyautogui.press('z')  

            if (horizontal_position_eye=='Right_tilt'):
                pyautogui.press('x')







        # Otherwise if the game has not started    
        else:
            
            # Write the text representing the way to start the game on the frame. 
            cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)
            
         # command to start the game and pause the game
 
        #result ma pose aaraha ha
        if checkHandsJoined(frame, results)[1] =='Hands Joined':

            counter +=1
            if counter==num_of_frames:
                # cheak if game not start
                if not(game_started):
                    game_started=True
                    # Retreive the y-coordinate of the left shoulder landmark.
                    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

                    # Retreive the y-coordinate of the right shoulder landmark.
                    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)

                    # Calculate the intial y-coordinate of the mid-point of both shoulders of the person.
                    MID_Y = abs(right_y + left_y) // 2

                    # Move to 1300, 800, then click the left mouse button to start the game.
                    pyautogui.click(x=1300, y=800, button='left')


                else:
    
                    # Press the space key.
                    pyautogui.press('space')

                counter =0

        else:
            counter =0
        #------------------------------------------------------------------------------------------------------------------

        # Commands to control the vertical movements of the character.
        #------------------------------------------------------------------------------------------------------------------
        # cheaking the y coordinate of both solder
        if MID_Y:
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)

            # for jump
            if posture == 'Jumping' and y_pos_index==1:
                ###########################################################
                pyautogui.press('up')

                y_pos_index+=1

            elif posture == 'Crouching' and y_pos_index == 1:
    
                # Press the down arrow key
                #########################################################################
                pyautogui.press('down')
                
                # Update the veritcal position index of the character.
                y_pos_index -= 1

            # Check if the person has stood.
            elif posture == 'Standing' and y_pos_index   != 1:
                
                # Update the veritcal position index of the character.
                y_pos_index = 1
        
        #------------------------------------------------------------------------------------------------------------------

    # Otherwise if the pose landmarks in the frame are not detected.       
    else:

        # Update the counter value to zero.
        counter = 0            


    # Calculate the frames updates in one second

    #set the time for frame

    time2=time()

    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2




    #----------------------------------------------------------------------------------------------------------------------
    
    # Display the frame.            
    cv2.imshow('Pose Detection', frame)
    
    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k=cv2.waitKey(1) & 0xFF
    

    if k==ord('q'):
        break



# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()





#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@