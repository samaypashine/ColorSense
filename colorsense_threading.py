# /**
#  * @file colorsense_threading.py
#  * @author Samay Pashine
#  * @brief File combining the hand pose detection, extracting region of interest, and recognizing the color in ROI using multi-threading
#  * @version 1.0
#  * @date 2023-06-12
#  * 
#  * @copyright Copyright (c) 2023
#  * 
#  */

# Importing necessary libraries
import os
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from threading import Thread
import webcolors
import logging
import math

# Class to capture frames in different thread.
class ThreadedCamera(object):
    def __init__(self, source=0):

        self.capture = cv2.VideoCapture(source)
        time.sleep(2)
        self.thread = Thread(target=self.update, args=())

        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                self.capture.grab()
                time.sleep(0.005)

    def grab_frame(self):
        _, img = self.capture.retrieve()
        return img

# Function to calculate the shade inside the ROI by averaging the pixel values.
def get_avg_color_shade(image, x, y, radius):
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the region of interest
    roi = lab_image[max(0, y-radius):y+radius, max(0, x-radius):x+radius]

    # Calculate the average color in the LAB color space
    average_lab_color = np.mean(roi, axis=(0, 1))

    # Convert the average color back to the BGR color space
    average_rgb_color = cv2.cvtColor(np.uint8([[average_lab_color]]), cv2.COLOR_LAB2RGB)[0][0]

    return average_rgb_color

# Functino to calculate the shade inside ROI using K-means clustering.
def get_dominant_color_shade(image, x, y, radius):
    # Extract the region of interest
    roi = image[max(0, y-radius):y+radius, max(0, x-radius):x+radius]

    # Reshape the ROI image to a 2D array of pixels
    pixels = roi.reshape(-1, 3)

    # Convert the pixel values to float32 for k-means clustering
    pixels = np.float32(pixels)

    # Define the criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the center value to integer RGB values
    dominant_color = np.uint8(centers[0])
    dominant_color = (dominant_color[2], dominant_color[1], dominant_color[0])

    return dominant_color

# Function to find the name of the closest related color of the shade.
def get_color_name(rgb_color):
    closest_color = None
    min_distance = float('inf')

    for color_name, color_rgb in webcolors.CSS3_NAMES_TO_HEX.items():
        color_rgb = webcolors.hex_to_rgb(color_rgb)
        distance = color_distance(rgb_color, color_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

# Function to find difference between the shades.
def color_distance(color1, color2):
    return abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])

# Driver code to initiate the application.
if __name__ == "__main__":
    # Configuring the log handler.
    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Initial Parameters
    DEBUG = True
    cap = ThreadedCamera(0)
    mp_Hands = mp.solutions.hands
    hands = mp_Hands.Hands()
    mpDraw = mp.solutions.drawing_utils
    finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
    thumb_Coord = (4,2)
    cvg_radius = 20
    distance_new_point = 50
    fps_history = list()

    while True:
        try:
            star_time = time.time()
            # Grabbing the image from the Thread and converting color type from BGR to RGB.
            image = cap.grab_frame()
            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Looking if a hand is in the frame or not.
            results = hands.process(RGB_image)
            multiLandMarks = results.multi_hand_landmarks

            # Only executing the color recognition when detects a hand in the frame.
            if multiLandMarks:
                handLms = multiLandMarks[0]
                
                # Calculating the positon of each hand landmark for the image.
                h, w, c = image.shape
                handList = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
                
                # Taking Two point on the Index finger to project the region of interest.
                x1, x2 = handList[7][0], handList[8][0]
                y1, y2 = handList[7][1], handList[8][1]

                # Distance between point A and point B
                distance_AB = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Unit vector from A to B
                unit_vector_AB = [(x2 - x1) / distance_AB, (y2 - y1) / distance_AB]

                # Coordinates of the new point
                x_new = int(x2 + unit_vector_AB[0] * distance_new_point)
                y_new = int(y2 + unit_vector_AB[1] * distance_new_point)

                # Get the color Shade and the color name of the projected point
                shade = get_dominant_color_shade(image, x_new, y_new, cvg_radius)
                name = get_color_name(shade)
                
                # Logging the information of the program like the coordinates of the point, calculated. color shade, and name, etc.
                logging.info("#####################################################################################################")
                if DEBUG:
                    mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
                    logging.info(f"Distance Between Index Tip from Mid Point    : {distance_AB}")
                    logging.info(f"Projected location of Color Coverage Point   : ({x_new}, {y_new})")
                
                fps_history.append(int(1 / (time.time() - star_time)))
                logging.info(f"Color RGB Detected                           : {shade}")
                logging.info(f"Color Name                                   : {name}")
                logging.info(f"FPS Rate                                     : {int(np.mean(fps_history))}")

                # Drawing the circle around the projected point, and showing the image
                cv2.circle(image, (x_new, y_new), cvg_radius, (119, 204, 22), 3)
                cv2.imshow("Color Recognition", image)
                key = cv2.waitKey(1)

                # Break condition to exit the code.
                if key == ord('q'):
                    logging.info('Received the KILL signal. Stopping the program.')
                    logging.info('==============================================:)=====================================================')
                    
                    break
        except Exception as e:
            if DEBUG:
                logging.error(e)
            continue
