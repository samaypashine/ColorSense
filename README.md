# ColorSense: Finger-Based Dominant Color Recognition
ColorSense is a project that leverages finger detection, point projection, and K-means clustering to recognize the dominant color in an image. By utilizing multi-threading and multi-processing techniques, ColorSense achieves real-time color analysis with enhanced efficiency.

## Key Features
1. Finger detection: ColorSense utilizes computer vision techniques to detect fingers within an image.
2. Point projection: The detected fingers are used to project a point within the image, focusing on the region of interest.
3. K-means clustering: ColorSense employs K-means clustering to extract the dominant color from the projected region.
4. Multi-threading: The project implements multi-threading to optimize the performance of concurrent tasks and enhance real-time color recognition.
5. Multi-processing: ColorSense utilizes multi-processing to leverage the power of multiple processors, further improving the speed and efficiency of color analysis.

## How It Works
1. Finger detection: The project employs computer vision algorithms to detect and track fingers in an image or video stream.
2. Point projection: Based on the detected fingers, ColorSense projects a point within the image, centering around the region of interest.
3. Region extraction: The projected point serves as the center for extracting a region of interest from the image.
4. Dominant color recognition: K-means clustering is applied to the extracted region, identifying the dominant color within that specific area.
5. Real-time analysis: The multi-threading and multi-processing techniques ensure that the color recognition process is efficient and operates in real-time.

## Requirements
1. Python (3.6.9)
2. OpenCV (4.7.0.72)
3. Mediapipe (0.8.2)
4. Numpy (1.19.3)
5. Webcolors (1.11.1)

## Installation and Usage
To use ColorSense, follow these steps:

1. Clone the repository: git clone https://github.com/samaypashine/ColorSense.git
2. Install the required dependencies: ```pip3 install -r requirements.txt```
3. Run the application: ```python3 colorsense_threading.py```
4. Wave your hand in-front of the camera to activate the feed to start detecting the finger and colors.
