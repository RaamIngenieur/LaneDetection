// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main() {

	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap("C:/Users/Raam/OneDrive/R8/Moovita/Videos/full_20170821-17-47-35.avi");

	Mat frame,bwFrame, bwFHalf, bwFFHalf, topView, topViewFF, lambda;

	double zspan = 20, xspan = 10, step = 0.02, y0 = 1.6, widthOffset;
	int x1, y1, bwhWidth,bwhHeight;


	// Input Quadilateral or Image plane coordinates
	Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];

	// The 4 points that select quadilateral on the input , from top-left in clockwise order
	// These four pts are the sides of the rect box used as input 
	inputQuad[0] = Point2f(586, 60);
	inputQuad[1] = Point2f(696, 60);
	inputQuad[2] = Point2f(1250, 651);
	inputQuad[3] = Point2f(32, 651);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = Point2f(426, 451);
	outputQuad[1] = Point2f(576, 451);
	outputQuad[2] = Point2f(576, 951);
	outputQuad[3] = Point2f(426, 951);

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		char c = (char)waitKey(3000);
		return -1;
	}

	while (1) {

		// Capture frame-by-frame
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
			break;

		cvtColor(frame, bwFrame, CV_RGB2GRAY);

		bwhHeight = bwFrame.size().height/2;
		bwhWidth = bwFrame.size().width;

		bwFHalf = Mat(bwFrame, cv::Rect(0, bwhHeight, bwhWidth, bwhHeight));
		//flip(bwFHalf, bwFFHalf, 1);

		warpPerspective(bwFHalf, topView, lambda, Size(1001, 1000), INTER_LINEAR, BORDER_CONSTANT,Scalar(127));

		//flip(topViewFF, topView, 1);

		topView = Mat(topView, cv::Rect(250, 500, 501, 500));

		// Display the resulting frame
		imshow("Frame", topView);
		
		// Press  ESC on keyboard to exit
		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();

	return 0;
}
