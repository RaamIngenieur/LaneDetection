// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2\core\cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main() {

	cuda::GpuMat GFHalf, topViewG,blurredG,sharpenedG,threshG, smallLaneG, largeLaneG, thinLinesSG, thinLinesLG, thinLinesG, bwTopViewG, thinLinesGU,linesG,newTopViewG, resultViewG;
	Mat frame, FHalf, lambda,opFrame,smallLane,largeLane,topView;

	vector<Vec4i> lines;

	double zspan = 20, xspan = 10, step = 0.02, y0 = 1.6, widthOffset;
	int hWidth,hHeight;
	Ptr<cuda::Filter> gaussSharp,gaussBlur;
	Ptr<cuda::TemplateMatching> NCCR = cuda::createTemplateMatching(threshG.type(), CV_TM_CCORR_NORMED, Size(0,0));
	Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1, CV_PI/180, 70, 5, 200);


	// Input Quadilateral or Image plane coordinates
	Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];

	// The 4 points that select quadilateral on the input , from top-left in clockwise order
	// These four pts are the sides of the rect box used as input 
	inputQuad[0] = Point2f(14, 677);
	inputQuad[1] = Point2f(1268, 677);
	inputQuad[2] = Point2f(743, 117);
	inputQuad[3] = Point2f(539, 117);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = Point2f(426, 951);
	outputQuad[1] = Point2f(576, 951);
	outputQuad[2] = Point2f(576, 701);
	outputQuad[3] = Point2f(426, 701);

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);

	//Design filters
	gaussSharp = cuda::createGaussianFilter(topViewG.type(), sharpenedG.type(), Size(31,31), 10, 10, BORDER_REPLICATE);
	gaussBlur = cuda::createGaussianFilter(sharpenedG.type(), blurredG.type(), Size(0,0), 1, 1, BORDER_REPLICATE);

	//Read filter templates
	smallLane = imread("C:/Users/Raam/OneDrive/R8/Moovita/Videos/laneTemplateSmall3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	largeLane = imread("C:/Users/Raam/OneDrive/R8/Moovita/Videos/laneTemplateSmall2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//Load templates to GPU
	smallLaneG.upload(smallLane);
	largeLaneG.upload(largeLane);

	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap("C:/Users/Raam/OneDrive/R8/Moovita/Videos/full_20170821-17-47-35.avi");

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

		hHeight = frame.size().height/2;
		hWidth = frame.size().width;

		FHalf = Mat(frame, Rect(0, hHeight, hWidth, hHeight));

		GFHalf.upload(FHalf);

		
		//Inverse perspective projection
		cuda::warpPerspective(GFHalf, topViewG, lambda, Size(1001, 1000), INTER_LINEAR, BORDER_REPLICATE, Scalar(127), cuda::Stream::Null());
		
		topViewG.download(topView);

		cuda::cvtColor(topViewG, bwTopViewG, CV_RGB2GRAY);

		//Sharpen
		gaussSharp->apply(bwTopViewG, blurredG);
		cv::cuda::addWeighted(bwTopViewG, 2.0, blurredG, -1.0, 0, sharpenedG);
		gaussSharp->apply(sharpenedG, blurredG);
		cv::cuda::addWeighted(sharpenedG, 2.0, blurredG, -1.0, 0, sharpenedG);
		gaussSharp->apply(sharpenedG, blurredG);
		cv::cuda::addWeighted(sharpenedG, 2.0, blurredG, -1.0, 0, sharpenedG);

		//Blur
		gaussBlur->apply(sharpenedG, blurredG);
		gaussBlur->apply(blurredG, sharpenedG);
		gaussBlur->apply(sharpenedG, blurredG);

		//threshold
		cuda::threshold(blurredG, threshG, 220, 255, THRESH_BINARY, cuda::Stream::Null());

		//Thin lines
		NCCR->match(threshG, smallLaneG, thinLinesSG, cuda::Stream::Null());
		cuda::threshold(thinLinesSG, thinLinesSG, 0.4, 255, THRESH_BINARY, cuda::Stream::Null());
		cuda::copyMakeBorder(thinLinesSG, thinLinesSG, 10, 10, 10, 10, BORDER_CONSTANT, 0);

		thinLinesSG.convertTo(thinLinesGU, CV_8UC1);

		NCCR->match(thinLinesGU, largeLaneG, thinLinesLG, cuda::Stream::Null());
		cuda::threshold(thinLinesLG, thinLinesLG, 0.4, 255, THRESH_BINARY, cuda::Stream::Null());

		cuda::copyMakeBorder(thinLinesLG, thinLinesLG, 10, 10, 10, 10, BORDER_CONSTANT, 0);

		//cuda::addWeighted(thinLinesSG, 255, thinLinesLG, 255, 0, thinLinesG);

		//Hough

		thinLinesSG.convertTo(thinLinesGU, CV_8UC1);

		
		hough->detect(thinLinesGU, linesG);

		if (linesG.size() != Size(0, 0))
		{
			linesG.download(lines);
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(topView, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(230, 50, 0), 2, CV_AA);
			}
		}
		
		
		
		//thinLinesSG.download(opFrame);

		newTopViewG.upload(topView);


		//Perspective projection
		cuda::warpPerspective(newTopViewG,resultViewG, lambda, Size(hWidth, hHeight), INTER_LINEAR+WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar(127), cuda::Stream::Null());

		resultViewG.download(opFrame);

		// Display the resulting frame
		imshow("Frame", opFrame);
		
		// Press  ESC on keyboard to exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();

	return 0;
}
