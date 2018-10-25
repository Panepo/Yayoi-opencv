// Yayoi-opencv-console.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <chrono>
#include <vector>

#include <opencv2\opencv.hpp>

typedef enum appState
{
	APPSTATE_EXIT,
	APPSTATE_NORMAL
} appState;

int main(int argc, char * argv[]) try
{
	appState state = APPSTATE_NORMAL;
	std::chrono::high_resolution_clock::time_point tmStart, tmEnd;
	std::chrono::high_resolution_clock::time_point time1, time2;
	std::chrono::duration<double> diff;
	
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow("./yayoi_srcnn_935_2x.pb");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	cv::Mat img = cv::imread("butterfly_GT.bmp", cv::IMREAD_COLOR);
	cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
	cv::imshow("original", img);

	int imgWidth = img.size().width;
	int imgHeight = img.size().height;
	
	cv::Mat imgCV;
	//cv::resize(img, imgCV, cv::Size(imgWidth / 2, imgHeight / 2), 0, 0, cv::INTER_CUBIC);
	cv::resize(img, imgCV, cv::Size(imgWidth * 2, imgHeight * 2), 0, 0, cv::INTER_CUBIC);

	cv::namedWindow("opencv", cv::WINDOW_AUTOSIZE);
	cv::imshow("opencv", imgCV);

	cv::Mat imgSR;
	cv::Mat imgSRsp[3];
	cv::cvtColor(imgCV, imgSR, cv::COLOR_BGR2YCrCb);
	cv::split(imgSR, imgSRsp);

	cv::Mat inputBlob = cv::dnn::blobFromImage(imgSRsp[0], 1.0f / 255, cv::Size(imgWidth, imgHeight), cv::Scalar(), false);
	net.setInput(inputBlob);

	time1 = std::chrono::high_resolution_clock::now();
	cv::Mat output = net.forward();
	time2 = std::chrono::high_resolution_clock::now();
	diff = time2 - time1;

	int outWidth = output.size().width;
	int outHeight = output.size().height;
	std::cout << outWidth << std::endl;
	std::cout << outHeight << std::endl;
	
	for (int i = 0; i < outWidth; i += 1) {
		for (int j = 0; j < outHeight; j += 1) {
			output.at<double>(i, j) = output.at<double>(i, j) * 255;
		}
	}
	
	for (int i = 0; i < outWidth; i += 1) {
		for (int j = 0; j < outHeight; j += 1) {
			int value = output.at<double>(i, j);
			if (value > 255) output.at<double>(i, j) = (double)255;
			if (value < 0) output.at<double>(i, j) = (double)0;
		}
	}

	for (int i = 0; i < outWidth; i += 1) {
		for (int j = 0; j < outHeight; j += 1) {
			imgSRsp[0].at<double>(i + 6, j + 6) = output.at<double>(i, j);
		}
	}

	std::vector<cv::Mat> channels;
	channels.push_back(imgSRsp[0]);
	channels.push_back(imgSRsp[1]);
	channels.push_back(imgSRsp[2]);
	cv::merge(channels, imgSR);
	cv::cvtColor(imgSR, imgSR, cv::COLOR_YCrCb2BGR);

	cv::namedWindow("srcnn", cv::WINDOW_AUTOSIZE);
	cv::imshow("srcnn", imgSR);
	
	while (state) {
		char key = cv::waitKey(10);
		if (key == 'q' || key == 'Q')
			state = APPSTATE_EXIT;
	}

	cv::destroyAllWindows();
	return EXIT_SUCCESS;
}
catch (const std::exception &error) {
	std::cerr << error.what() << std::endl;
	system("pause");
	return EXIT_FAILURE;
}
catch (...) {
	std::cerr << "Unknown/internal exception happened" << std::endl;
	system("pause");
	return EXIT_FAILURE;
}
