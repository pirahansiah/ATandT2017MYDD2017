// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Sample of using OpenCV dnn module with Tensorflow Inception model.
*/
// Farshid Pirahansiah
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//using namespace cv;
using namespace cv::dnn;
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <string>
#include <locale>
#pragma comment(lib,"ws2_32.lib")
using namespace std;

string website_HTML;
locale local;

//***************************
void get_Website(char *url)
{
	WSADATA wsaData;
	SOCKET Socket;
	SOCKADDR_IN SockAddr;


	int lineCount = 0;
	int rowCount = 0;

	struct hostent *host;
	char *get_http = new char[256];

	memset(get_http, ' ', sizeof(get_http));
	strcpy(get_http, "GET / HTTP/1.1\r\nHost: ");
	strcat(get_http, url);
	strcat(get_http, "\r\nConnection: close\r\n\r\n");

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		cout << "WSAStartup failed.\n";
		system("pause");
		//return 1;
	}

	Socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	host = gethostbyname(url);

	SockAddr.sin_port = htons(80);
	SockAddr.sin_family = AF_INET;
	SockAddr.sin_addr.s_addr = *((unsigned long*)host->h_addr);

	cout << "Connecting to " << url << " ...\n";

	//if (connect(Socket, (SOCKADDR*)(&SockAddr), sizeof(SockAddr)) != 0)
	//{
	//	cout << "Could not connect";
	//	system("pause");
	//	//return 1;
	//}

	//cout << "Connected.\n";
	//send(Socket, get_http, strlen(get_http), 0);

	//char buffer[10000];

	//int nDataLength;
	///*while ((nDataLength = recv(Socket, buffer, 10000, 0)) > 0)
	//{
	//	int i = 0;

	//	while (buffer[i] >= 32 || buffer[i] == '\n' || buffer[i] == '\r')
	//	{
	//		website_HTML += buffer[i];
	//		i += 1;
	//	}
	//}*/
	//closesocket(Socket);
	//WSACleanup();

	//delete[] get_http;
}

const cv::String keys =
"{help h    || Sample app for loading Inception TensorFlow model. "
"The model and class names list can be downloaded here: "
"https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip }"
"{model m   |tensorflow_inception_graph.pb| path to TensorFlow .pb model file }"
"{image i   || path to image file }"
"{i_blob    | input | input blob name) }"
"{o_blob    | softmax2 | output blob name) }"
"{c_names c | imagenet_comp_graph_label_cv::cv::Strings.txt | path to file with classnames for class id }"
"{result r  || path to save output blob (optional, binary format, NCHW order) }"
;

void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb);
std::vector<cv::String> readClassNames(const char *filename);

int main(int argc, char **argv)
{
	//get_Website("http://192.168.2.9:8080/object?object?id=1");

	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	cv::String modelFile = "C:/opencv33/FarshidPirahanSiah/FarshidPirahanSiah/tensorflow_inception_graph.pb";
	cv::String imageFile = "C:/opencv33/FarshidPirahanSiah/FarshidPirahanSiah/space_shuttle.jpg";
	cv::String inBlobName = "input";// ".input";
	cv::String outBlobName = "softmax2";
	/*cv::String modelFile = parser.get<cv::String>("model");
	cv::String imageFile = parser.get<cv::String>("image");
	cv::String inBlobName = parser.get<cv::String>("i_blob");
	cv::String outBlobName = parser.get<cv::String>("o_blob");*/

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	cv::String classNamesFile = "C:/opencv33/FarshidPirahanSiah/FarshidPirahanSiah/imagenet_comp_graph_label_strings.txt";   //parser.get<String>("c_names");
	cv::String resultFile = parser.get<cv::String>("result");

	//! [Create the importer of TensorFlow model]
	cv::Ptr<cv::dnn::Importer> importer;
	try                                     //Try to import TensorFlow AlexNet model
	{
		importer = cv::dnn::createTensorflowImporter(modelFile);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}
	//! [Create the importer of Caffe model]

	if (!importer)
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	//! [Initialize network]
	cv::dnn::Net net;
	importer->populateNet(net);
	importer.release();                     //We don't need importer anymore
											//! [Initialize network]

											//! [Prepare blob]
	cv::Mat img;
	cv::VideoCapture cap(0);
	for (int i = 0; i < 123; i++)
	{

		cap >> img;

		//cv::Mat img = imread(imageFile);
		if (img.empty())
		{
			std::cerr << "Can't read image from the file: " << imageFile << std::endl;
			exit(-1);
		}

		cv::Size inputImgSize = cv::Size(224, 224);

		if (inputImgSize != img.size())
			resize(img, img, inputImgSize);       //Resize image to input size

		cv::Mat inputBlob = blobFromImage(img);   //Convert cv::Mat to image batch
											  //! [Prepare blob]
		inputBlob -= 117.0;
		//! [Set input blob]
		imshow("ffffffffffff", img);
		cvWaitKey(100);


		net.setInput(inputBlob, inBlobName);        //set the network input
													//! [Set input blob]

		cv::TickMeter tm;
		tm.start();

		//! [Make forward pass]
		cv::Mat result = net.forward(outBlobName);                          //compute output
																		//! [Make forward pass]

		tm.stop();

		if (!resultFile.empty()) {
			//cv::CV_Assert(result.isContinuous());

			ofstream fout(resultFile.c_str(), ios::out | ios::binary);
			fout.write((char*)result.data, result.total() * sizeof(float));
			fout.close();
		}

		std::cout << "Output blob shape " << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << std::endl;
		std::cout << "Inference time, ms: " << tm.getTimeMilli() << std::endl;

		if (!classNamesFile.empty()) {
			std::vector<cv::String> classNames = readClassNames(classNamesFile.c_str());

			int classId;
			double classProb;
			getMaxClass(result, &classId, &classProb);//find the best class

													  //! [Print results]
			std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
			std::cout << "Probability: " << classProb * 100 << "%" << std::endl;

			if (classId == 948)
			{
				//pizza
			}
			if (classId == 543 || classId == 552)
			{
				//keyboard
			}
			if (classId == 524)
			{
				//watch
			}
//			get_Website("http://192.168.2.9:8080/object?object?id=1");


		}

	}
	int farshid;
	cin >> farshid;

	return 0;
} //main


  /* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb)
{
	cv::Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	cv::Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

std::vector<cv::String> readClassNames(const char *filename)
{
	std::vector<cv::String> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}
