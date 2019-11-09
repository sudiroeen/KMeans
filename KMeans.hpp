#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class KMeans{
public:
	KMeans(int jmlKluster, vector<Mat> dataset_, string centroid_yaml_, string lutYAML_, int row_, int col_);
	bool train();

private:
	void saveParam();
	void initializeParam();

private:
	int nKluster;
	vector<Mat> nData;
	vector<Mat> centroid;
	string lutYAML;
	string centroid_yaml;
	Mat canvasYAML;
	vector<Mat> dataset;
	Mat _data_training;

private:
	Mat distance;
	Mat clusterNow;
	Mat clusterBefore;
	float* ndata_per_cluster;

public:
	enum STATE{
	   	STATE_TRAIN = 0,
	   	STATE_SEGMENT = 1,
	   	STATE_OTHER = 2
   	};
};