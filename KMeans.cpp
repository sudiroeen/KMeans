#include "KMeans.hpp"

KMeans::KMeans(int jmlKluster, vector<Mat> dataset_, string centroid_yaml_, string lutYAML_, int row_, int col_)
	:nKluster(jmlKluster), nData(dataset.size()), lutYAML(lutYAML_), centroid_yaml(centroid_yaml_), 
	_data_training(Mat::zeros(dataset_.size(), dataset_[0].rows, CV_32FC1)), dataset(dataset_),
	 ndata_per_cluster(new float[nKluster])
{	
	initializeParam();
}


void KMeans::initializeParam(){
	for(int n=0; n<nKluster; n++)
		ndata_per_cluster[n] = 0.0;
	for(int s=0; s<dataset.size(); s++)
		_data_training.row(s) = dataset[s].t();

	centroid.resize(nKluster);
	for(int k=0; k<nKluster; k++){
		Mat dt_ = dataset[k].t();
		centroid[k]= 0.0 * Mat::ones(dataset.size(), 1, CV_32FC1) * dt_;
	}
	distance = Mat(dataset.size(), nKluster, CV_32FC1);
	clusterBefore = Mat(dataset.size(), 1, CV_8UC1);
	clusterNow = Mat(dataset.size(), 1, CV_8UC1);
}

bool KMeans::train(){
	int step = 0;
	while(true){
		cout << "step: " << step << endl;
		for(int c=0; c<nKluster; c++){
			Mat part2;
			pow(_data_training - centroid[c], 2.0, part2);
			for(int r=0; r<dataset.size(); r++){
				distance.at<float>(r,c) = sum(part2.row(r))[0];
			}
		}

		sqrt(distance, distance);

		for(int l=0; l<nKluster; l++){
			cout << "centroid[" << l << "]: " << centroid[l].row(0) << endl;
			centroid[l] = 0.0*centroid[l].clone();
		}

		for(int r=0; r<dataset.size(); r++){
			double minval, maxval;
			Point minpos, maxpos;

			minMaxLoc(distance.row(r), &minval, &maxval, &minpos, &maxpos);
			clusterNow.at<uchar>(r) = (uchar)minpos.x;
			centroid[minpos.x] +=  Mat::ones(dataset.size(), 1, CV_32FC1) * _data_training.row(r);
			ndata_per_cluster[minpos.x] += 1;
		}

		for(int k=0; k<nKluster; k++)
			centroid[k] /= ndata_per_cluster[k];

		cout << "norm: " << (float)cv::norm(clusterNow, clusterBefore, NORM_L1) << endl;

		if(!cv::norm(clusterNow, clusterBefore, NORM_L1)){
			saveParam();
			cout << "Training done !!!" << endl;
			return true;
		}

		clusterBefore = clusterNow.clone();
		step ++;
	}
}

void KMeans::saveParam(){
	FileStorage fs(centroid_yaml, FileStorage::WRITE);

	for(int w=0; w<centroid.size(); w++){
		stringstream ss;
		ss << w;

		fs << "centroid"+ss.str() << centroid[w].row(0);
	}

	fs.release();
}