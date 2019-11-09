#include "KMeans.hpp"

Mat gambar;
Mat gambarClone;
Point titikStart;
bool afterDownBeforeUp = false;
Rect rectROI;

static void onMouse(int event, int x, int y, int, void*);
vector<Mat> masukanMatrix(Mat gambar, Rect kotak);


int main(){
	vector<Mat> _datasetWarna;

	std::vector<string> namaKluster;

	int nklaster = 3;
	for(int s=0; s<nklaster; s++){
		stringstream ss;
		ss << "class_" << s;
		namaKluster.push_back(ss.str());
	}

	int state = KMeans::STATE_TRAIN;

	Mat foto = imread("source.png");

	while(true){
		Mat frame = foto.clone();

		resize(frame, frame, Size(), 640.0/(double)frame.cols, 480.0/(double)frame.rows);

		namedWindow("frame", CV_WINDOW_NORMAL);
		imshow("frame", frame);
		int key = waitKey(10);

		if((char) key == 'k'){
			namedWindow("kalibrasiFrame", CV_WINDOW_NORMAL);
			setMouseCallback("kalibrasiFrame", onMouse);
			
			gambar = frame;
			gambarClone = gambar.clone();

			while(true){
				int inkey = waitKey(10);
				imshow("kalibrasiFrame", gambar);

				if((rectROI.width != 0) || (rectROI.height != 0) ){
	                bool kalib = true;
	                vector<Mat> dataTemp;

	                if((char)inkey == 's'){
                    	dataTemp = masukanMatrix(gambar, rectROI);
                    	cout << "data saved !!!" << endl;
	                }

		            if(dataTemp.size()){
	            		_datasetWarna.insert(_datasetWarna.end(), dataTemp.begin(), dataTemp.end());
		            }
            	}

				if((char) inkey == 'c'){
					destroyAllWindows();
					break;
				}
			}
		}


		if(state == KMeans::STATE_TRAIN){
			if(! _datasetWarna.size()){
				cout << "Harus ada dataset BROO" << endl;
			}else if(_datasetWarna.size() > 3000){
				int banyakKluster = namaKluster.size();

				FileStorage fsdata("dataset.yaml", FileStorage::WRITE);
				fsdata << "datasetSize" << (int)_datasetWarna.size();

				for(int d=0; d<_datasetWarna.size(); d++){
					stringstream ss; ss << d;
					fsdata << "Mat_"+ss.str() <<  _datasetWarna[d];
				}
				fsdata.release();

				string saveToYAML_ = "RGB.yaml";
				KMeans KM(banyakKluster, _datasetWarna, saveToYAML_, "lutYAML.yaml", 1, (1 << 18));
				bool isSuccess = KM.train();
				if(isSuccess)
					return 0;
			}
		}
	}
}


static void onMouse(int event, int x, int y, int, void*){
    int xrs, yrs, lx, ly;

    if(afterDownBeforeUp){
        gambar = gambarClone.clone();
        xrs = min(titikStart.x, x);
        yrs = min(titikStart.y, y);
        lx = max(titikStart.x, x) - min(titikStart.x, x);
        ly = max(titikStart.y, y) - min(titikStart.y, y);
        rectROI = Rect(xrs, yrs, lx+1, ly+1);

        rectangle(gambar, rectROI,Scalar(255, 0, 0), 1);
    }
    if(event == EVENT_LBUTTONDOWN){
        titikStart = Point(x,y);
        rectROI = Rect(x,y,0,0);
        afterDownBeforeUp = true;

    }else if(event == EVENT_LBUTTONUP){
        Mat roi(gambarClone.clone(), rectROI);
        imshow("roi", roi);

        afterDownBeforeUp = false;
    }
}


vector<Mat> masukanMatrix(Mat gambar, Rect kotak){
    int xrs, yrs, xrf, yrf;
    xrs = kotak.x;
    yrs = kotak.y;
    xrf = xrs + kotak.width;
    yrf = yrs + kotak.height;

    vector<Mat> RGB;
    for(int xx=xrs+1; xx<xrf; xx++){
        for(int yy=yrs+1; yy<yrf; yy++){
            Vec3b pixel = gambar.at<Vec3b>(yy,xx);

            float R = (float)pixel[2];
            float G = (float)pixel[1];
            float B = (float)pixel[0];

            RGB.push_back((Mat_<float>(3,1) << B, G, R));
        }
    }

    return RGB;
}