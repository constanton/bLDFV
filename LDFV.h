/*
Copyright 2014 Konstantinos Antonakoglou

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vl/gmm.h>
#include <vl/fisher.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

struct options
{
	 int numofblocksX;
	 int numofblocksY;
	 vl_size maxiter;
	 vl_size maxrep;
	 double sigmaLowerBound;
	 /* LDFVfunction is either 0,1,2 for:
	  * Find database descriptors, Training to find GMM or Image Re-id */
	 int LDFVfunction;
	 vl_size numofclusters;
	 int numoftrainimgs;
	 int numoftestimgs;
	 vl_size dimensions;
	 string pathToTrainImages;
	 string pathToTestImages;
	 string queryImg;
};

class LDFV
{
private:
	options opts;
	void exportDescriptor(string path, float** fisherVectors, int imgcount);
	void getLDFVs(const Mat pic,const Mat* Sigmas,const Mat* Means,const Mat* Weights,
			float*** fisherVects, int imgcount,vector<float>& featVector);
	void training(const vector<Mat> trainPics, VlGMM** gmm);
	void blockFeatVector(int x, int y,const Mat* layers, vector<float>& featVect  );
	void computeLayers(const Mat img, Mat* layers);
	void computeGMM(int x, int y, float featVectors[], VlGMM** gmm,	vl_size numofdata);
	void loadconfigxml(string config);
	void exportGMMparams( VlGMM** gmm);
	void importGMMparams(Mat* Sigmas,Mat* Means,Mat* Weights);
	int findMinDistancePic(Mat* FVs);
	void importFisherVectors(Mat* FVs);
public:
	LDFV(string config);
};
