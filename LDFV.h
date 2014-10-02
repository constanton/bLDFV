/*
 * FVReid.h
 *
 *  Created on: Jul 13, 2014
 *      Author: Kostas Antonakoglou
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
