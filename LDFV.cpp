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

#include "LDFV.h"

LDFV::LDFV(string config)
{
	vector<Mat> imgs;
	loadconfigxml(config);
	int imgcount;
	string fullpath;
	char filenum[10];

	//initialize GMM
	VlGMM** gmm = new VlGMM*[opts.numofblocksX*opts.numofblocksY];
	for(int i=0; i<opts.numofblocksX*opts.numofblocksY;i++)
	{
		gmm[i] = vl_gmm_new (VL_TYPE_FLOAT, opts.dimensions, opts.numofclusters);
	}

	if(opts.LDFVfunction == 1){
		//Find and export LDFV descriptors from the test image database

		Mat Sigmas[opts.numofblocksX*opts.numofblocksY];
		Mat Means[opts.numofblocksX*opts.numofblocksY];
		Mat Weights[opts.numofblocksX*opts.numofblocksY];


		/* Initialize an array of [number of test images] pointers to an
		 * array of [number of blocks] pointers to an
		 * array of floats (the fisher vectors).*/
		float*** fisherVects = new float**[opts.numoftestimgs];
		for(int z=0; z<opts.numoftestimgs;z++){
			fisherVects[z] = new float*[opts.numofblocksX*opts.numofblocksY];
			for (int p=0; p< opts.numofblocksX*opts.numofblocksY;p++)
				/* One array of FVs is vl_malloc(sizeof(float)*2*DIMENSIONS*opts.numofclusters); */
				fisherVects[z][p] = new float[2*opts.numofclusters*opts.dimensions];
		}

		importGMMparams(Sigmas, Means, Weights);

		fullpath = opts.pathToTestImages;

		for(imgcount = 0; imgcount < opts.numoftestimgs; imgcount++)
		{
			//image is 3 characters wide plus the image format (NNN.bmp)
			sprintf(filenum,"%03d.bmp",imgcount);
			imgs.push_back(imread(fullpath+filenum));

			if (!imgs[imgcount].data){
				perror("Image data not loaded properly");
			}
		}

		//Convert from (default) BGR to HSV for each image of the vector imgs
		vector<Mat>::const_iterator iter;
		vector<float> featVector;
		int numofdata = 3*(imgs[0].cols/opts.numofblocksX)*(imgs[0].rows/opts.numofblocksY);
		featVector.reserve(numofdata*opts.dimensions);

		for(iter = imgs.begin(); iter != imgs.end(); iter++){
			cvtColor((*iter), (*iter), CV_BGR2HSV,CV_32FC3);
			getLDFVs((*iter), Sigmas, Means, Weights, fisherVects,iter - imgs.begin(), featVector);
			exportDescriptor(opts.pathToTestImages, fisherVects[iter - imgs.begin()], iter - imgs.begin() );
		}

		//Memory deallocation
		for(int z=0; z<opts.numoftestimgs;z++){
			for (int p=0; p< opts.numofblocksX*opts.numofblocksY;p++)
				delete [] fisherVects[z][p];
			delete [] fisherVects[z];
		}
		delete [] fisherVects;

	}else if(opts.LDFVfunction == 0){

		/* Training...fill in the vector of training images
		 * to compute GMM */

		fullpath = opts.pathToTrainImages;

		for(imgcount = 0; imgcount < opts.numoftrainimgs; imgcount++)
		{
			//image is 3 characters wide plus the image format (NNN.bmp)
			sprintf(filenum,"%03d.bmp",imgcount);
			imgs.push_back(imread(fullpath+filenum));
			cout << filenum << endl;
			if (!imgs[imgcount].data){
				perror("Image data not loaded properly");
			}
		}

		vector<Mat>::const_iterator iter;

		//Convert from (default) BGR to HSV for each image of the vector imgs
		for(iter = imgs.begin(); iter != imgs.end(); iter++){
			cvtColor((*iter), (*iter), CV_BGR2HSV, CV_32FC3);
		}

		//Start training and extract GMM parameters
		training(imgs, gmm);
		for(int i=0; i< opts.numofblocksX*opts.numofblocksY;i++){
			vl_gmm_delete(gmm[i]);
		}

	}else if(opts.LDFVfunction == 2){

		Mat testDescriptors[opts.numoftestimgs+1];
		Mat Sigmas[opts.numofblocksX*opts.numofblocksY];
		Mat Means[opts.numofblocksX*opts.numofblocksY];
		Mat Weights[opts.numofblocksX*opts.numofblocksY];
		Mat queryImg = imread(opts.queryImg);

		vector<float> featVector;
		int numofdata = 3*(queryImg.cols/opts.numofblocksX)*(queryImg.rows/opts.numofblocksY);
		featVector.reserve(numofdata*opts.dimensions);

		/* Initialize an array of [number of test images] pointers to an
		 * array of [number of blocks] pointers to an
		 * array of floats (the fisher vectors).*/
		float*** fisherVects = new float**[1];
		for(int z=0; z<1;z++){
			fisherVects[z] = new float*[opts.numofblocksX*opts.numofblocksY];
			for (int p=0; p< opts.numofblocksX*opts.numofblocksY;p++)
				/* One array of FVs is vl_malloc(sizeof(float)*2*DIMENSIONS*opts.numofclusters); */
				fisherVects[z][p] = new float[2*opts.numofclusters*opts.dimensions];
		}
		//Import GMM parameters
		importGMMparams(Sigmas, Means, Weights);
		//Convert image to HSV
		cvtColor(queryImg, queryImg, CV_BGR2HSV,CV_32FC3);
		//Get feature vectors
		getLDFVs(queryImg, Sigmas, Means, Weights, fisherVects, 0, featVector);
		//Export the final image descriptor as fisher-999.xml
		exportDescriptor(opts.pathToTestImages,fisherVects[0], -1 );
		//import Fisher Vectors of all test images in separate Mat arrays
		importFisherVectors(testDescriptors,0);

		int x = findMinDistancePic(testDescriptors);
		cout << "Matching picture is: " << endl;
		cout << x << endl;

		//Memory deallocation
			for (int p=0; p< opts.numofblocksX*opts.numofblocksY;p++)
				delete [] fisherVects[0][p];
			delete [] fisherVects[0];
		delete [] fisherVects;

	}else if(opts.LDFVfunction == 3){

		Mat testDescriptors[opts.numoftestimgs+1];
		Mat queryDescriptors[opts.numofqueryimgs+1];

		//import Fisher Vectors of all test images in separate Mat arrays
		importFisherVectors(testDescriptors,0);
		//import Fisher Vectors of all Query images in separate Mat arrays
		importFisherVectors(queryDescriptors,1);

		exportDistancesCSV(testDescriptors,queryDescriptors);

	}else{
		//xml configuration is wrong
		perror("'DBdescriptors-Training-ReId' option in config.xml must be either 0,1,2");
	}
};

void LDFV::getLDFVs(const Mat pic,
		const Mat* Sigmas, const Mat* Means,
		const Mat* Weights, float*** fisherVects,
		int imgcount, vector<float>& featVector)
{
	//initialize loop counters
	int x,y,numofdata;

	/* initialize "layers" Mat and
	 * STL vector of floats (used for feature vectors) */
	Mat layers[3*opts.dimensions];

	for(int i=0; i< 3 * (int) opts.dimensions; i++){
		layers[i].create(pic.rows, pic.cols,CV_32FC1);
	}
	numofdata = 3*(pic.cols/opts.numofblocksX)*(pic.rows/opts.numofblocksY);

	/* split colors, compute derivatives,
	 * place them all layers in "layers" array */
	computeLayers(pic, layers);

	//horizontal scan block by block
	for (y=0; y < opts.numofblocksY; y++){
		for(x=0; x< opts.numofblocksX; x++){

			/* retrieve all feature vectors of this block and then
			 * array the std::vector is using internally */
			blockFeatVector(x,y,layers, featVector);

			vl_fisher_encode(fisherVects[imgcount][y*opts.numofblocksX+x],
					VL_TYPE_FLOAT,(float const*) Means[y*opts.numofblocksX+x].data,
					opts.dimensions, opts.numofclusters,
					(float const*) Sigmas[y*opts.numofblocksX+x].data,
					(float const*) Weights[y*opts.numofblocksX+x].data,
					&featVector[0],numofdata,
					VL_FISHER_FLAG_IMPROVED);
			featVector.resize(0);
		}
	}
	for(int i=0;i< 3 * (int) opts.dimensions; i++){
		layers[i].release();
	}
}

//Export final descriptor (concatenation of fisher vectors from each block)
void LDFV::exportDescriptor(string path, float** fisherVectors, int imgcount){
	int x,y,z;
	char filename[15];
	if(imgcount != -1){
		sprintf(filename,"fisher-%03d.xml",imgcount);
	}else{
		sprintf(filename,"fisher-%03d.xml",opts.numoftestimgs);
	}
	string fullpath = path + filename;

	FILE * ofpFV = fopen(fullpath.c_str(), "w");
	fprintf(ofpFV,
	"<?xml version=\"1.0\"?>\n<opencv_storage>\n<final_descriptor type_id=\"opencv-matrix\">\n<rows>1</rows>\n<cols>%d</cols>\n<dt>f</dt>\n<data>"
			,2*(int) opts.numofclusters * (int) opts.dimensions*opts.numofblocksX*opts.numofblocksY);
	for (y=0; y< opts.numofblocksY; y++){
		for(x=0; x< opts.numofblocksX; x++){
			 for(z = 0; z < 2*(int) opts.numofclusters* (int) opts.dimensions; z++) {
				 fprintf(ofpFV, "%f ", (float)fisherVectors[y*opts.numofblocksX+x][z]);
			 }
		}
	}
	fprintf(ofpFV, "</data>\n</final_descriptor>\n</opencv_storage>\n");
	fclose(ofpFV);
}

void LDFV::training(const vector<Mat> trainPics, VlGMM** gmm)
{
	//initialize loop counters
	int x,y;

	vl_size numofdata;

	/* initialize layers Mat and
	 * STL vector of floats (used for feature vectors) */
	Mat layers[3*opts.dimensions];
	for(int i=0;i< 3 * (int) opts.dimensions; i++){
		layers[i].create(trainPics[0].rows, trainPics[0].cols,CV_32FC1);
	}

	vector<float> featVector;

	//iterator of training images vector
	vector<Mat>::const_iterator iter;

	//horizontal scan block by block
	for (y=0; y< opts.numofblocksY; y++){
		for(x=0; x< opts.numofblocksX; x++){
			//of every picture (applies only to the training process
			for(iter = trainPics.begin(); iter != trainPics.end(); iter++){

				/* split colors, compute derivatives,
				 * place them all as layers in "layers" array*/
				computeLayers((*iter), layers);

				/* retrieve all feature vectors of this block and then use the
				 * array the std::vector is using internally */
				blockFeatVector(x,y,layers, featVector);
			}

			//Get number of multidimensional data
			numofdata= featVector.size()/opts.dimensions;

			/* convert the std::vector to an array by pointing to the actual
			 * array the std::vector is using internally */
			computeGMM(x,y,&featVector[0], gmm, numofdata);
			featVector.clear();
		}
	}
	exportGMMparams(gmm);
}

void LDFV::blockFeatVector(int x, int y,const Mat* layers, vector<float>& featVect )
{
	int blockx, blocky, layercount;

	/* get block width & height using one of the layers
	 * not the original picture (Mat includes channels) */
	int blockwidth = layers[0].cols / opts.numofblocksX;
	int blockheight = layers[0].rows / opts.numofblocksY;

	//horizontal scan of blocks (left to right)
	for(blocky=y*blockheight; blocky < y*blockheight + blockheight; blocky++){
		for (blockx = x*blockwidth; blockx < x*blockwidth + blockwidth ; blockx++){
			//for each layer
			for(layercount = 0; layercount < (int) opts.dimensions*3; layercount++){
				//get all values of all layers at blockx,blocky coordinates
				featVect.push_back(layers[layercount].at<float>(blockx,blocky));
			}
		}
	}
}

/* Create a Mat array for each channel's intensity(H/S/V) and required
 * properties H/Ix(H)/Iy(H)/Ixx(H)/Iyy(H).
 * 3 channels x 5 properties = 15 layers*/
void LDFV::computeLayers(const Mat img, Mat* layers)
{
	//initialize loop counter
	int x, y, chans, l;

	//split and place HSV layers inside the vector
	split(img,layers);

	/* Rearrange HSV in layers array to create
	   separate feature vectors for each color */
	layers[2].copyTo(layers[16]);
	layers[1].copyTo(layers[9]);
	layers[0].copyTo(layers[2]);

	//insert x,y layers and calculate derivatives for each channel
	for (chans=0; chans<3; chans++){
		for(x=0; x<img.cols;x++) layers[chans*opts.dimensions].col(x) = Scalar::all(x);
		for(y=0; y<img.rows;y++) layers[chans*opts.dimensions+1].row(y) = Scalar::all(y);

			Sobel(layers[chans*opts.dimensions+2], layers[chans*opts.dimensions+3], CV_32FC1,1,0);
			Sobel(layers[chans*opts.dimensions+2], layers[chans*opts.dimensions+4], CV_32FC1,0,1);
			Sobel(layers[chans*opts.dimensions+2], layers[chans*opts.dimensions+5], CV_32FC1,2,0);
			Sobel(layers[chans*opts.dimensions+2], layers[chans*opts.dimensions+6], CV_32FC1,0,2);
	}

	/* Normalize values from range [0,255] to
	 * [0,1] (floating point values for VLFeat) */
	for( l = 0; l < (int)opts.dimensions*3; l++)
		normalize(layers[l],layers[l],0,1,NORM_MINMAX, CV_32FC1);

}

void LDFV::computeGMM(int x, int y, float* featVectorArray, VlGMM** gmm, vl_size numofdata)
{
	 //set the GMM parameters
     vl_gmm_set_initialization(gmm[y*opts.numofblocksX+x],VlGMMRand);
	 vl_gmm_set_max_num_iterations (gmm[y*opts.numofblocksX+x], opts.maxiter) ;
	 vl_gmm_set_num_repetitions(gmm[y*opts.numofblocksX+x], opts.maxrep);
	 vl_gmm_set_verbosity(gmm[y*opts.numofblocksX+x],1);
	 vl_gmm_set_covariance_lower_bound (gmm[y*opts.numofblocksX+x],opts.sigmaLowerBound);

	 //compute GMM
	 vl_gmm_cluster(gmm[y*opts.numofblocksX+x], featVectorArray, numofdata);
}

void LDFV::loadconfigxml(string configfile)
{
	//Load all configuration from
	FileStorage fs;
	fs.open(configfile, FileStorage::READ);
	opts.numofblocksX = (int) fs["NUMOFBLOCKSX"];
	opts.numofblocksY = (int) fs["NUMOFBLOCKSY"];
	opts.LDFVfunction = (int) fs["Training-DBdescriptors-ReId-Folders"];
	opts.numoftrainimgs = (int) fs["NumOfTrainingImages"];
	opts.numoftestimgs = (int) fs["NumOfTestImages"];
	opts.numofqueryimgs = (int) fs["NumOfQueryImages"];
	opts.numofclusters = (int) fs["NumOfClusters"];
	opts.dimensions = (int) fs["dimensions"];
	opts.pathToTrainImages = (string) fs["pathToTrainImages"];
	opts.pathToTestImages = (string) fs["pathToTestImages"];
	opts.pathToQueryImages = (string) fs["pathToQueryImages"];
	opts.queryImg = (string) fs["QueryImage"];
	opts.maxiter = (int) fs["maxIterations"];
	opts.maxrep = (int) fs["maxRepetitions"];
	opts.sigmaLowerBound = (double) fs["sigmaLowerBound"];
}

void LDFV::exportGMMparams(VlGMM** gmm)
{
	int x,y,Idx;

	FILE * ofp = fopen("gmm_parameters.xml", "w");
	fprintf(ofp, "<?xml version=\"1.0\"?>\n<opencv_storage>\n");
	for (y=0; y<=opts.numofblocksY-1; y++){
		for(x=0; x<=opts.numofblocksX-1; x++){
			//export GMM sigmas/covariances
			float const * sigmas = (float *) vl_gmm_get_covariances(gmm[y*opts.numofblocksX+x]) ;
		    fprintf(ofp,
		     "<sigmas-block%02d type_id=\"opencv-matrix\">\n<rows>1</rows>\n<cols>%d</cols>\n<dt>f</dt>\n<data>",
		     y*opts.numofblocksX+x, (int) opts.numofclusters* (int) opts.dimensions);
			for(Idx = 0; Idx < (int) opts.numofclusters* (int) opts.dimensions; Idx++) {
					fprintf(ofp, "%f ", ((float*)sigmas)[Idx]);
			}
			fprintf(ofp, "</data>\n</sigmas-block%02d>\n",y*opts.numofblocksX+x);

			//export GMM means
			float const * means = (float *) vl_gmm_get_means(gmm[y*opts.numofblocksX+x]) ;
			fprintf(ofp,
			"<means-block%02d type_id=\"opencv-matrix\">\n<rows>1</rows>\n<cols>%d</cols>\n<dt>f</dt>\n<data>",
			y*opts.numofblocksX+x, (int) opts.numofclusters* (int) opts.dimensions);
			for(Idx = 0; Idx < (int) opts.numofclusters* (int) opts.dimensions; Idx++) {
					fprintf(ofp, "%f ", ((float*)means)[Idx]);
			}
			fprintf(ofp, "</data>\n</means-block%02d>\n",y*opts.numofblocksX+x);

			//export GMM weights/priors of each cluster
			float const * weights = (float const*) vl_gmm_get_priors(gmm[y*opts.numofblocksX+x]) ;
			fprintf(ofp,
			"<weights-block%02d type_id=\"opencv-matrix\">\n<rows>1</rows>\n<cols>%d</cols>\n<dt>f</dt>\n<data>",
			y*opts.numofblocksX+x,(int) opts.numofclusters);
			for(Idx = 0; Idx < (int) opts.numofclusters; Idx++) {
					fprintf(ofp, "%f ", ((float*)weights)[Idx]);
			}
			fprintf(ofp, "</data>\n</weights-block%02d>\n",y*opts.numofblocksX+x);
		}
	}
	fprintf(ofp, "</opencv_storage>\n");
	fclose(ofp);
}

void LDFV::importGMMparams(Mat* Sigmas,Mat* Means,Mat* Weights)
{
	int x,y;
	char blocknum[10];
	string blocksigma = "sigmas-";
	string blockmeans = "means-";
	string blockweights = "weights-";

	string gmmfile = "gmm_parameters.xml";
	FileStorage fs;
	fs.open(gmmfile, FileStorage::READ);

	for (y=0; y<opts.numofblocksY; y++){
		for(x=0; x<opts.numofblocksX; x++){
			sprintf(blocknum,"block%02d",y*opts.numofblocksX+x);
			//import from xml to OpenCV	matrices
			fs[blocksigma + blocknum] >> Sigmas[y*opts.numofblocksX+x];
			fs[blockmeans + blocknum] >> Means[y*opts.numofblocksX+x];
			fs[blockweights + blocknum] >> Weights[y*opts.numofblocksX+x];
		}
	}
}

void LDFV::importFisherVectors(Mat* FVs, bool testORquery)
{
	char filename[15];
	string pathToImages;
	string fullpath;
	int numofimgs;

	if(testORquery){
		numofimgs = opts.numofqueryimgs;
		pathToImages = opts.pathToQueryImages;
	}else{
		numofimgs = opts.numoftestimgs;
		pathToImages = opts.pathToTestImages;
	}

	for(int im=0; im<numofimgs+1; im++)
	{
		sprintf(filename,"%03d",im);
		fullpath = pathToImages + "fisher-" + filename + ".xml";

		FileStorage fs(fullpath, FileStorage::READ);
		fs["final_descriptor"] >> FVs[im];
		fs.release();
	}
}

int LDFV::findMinDistancePic(Mat* FVs)
{
	//A point to locate the minimum distance value
	Point p;
	Mat distances(1,opts.numoftestimgs,DataType<double>::type);

		for(int im=0; im<opts.numoftestimgs; im++){
			distances.at<double>(0,im) = norm(FVs[im],FVs[opts.numoftestimgs],NORM_L2);
		}

	cout << distances << endl;
    FILE *ofp = fopen("test.csv","w+");

    for(int i = 0 ; i < distances.cols;i++)
        {
          fprintf(ofp,"%F\t     %d\n",distances.at<double>(0,i),i);
        }
        fclose(ofp);
	minMaxLoc(distances, 0, 0, &p);

	return p.x;
}

void LDFV::exportDistancesCSV(Mat* FVs, Mat* QFVs){
	char buf[0x100];

	Mat distances(1,opts.numoftestimgs,DataType<double>::type);

	for (int qim=0; qim<opts.numofqueryimgs; qim++){
		for(int im=0; im<opts.numoftestimgs; im++){
			distances.at<double>(0,im) = norm(FVs[im],QFVs[qim],NORM_L2);
		}
		snprintf(buf, sizeof(buf), "%03d.csv", qim);
		FILE *ofp = fopen(buf,"w+");

		for(int i = 0 ; i < distances.cols;i++)
		{
			fprintf(ofp,"%F\t     %d\n",distances.at<double>(0,i),i);

		}
		fclose(ofp);
	}
}
