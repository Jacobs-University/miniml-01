// Example "Training" 2D-case with model training

#include "Bayes.h"
#include "timer.h"
#include "opencv2\opencv.hpp";
#include <opencv2\highgui\highgui.hpp>;
#include <iostream>;

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
#ifdef WIN32
	const std::string dataPath = "C:\\Bonaventure_Files_Old_Computer\\ML-Assignment\\data\\"; // ----> this original path linking didn't work with my IDE, I had to put the full link "../data/";
#else
	const std::string dataPath = "../../../data/";
#endif

	const Size	imgSize		= Size(400, 400);
	const int	width		= imgSize.width;
	const int	height		= imgSize.height;
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;		

	// Reading the images

	Mat train_fv = imread(dataPath + "001_fv.jpg", 1); resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt	= imread(dataPath + "001_gt.bmp", 0);	resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training
	Mat test_fv		= imread(dataPath + "002_fv.jpg", 1);	resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	Mat test_gt		= imread(dataPath + "002_gt.bmp", 0);	resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);	// groundtruth for evaluation
	Mat test_img	= imread(dataPath + "002_img.jpg", 1);	resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image
	Mat	featureVector(nFeatures, 1, CV_8UC1);
	Mat	solution(imgSize, CV_8UC1);

	auto classifier = std::make_shared<CBayes>(nStates, nFeatures);

	// ========================= Training =========================
	Timer::start("Training... ");
	for (int y = 0; y < imgSize.height; y++) {
		for (int x = 0; x < imgSize.width; x++) {
			// we get the bytes coordinates and pass (fill) it to the features vector
			// first output == [132, 12, 73]
			// we can bypass the affectation to the featureVector since train_fv.at<Vec3b>(y, x)
			// aleady gives an array of shape (n_features, 1) == featureVector
			
			featureVector = (Mat)train_fv.at<Vec3b>(y, x);

			byte gt = train_gt.at<byte>(y, x);
			classifier->addFeatureVec(featureVector, gt);
		} // x
	} // y
	Timer::stop();

	classifier->printPriorProbabilities();

	// ========================= Testing =========================
	Timer::start("Testing... ");
	for (int y = 0; y < imgSize.height; y++) {
		for (int x = 0; x < imgSize.width; x++) {

			featureVector = (Mat)test_fv.at<Vec3b>(y, x);
			// get potentials
			Mat potentials = classifier->getPotentials(featureVector); // return list of probabilities for each class
			// find the largest potential
			
			byte classLabel = 0;
			// we initialize the maximum element at the first element
			float init_max = potentials.at<float>(0);

			float init_max_index = 0;
			for (int i = 1; i < potentials.rows; i++) {
				if (potentials.at<float>(i) > init_max) {
					init_max = potentials.at<float>(i);
					init_max_index = i;
				}
			}
			classLabel = (byte)init_max_index;
			solution.at<byte>(y, x) = classLabel;
		}
	}
	Timer::stop();

	// ====================== Evaluation =======================	
	Timer::start("Evaluation... ");
	double accuracy = 0;
	for (int y = 0; y < imgSize.height; y++)
		for (int x = 0; x < imgSize.width; x++)
			if (solution.at<byte>(y, x) == test_gt.at<byte>(y, x))
				accuracy++;
	accuracy /= (imgSize.height * imgSize.width);
	Timer::stop();
	printf("Accuracy = %.2f%%\n", accuracy * 100);

	// This prints 79,10 % as stated in the repository
	// the image are stored as well in the renders folder

	// ====================== Visualization =======================
	solution *= 32;
	test_gt *= 32;
	imshow("Bonaventure_Dossou_Test_Image", test_img);
	imshow("Bonaventure_Dossou_Ground_Truth", test_gt);
	imshow("Bonaventure_Dossou_Solution", solution);

	// Again, my system didn't really read the short link - I had to use the full path
	const std::string renderdataPath = "C:\\Bonaventure_Files_Old_Computer\\ML-Assignment\\renders\\";
	imwrite(renderdataPath + "Bonaventure_Dossou_Test_Image.png", test_img);
	imwrite(renderdataPath + "Bonaventure_Dossou_GroundTruth.png", test_gt);
	imwrite(renderdataPath + "Bonaventure_Dossou_Solution.png", solution);

	waitKey();

	return 0;
}

