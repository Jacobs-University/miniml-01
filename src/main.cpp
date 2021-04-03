// Example "Training" 2D-case with model training

#include "Bayes.h"
#include "timer.h"

int main(int argc, char *argv[])
{
#ifdef WIN32
	const std::string dataPath = "../data/";
#elif defined(__APPLE__) 
	const std::string dataPath = "../data/"; 
	// not really OS-specific, but we used the build directory right inside the repo root + console cmake,
	// then ran inside the build/ dir, not the build/bin/ dir
#else
	const std::string dataPath = "../../../data/";
#endif
	
	const Size	imgSize		= Size(400, 400);
	const int	width		= imgSize.width;
	const int	height		= imgSize.height;
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;		

	// Reading the images
	Mat train_fv	= imread(dataPath + "001_fv.jpg", 1);	
	resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector

	Mat train_gt	= imread(dataPath + "001_gt.bmp", 0);	
	resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training

	Mat test_fv		= imread(dataPath + "002_fv.jpg", 1);	
	resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector

	Mat test_gt		= imread(dataPath + "002_gt.bmp", 0);	
	resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);	// groundtruth for evaluation

	Mat test_img	= imread(dataPath + "002_img.jpg", 1);	
	resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	Mat	featureVector(nFeatures, 1, CV_8UC1);
	Mat	solution(imgSize, CV_8UC1);

	auto classifier = std::make_shared<CBayes>(nStates, nFeatures);

	// ========================= Training =========================
	Timer::start("Training... ");
	for (int y = 0; y < imgSize.height; y++) {
		for (int x = 0; x < imgSize.width; x++) {
			// --- PUT YOUR CODE HERE ---
			auto featureVector = Mat(train_fv.at<Vec3b>(y, x));

			if (x == 0 && y == 0)
				std::cout << featureVector << std::endl;

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
			// --- PUT YOUR CODE HERE ---
			auto featureVector = Mat(test_fv.at<Vec3b>(y, x));

			// get potentials
			Mat potentials = classifier->getPotentials(featureVector);
			
			// find the largest potential
			byte classLabel = 0;
			// --- PUT YOUR CODE HERE ---
			auto pot = std::vector<double>(potentials);
			classLabel = std::distance(pot.begin(), std::max_element(pot.begin(), pot.end()));


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


	// ====================== Visualization =======================
	solution *= 32;
	test_gt *= 32;
	imshow("Test image", test_img);
	imshow("groundtruth", test_gt);
	imshow("solution", solution);
	imwrite(dataPath+"../renders/solution.jpg", solution);
	waitKey();

	return 0;
}

