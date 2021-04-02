// Example "Training" 2D-case with model training

#include "Bayes.h"
#include "timer.h"

int main(int argc, char *argv[])
{
#ifdef WIN32
	const std::string dataPath = "../data/";
#else
	const std::string dataPath = "../../../data/";
#endif
	
	const Size	imgSize		= Size(400, 400);
	const int	width		= imgSize.width;
	const int	height		= imgSize.height;
	const byte	nStates		= 6;				// {road, traffic island, grass, agriculture, tree, car} 	
	const word	nFeatures	= 3;		

	// Reading the images
	Mat train_fv	= imread(dataPath + "001_fv.jpg", 1);	resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
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
			// --- PUT YOUR CODE HERE ---
            featureVector.row(0) = train_fv.at<Vec3b>(y, x)[0];
            featureVector.row(1) = train_fv.at<Vec3b>(y, x)[1];
            featureVector.row(2) = train_fv.at<Vec3b>(y, x)[2];


			if (x == 0 && y == 0)
				std::cout << featureVector << std::endl;

			byte gt = train_gt.at<byte>(y, x);
            
			classifier->addFeatureVec(featureVector, gt);
		} // x
	} // y
	Timer::stop();

	classifier->printPriorProbabilities();
    //    17.2%    0.4%    59.5%    9.9%    13.0%    0.0%
    //    Car class is not represented in training image because 0.0% probability

	// ========================= Testing =========================
	Timer::start("Testing... ");
	for (int y = 0; y < imgSize.height; y++) {
		for (int x = 0; x < imgSize.width; x++) {
			// --- PUT YOUR CODE HERE ---
            featureVector.row(0) = test_fv.at<Vec3b>(y, x)[0];
            featureVector.row(1) = test_fv.at<Vec3b>(y, x)[1];
            featureVector.row(2) = test_fv.at<Vec3b>(y, x)[2];


			// get potentials
			Mat potentials = classifier->getPotentials(featureVector);
			
			// find the largest potential
			byte classLabel = 0;
			// --- PUT YOUR CODE HERE ---
            double minVal;
            double maxVal;
            Point minLoc;
            Point maxLoc;
            minMaxLoc( potentials, &minVal, &maxVal, &minLoc, &maxLoc );
            float maxV = 0.0;
            byte maxL = 0;
            for (byte i = 0; i < potentials.rows; i++) {
//                std::cout << potentials.at<float>(i) << " ";
                if (potentials.at<float>(i) > maxV) {
                    maxV = potentials.at<float>(i);
                    maxL = i;
//                    std::cout << "MaxV:" << maxV <<  "maxL:" << " " << maxL << "i: " << i;
                }
            }
            
//            std::cout << "rows:" << maxLoc.x <<  "cols:" << " ", maxLoc.y;
            classLabel = maxL;
//            std::cout << classLabel << " ";
			solution.at<byte>(y, x) = classLabel;
		}
//        cout << "\n";
	}
	Timer::stop();

	// ====================== Evaluation =======================	
	Timer::start("Evaluation... ");
	double accuracy = 0;
    for (int y = 0; y < imgSize.height; y++) {
        for (int x = 0; x < imgSize.width; x++){
            if (solution.at<byte>(y, x) == test_gt.at<byte>(y, x)) {
//                printf("%hu ", test_gt.at<byte>(y, x));
//                printf("%hu ", solution.at<byte>(y, x));
                accuracy++;
            }
        }
//        std::cout << "\n";
    }
	accuracy /= (imgSize.height * imgSize.width);
	Timer::stop();
	printf("Accuracy = %.2f%%\n", accuracy * 100);


	// ====================== Visualization =======================
	solution *= 32;
	test_gt *= 32;
	imshow("Test image", test_img);
	imshow("groundtruth", test_gt);
	imshow("solution", solution);
	waitKey();

	return 0;
}

