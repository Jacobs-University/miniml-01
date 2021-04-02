#include "Bayes.h"
#include <numeric>

CBayes::CBayes(byte nStates, word nFeatures) 
	: m_nStates(nStates)
	, m_nFeatures(nFeatures)
{
	m_pPrior = std::make_shared<CPDFHistogram>(nStates);

	for (int i = 0; i < m_nStates * m_nFeatures; i++)
		m_vPDF.push_back(std::make_shared<CPDFHistogram>(256));
}

void CBayes::addFeatureVec(const Mat & featureVector, byte gt)
{
	// --- PUT YOUR CODE HERE ---
    
	for (word f = 0; f < m_nFeatures; f++) {
		byte feature = featureVector.at<byte>(f, 0);
		m_vPDF[f * m_nStates + gt]->addPoint(feature);
	}
    
    m_pPrior->addPoint(gt);
}

Mat CBayes::getPotentials(const Mat & featureVector) const
{
	Mat res(m_nStates, 1, CV_32FC1);
	
	for (byte s = 0; s < m_nStates; s++)
		res.at<float>(s, 0) = static_cast<float>(m_pPrior->getDensity(s));
	
	// --- PUT YOUR CODE HERE ---


	
	return res;
}

void CBayes::printPriorProbabilities(void) {
	for (byte s = 0; s < m_nStates; s++)
		printf("%.1f%%\t", 100 * m_pPrior->getDensity(s));
	printf("\n");
}
