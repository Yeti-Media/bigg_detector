/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */


/**

\author Gary Bradski and Marius Muja

**/

#ifndef BINARY_GRADIENT_GRID_H_
#define BINARY_GRADIENT_GRID_H_

#include "rein/core/detector.h"
#include "rein/core/trainable.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <string>
#include <map>

#include <flann/flann.hpp>

namespace bigg_detector {

using namespace rein;

///////////////////////////////////////////////////////
// Pyramid
//////////////////////////////////////////////////
class BinarizedGradientPyramid
{
public:
	BinarizedGradientPyramid(const cv::Mat& image, int start_level_, int levels);

	cv::Mat& operator[](size_t idx) { return pyramid[idx-start_level]; }
	const cv::Mat& operator[](size_t idx) const { return pyramid[idx-start_level]; }

	int start_level;
	int levels;
private:
	std::vector<cv::Mat> pyramid;
};

class BinarizedGradientTemplate;

struct BiGGDetection
{
//	std::string id;
	std::string name;
	float score;
	cv::Rect roi;
	cv::Mat mask;
	int index;
	int x;
	int y;
	const BinarizedGradientTemplate* tpl;
	boost::shared_ptr<BinarizedGradientTemplate> crt_tpl;
};

//////////////////////////////////////////////////////
// template
//////////////////////////////////////////////////////
class BinarizedGradientTemplate
{
public:
	/**
	 * IDdentiy of the object this template belongs to
	 */
//	int id;
	/**
	 * In the reduced image. The side of the template square is then 2*r+1.
	 */
	int radius;
	/**
	 * Holds a tighter bounding box of the object in the original image scale
	 */
	cv::Rect rect;
	std::vector<int> mask_list;
	/**
	 * Pyramid level of the template (reduction_factor = 2^level)
	 */
	int level;
	/**
	 * The list of gradients in the template
	 */
	std::vector<unsigned char> binary_gradients;
	/**
	 * indices to use for matching (skips zeros inside binary_gradients)
	 */
	std::vector<int> match_list;
	/**
	 * This is a match list of list of sub-parts. Currently unused.
	 */
	std::vector<std::vector<int> > occlusions;

	std::vector<BinarizedGradientTemplate> templates;

	std::vector<float> hist;


	BinarizedGradientTemplate() : radius(0.0)
	{
		rect.x = rect.y = rect.width = rect.height = 0;
	}


	/**
	 * Score this template against another template
	 * @param test the other template
	 * @param match_thresh allow for early exit if you are not above this
	 * @return the score
	 */
	float score(const BinarizedGradientTemplate& test, float match_thresh,
		vector< vector<unsigned char> >& match_table, int match_method) const
	{
	    if(radius != test.radius) {
	    	return -1.0;
	    }
	    float total = (float)match_list.size();
	    if (total == 0.0) {
	    	return -1.0;
	    }
	    float num_test = (float)test.match_list.size();
	    if((num_test/total) < match_thresh) {
	    	return num_test/total; //Not enough entries in the other list to be above match_thresh
	    }
	    int matches = 0;
	    std::vector<int>::const_iterator imodel = match_list.begin();
		if(match_method == 0) {
			int limit = (int)(total*(1.0 - match_thresh) + 0.5); //Miss more than this number and we can't be above match_thresh
			while(imodel != match_list.end())
			{
				if (binary_gradients[*imodel]==0 && test.binary_gradients[*imodel]==0) {
					++matches;
				}
				else if((binary_gradients[*imodel])&(test.binary_gradients[*imodel])) {
					++matches;
				}
				else if (!(--limit)) {
					return(match_thresh - 0.000001); //sunk below the limit of misses, early terminate
				}
				++imodel;
			}
		} else { //match_method == 1, so we use the cosine matching table
			int limit = (int)(total*match_thresh*100.0); //Since the matchtable are unsigned chars going from 0 to 100;
			int res;
			while(imodel != match_list.end())
			{
				res = (int)(match_table[binary_gradients[*imodel]][test.binary_gradients[*imodel]]);
				matches += res;
				if(matches > limit)
					return(match_thresh + 0.0001);
				++imodel;
			}			
			matches /= 100; //Since we were 100x too high
		}
	    return( ((float)matches)/total);
	}

//	float score_view(const BinarizedGradientTemplate& test, float match_thresh,
//	std::vector<int>& match_indices, std::vector<int>& nonmatch_indices,
//	vector< vector<unsigned char> >& match_table, int match_method) const
//	{
//	    if(radius != test.radius) {
//	    	return -1.0;
//	    }
//	    float total = (float)match_list.size();
//	    if (total == 0.0) {
//	    	return -1.0;
//	    }
//	    float num_test = (float)test.match_list.size();
//	    if((num_test/total) < match_thresh) {
//	    	return num_test/total; //Not enough entries in the other list to be above match_thresh
//	    }
//	    int matches = 0;
//	    std::vector<int>::const_iterator imodel = match_list.begin();
//		if(match_method == 0) {
//			int limit = (int)(total*(1.0 - match_thresh) + 0.5); //Miss more than this number and we can't be above match_thresh
//			while(imodel != match_list.end())
//			{
//				if((binary_gradients[*imodel])&(test.binary_gradients[*imodel])) {
//					++matches;
//					match_indices.push_back(*imodel);
//				}
//				else {
//					nonmatch_indices.push_back(*imodel);
//					if (!(--limit)) {
//						return (match_thresh - 0.000001); //sunk below the limit of misses, early terminate
//					}
//				}
//				++imodel;
//			}
//		} else { //match_method == 1, so we use the cosine matching table
//			unsigned int limit = (unsigned int)(match_thresh*100.0); //Since the matchtable are unsigned chars going from 0 to 100;
//			unsigned int max_score = 100*total;
//			unsigned int res;
//			while(imodel != match_list.end())
//			{
//				res = (unsigned int)(match_table[binary_gradients[*imodel]][test.binary_gradients[*imodel]]);
//				if(res == 100)
//					match_indices.push_back(*imodel); 	//RECORDING ONLY PERFECT MATCHES AS "MATCHED"
//				else
//					nonmatch_indices.push_back(*imodel);//ELSE NOT MATCHED
//				matches += res;
//				max_score -= (100 - res); //Remember, we coded perfect match to be 100.
//				if(max_score < limit)     //We have sunk below the limit possible to make a match
//					return(match_thresh - 0.0001);
//				++imodel;
//			}
//			matches /= 100; //Since we were 100x too high
//		}
//	    return( ((float)matches)/total);
//	}



	/**
	 * Serialization support, using boost serialization.
	 * @param ar
	 * @param version
	 */
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
//		ar & id;
		ar & radius;
		ar & rect.x;
		ar & rect.y;
		ar & rect.width;
		ar & rect.height;
		ar & mask_list;
		ar & level;
		ar & binary_gradients;
		ar & match_list;
		ar & hist;
		ar & templates;
	}
};

/////////////////////////////////////////////////////
// Hash experiments
/////////////////////////////////////////////////////

struct Bucket
{
	int idx;
	Bucket* next;
};

class HashFunction
{
public:
	HashFunction(const std::vector<int> idx) : idx_(idx)
	{
		buckets_.resize(1<<idx.size(),NULL);
	}

	int hash(const std::vector<unsigned char>& value)
	{
//
//		for (size_t i=0;i<value.size()*8;++i) {
//			int pos = i/8;
//			int bit = i%8;
//			if ((value[pos] & (1<<bit))>>bit) {
//				printf("%d ", int(i));
//			}
//		}
//		printf("\n");


		int val = 0;
		for (std::vector<int>::const_iterator it = idx_.begin();it!=idx_.end();++it) {
			int pos = *it/8;
			int bit = *it%8;
//			printf("%d %d\n", *it, value[pos]);
			val = val<<1 | ((value[pos] & (1<<bit))>>bit);
		}
		return val;
	}

	void add_value(const std::vector<unsigned char>& value, int idx)
	{
		int hidx = hash(value);
		if (hidx==0) return;
//		printf("hidx: %d\n", hidx);
		Bucket* b = new Bucket();
		b->idx = idx;
		b->next = buckets_[hidx];
		buckets_[hidx] = b;
	}

	void get_indices(const std::vector<unsigned char>& value, std::vector<int>& indices)
	{
		int hidx = hash(value);
		if (hidx==0) return;
		Bucket* b = buckets_[hidx];
		while (b) {
			indices.push_back(b->idx);
			b = b->next;
		}
	}

	const std::vector<int> idx_;
	std::vector<Bucket*> buckets_;
};




/////////////////////////////////////////////////////
// BIGG
/////////////////////////////////////////////////////
class BinarizedGradientGrid : public Detector, public Trainable
{
public:

	BinarizedGradientGrid(ModelStoragePtr model_storage);
	vector< vector< unsigned char> > match_table;

	void computeHashFunctions();

	/**
	 * Loads pre-trained models for a list of objects.
	 * @param models vector of objects to load
	 */
	virtual void loadModels(const std::vector<std::string>& models);


	/**
	 * Construct a template from a region of a gradisnt summary image.
	 * @param tpl the BinarizedGradientTemplate structure.
	 * @param gradSummary the single channel uchar binary gradient summary image
	 * @param xc the center of a template in the gradSummary (in reduced scale), x coordinate
	 * @param yc the center of a template in the gradSummary (in reduced scale), ycoordinate
	 * @param r the radius of the template (in reduced scale)
	 * @param reduct_factor reduction factor
	 */
	void fillTemplateFromGradientImage(BinarizedGradientTemplate& tpl, const cv::Mat& gradSummary, int xc, int yc, int r, int reduct_factor);

	void fillTemplateFromGradientImage(BinarizedGradientTemplate& tpl, const cv::Mat& gradSummary, const cv::Rect& roi,
			const cv::Mat& mask, const cv::Mat& img_mask, int r, int reduct_factor);



	void detect(const cv::Mat& img, int level, const BinarizedGradientPyramid& pyr, const std::vector<cv::Point2i>& locations,
			const std::vector<BinarizedGradientTemplate>& templates, std::vector<BiGGDetection>& detections,
			float template_radius, float accept_threshold, float accept_threshold_decay);
	/**
	 * Run detection for this object class.
	 * @param gradientSummary
	 * @param detections
	 */
	void detect(const cv::Mat& gradSummary,
			const std::vector<cv::Point2i> locations, const std::vector<BinarizedGradientTemplate>& templates,
			std::vector<BiGGDetection>& detections,
			float template_radius, int level, float accept_threshold);


	void flat_detect(const cv::Mat& gradSummary,
			std::vector<BiGGDetection>& detections,
			float template_radius, int level, float accept_threshold);

	/**
	 * Runs the object detection of the current image.
	 */
	virtual void detect();


	/**
	 * Starts training for a new object category model. It may allocate/initialize
	 * data structures needed for training a new category.
	 * @param name
	 */
	virtual void startTraining(const std::string& name);


	void trainInstance(const cv::Mat& img, int level, const BinarizedGradientPyramid& pyr, 
			const BinarizedGradientPyramid& mask_pyr, std::vector<BinarizedGradientTemplate>& templates, 
			const cv::Rect& roi, const cv::Mat& mask, float template_radius, float accept_threshold);

	/**
	 * Trains the model on a new data instance.
	 * @param name The name of the model
	 * @param data Training data instance
	 */
	virtual void trainInstance(const std::string& name, const TrainingData& data);

	/**
	 * Saves a trained model.
	 * @param name model name
	 */
	virtual void endTraining(const std::string& name);



	/**
	 * Returns the name of the detector.
	 * @return name of the detector
	 */
	inline std::string getName() { return "BiGG"; }


	/**
	 * Getter and setter for template_radius_.
	 */
	inline int getTemplateRadius() const { return template_radius_;	}
	inline void setTemplateRadius(int template_radius)	{ template_radius_ = template_radius; }

	/**
	 * Getter and setter for accept_threashold_.
	 */
	inline float getAcceptThreshold() const	{ return accept_threshold_; }
	inline void setAcceptThreshold(float accept_threashold)	{ accept_threshold_ = accept_threashold; }


	/**
	 * Getter and setter for accept_threashold_decay_.
	 */
	inline float getAcceptThresholdDecay() const { return accept_threshold_decay_; }
	inline void setAcceptThresholdDecay(float accept_threashold_decay)	{ accept_threshold_decay_ = accept_threashold_decay; }

	/*
	 * Getter and setter for magnitude_threshold_.
	 */
	inline float getMagnitudeThreshold() const	{ return magnitude_threashold_; }
	inline void setMagnitudeThreshold(float magnitude_threashold) { magnitude_threashold_ = magnitude_threashold; }

	/*
	 * Getter and setter for start_level_.
	 */
	int getStartLevel()	{ return start_level_;	}
	void setStartLevel(int start_level)	{ start_level_ = start_level; }

	/*
	 * Getter and setter for levels_
	 */
	int getLevels()	{ return levels_; }
	void setLevels(int levels) { levels_ = levels; }

	/*
	 * Getter and setter for fraction_overlap_.
	 */
	inline float getFractionOverlap() const	{ return fraction_overlap_; 	}
	inline void setFractionOverlap(float fraction_overlap) { fraction_overlap_ = fraction_overlap;	}

	/*
	 * Getter and setter for match_method_.
	 */
	inline int getMatchMethod() const	{ return match_method_;}
	inline void setMatchMethod(int match_method) { match_method_ = match_method;}


	/**
	 * Serialization support, using boost serialization.
	 * @param ar
	 * @param version
	 */
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & names_;
		ar & root_templates_;
	}

private:

	/**
	 * Compute magnitude and phase in degrees from single channel image
	 * @param img input image
	 * @param mag gradient magnitude, returned
	 * @param phase gradient phase, returned
	 *
	 * \pre img.channels()==1 && img.rows>0 && img.cols>0
	 */
	void computeGradients(const cv::Mat &img, cv::Mat &mag, cv::Mat &phase);

	/**
	 * Turn mag and phase into a binary representation of 8 gradient directions.
	 * @param mag Floating point gradient magnitudes
	 * @param phase Floating point angles
	 * @param binryGradient binarized gradients, returned value. This will be allocated if it is empty or is the wrong size.
	 * @return
	 *
	 * \pre mag.rows==phase.rows && mag.cols==phase.cols
	 * \pre mag.rows>0 && mag.cols>0
	 */
	void binarizeGradients(const cv::Mat &mag, const cv::Mat &phase, cv::Mat &binaryGradient);

	/**
	 * Filter out noisy gradients via non-max suppression in a 3x3 area.
	 * @param binaryGradient input binarized gradient
	 * @param cleaned gradient, will be allocated if not already
	 */
	void gradMorphology(cv::Mat &binaryGradient, cv::Mat &cleanedGradient);

//	/**
//	 * This function will OR gradients together within a given square region of diameter d.
//	 * No checking is done to keep this in bounds ... you must do that
//	 * @param grad The single channel uchar image of binarized gradients
//	 * @param xx The center pixel, x coordinate
//	 * @param yy The center pixel, y coordinate
//	 * @param d The diameter
//	 * @return The value of the central byte containing the OR'd gradients of this patch.
//	 */
//	inline unsigned char orGradients(const cv::Mat &grad, int xx, int yy, int d);
//
//	/**
//	 * This function will produce a downsampled by logical OR'ing of the binarized gradient image.
//	 * The reduction factor of the resulting gradSummary image is the "reduction_factor" parameter,
//	 * the diameter of each summary region is "diameter" parameter
//	 * @param grad The single channel uchar image of binarized gradients
//	 * @param gradSummary This is the downsampled (by "reduction_factor") image containing the OR of
//	 * the binarized gradient in each patch of size "diameter"
//	 */
//	void gradientSummaryImage(const cv::Mat &grad, cv::Mat &gradSummary, int factor);



	/**
	 * Computes the downsampled binarized gradient summary image from the initial image.
	 * @param img
	 * @param gradSummary
	 */
//	void imgToGradientSummary(const cv::Mat &img, cv::Mat &gradSummary);
//	void imgToGradientSummary2(const cv::Mat &img, cv::Mat &gradSummary);


protected:
    // PARAMETERS
    /**
     * The radius of the template  (in reduced scale image)
     */
    int template_radius_;

    /**
     * Abandon early if you slip below this (1 is perfect, 0 is a total loss
     */
    float accept_threshold_;

    /**
     * How much teh accept threshold decays on each lower levels of teh pyramid
     */
    float accept_threshold_decay_;

    /**
     * Ignore gradients whose magnitude is lower
     */
    float magnitude_threashold_;


    /**
     * Start level in teh pyramid
     */
    int start_level_;

    /**
     * Levels in the pyramid
     */
    int levels_;

    /**
     * Fraction overlap between two detections above which one of them is suppressed.
     */
    float fraction_overlap_;
    
    /**
     * If match_match_method is 1, use cosine matching, if 0, use And of bytes
     */
    int match_method_;


    std::vector<HashFunction*> hash_functions_;

    std::vector<std::string> names_;
    int crt_object_;
    std::vector<BinarizedGradientTemplate> root_templates_;
    std::vector<BinarizedGradientTemplate> templates_;

    typedef std::vector<BinarizedGradientTemplate> BiGGTemplateTree;
    std::vector<BiGGTemplateTree> all_templates_;

};

}


#endif /* BINARY_GRADIENT_GRID_H_ */
