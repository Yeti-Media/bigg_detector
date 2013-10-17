/*
O* * Software License Agreement (BSD License)
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


#include "bigg_detector/bigg.h"
#include <rein/nodelets/type_conversions.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <fstream>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/vector.hpp>


#define foreach BOOST_FOREACH

REGISTER_TYPE_CONVERSION(rein::Rect, cv::Rect,
		(x,x)
		(y,y)
		(width,width)
		(height,height)
)

REGISTER_TYPE_CONVERSION(rein::Detection, bigg_detector::BiGGDetection,
		(label,name)
		(score,score)
		(mask.roi,roi)
		(mask.mask,mask)
)



namespace { // anonymous namespace

	using namespace bigg_detector;


	// this namespace contains helper functions used mainly for debugging
	// they are all placed in the anonymous namespace since they are not used outside of this compilation unit


	void extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask = cv::Mat())
	{
		static cv::FeatureDetector* featureDetector = new cv::FastFeatureDetector(10/*threshold*/, true/*nonmax_suppression*/);
		static cv::DescriptorExtractor* descriptorExtractor = new cv::SiftDescriptorExtractor;

		cv::Mat img_gray;
		// compute the gradient summary image
		if(image.channels() != 1) {
			cv::cvtColor(image, img_gray, CV_BGR2GRAY);
		}
		else {
			img_gray = image;
		}


		printf("In extractFeatures()\n");
		featureDetector->detect(img_gray, keypoints, mask);
		descriptorExtractor->compute(img_gray, keypoints, descriptors);

		printf("Extracted %d features\n", (int)keypoints.size());
	}



	void show_keypoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
	{
		for (std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin(); it < keypoints.end(); ++it) {
			cv::Point p = it->pt;
			cv::circle(img, cv::Point(p.x, p.y), 3, CV_RGB(255, 0, 0));
		}

		cv::namedWindow("keypoints", 1);
		cv::imshow("keypoints", img);
		cv::waitKey(30);
	}



	template <typename T>
	std::string tostr(const T& t)
	{
		std::ostringstream os;
		os<<t;
		return os.str();
	}

	void show_detection_on_image(cv::Mat& show_img, const BiGGDetection& det, float dist = -1)
	{
		cv::Point p1;
		p1.x = det.roi.x;
		p1.y = det.roi.y;
		cv::Point p2;
		p2.x = det.roi.x+det.roi.width;
		p2.y = det.roi.y+det.roi.height;
		cv::rectangle(show_img, p1,p2,cv::Scalar(255,0,0),2);
		cv::putText(show_img,det.name,cv::Point(p1.x + 3,p1.y+ 12),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(255,0,0),2);
		std::string strscore = std::string("(") + tostr(det.score) + std::string(")");
		putText(show_img,strscore,cv::Point(p1.x + 6,p1.y+24),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0),1);

		if (dist>0) {
			std::string strdist = std::string("(") + tostr(dist) + std::string(")");
			putText(show_img,strdist,cv::Point(p1.x + 6,p1.y+50),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0),1);
		}
	}

	void show_detections_on_image(const cv::Mat& img, const std::vector<BiGGDetection>& det, const char* name, float dist = -1)
	{
		cv::Mat show_img;
		img.copyTo(show_img);

		for (size_t i=0;i<det.size();++i) {
			cv::Point p1;
			p1.x = det[i].roi.x;
			p1.y = det[i].roi.y;
			cv::Point p2;
			p2.x = det[i].roi.x+det[i].roi.width;
			p2.y = det[i].roi.y+det[i].roi.height;
			cv::rectangle(show_img, p1,p2,cv::Scalar(255,0,0),2);
	        cv::putText(show_img,det[i].name,cv::Point(p1.x + 3,p1.y+ 12),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(255,0,0),2);
	        std::string strscore = std::string("(") + tostr(det[i].score) + std::string(")");
	        putText(show_img,strscore,cv::Point(p1.x + 6,p1.y+24),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0),1);

	        if (dist>0) {
	            std::string strdist = std::string("(") + tostr(dist) + std::string(")");
	            putText(show_img,strscore,cv::Point(p1.x + 6,p1.y+36),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0),1);
	        }

		}

		cv::namedWindow(name,1);
		cv::imshow(name,show_img);
		cv::waitKey(0);
	}

	void show_template(const BinarizedGradientTemplate& tpl, const std::string wname)
	{
		int span = 2*tpl.radius;
		cv::Mat tpl_img(span,span,CV_8UC3, cv::Scalar::all(0));
		for (size_t i=0;i<tpl.match_list.size();++i) {
			int pos = tpl.match_list[i];
			int y = pos/span;
			int x = pos%span;
			uchar val = tpl.binary_gradients[pos];
			uchar* im = tpl_img.ptr(y)+3*x;
			im[0] = val&0xC0;
			im[1] = (val&0x38)<<2;
			im[2] = (val&0x07)<<5;
		}

		cv::namedWindow(wname.c_str(),1);
		cv::imshow(wname.c_str(),tpl_img);
		cv::imwrite((wname+".png").c_str(), tpl_img);
		cvWaitKey(30);
	}

	void show_matching_templates(const BinarizedGradientTemplate& actual, const BinarizedGradientTemplate& tpl )
	{
		show_template(actual,"actual");
		show_template(tpl,"tpl");
	}


	void showBinarizedGradient(const cv::Mat& binGrad, cv::Mat& image)
	{
		for (int i=0;i<binGrad.rows;++i) {
			const uchar* bgp = binGrad.ptr(i);
			uchar* im = image.ptr(i);
			for (int j=0;j<binGrad.cols;++j) {
				im[0] = ((*bgp)&0xC0);
				im[1] = ((*bgp)&0x38)<<2;
				im[2] = ((*bgp)&0x07)<<5;
				im+=3;
				++bgp;
			}
		}
	}



	void show_template(cv::Mat& image, const BiGGDetection& id, const BinarizedGradientTemplate& tpl, const BinarizedGradientTemplate& crt_tpl,
			const std::vector<int>& indices, const std::vector<int>& nm_indices)
	{
		int s = 2;
		cv::Size size = image.size();
	    cv::Mat show_image;
	    size.height*=s;
	    size.width*=s;
	    cv::resize(image,show_image,size);
	//    image.copyTo(show_image);



	    int reduction_factor = s*(1<<tpl.level);
		int r = reduction_factor/2;
	    int d = (int) (2*tpl.radius);

	    for (size_t i=0;i<crt_tpl.binary_gradients.size();++i) {
	    	uchar value = crt_tpl.binary_gradients[i];

	    	int x = i%d;
	    	int y = i/d;
	    	int xd = (id.x+x-tpl.radius)*reduction_factor;
	    	int yd = (id.y+y-tpl.radius)*reduction_factor;
	    	int xc = xd+r;
	    	int yc = yd+r;

	    	for (int i=0;i<8;++i) {
	    		if (value&(1<<i)) {
	    			double angle = M_PI*i*22.5/180;
	    			int xstart = xc-r*cos(angle);
	    			int ystart = yc-r*sin(angle);
	    			int xend = xc+r*cos(angle);
	    			int yend = yc+r*sin(angle);

	    			cv::line(show_image, cv::Point(xstart,ystart),cv::Point(xend,yend), cv::Scalar(255,255,0),1);
	    		}
	    	}
	    }


	    for (size_t i=0;i<indices.size();++i) {
	    	uchar value = tpl.binary_gradients[indices[i]];

	    	int x = indices[i]%d;
	    	int y = indices[i]/d;
	    	int xd = (id.x+x-tpl.radius)*reduction_factor;
	    	int yd = (id.y+y-tpl.radius)*reduction_factor;
	    	int xc = xd+r;
	    	int yc = yd+r;

	    	for (int i=0;i<8;++i) {
	    		if (value&(1<<i)) {
	    			double angle = M_PI*i*22.5/180;
	    			int xstart = xc-r*cos(angle);
	    			int ystart = yc-r*sin(angle);
	    			int xend = xc+r*cos(angle);
	    			int yend = yc+r*sin(angle);

	    			cv::line(show_image, cv::Point(xstart,ystart),cv::Point(xend,yend), cv::Scalar(255,0,0),1);
	    		}
	    	}
	    }

	    for (size_t i=0;i<nm_indices.size();++i) {
	    	uchar value = tpl.binary_gradients[nm_indices[i]];
	    	int x = nm_indices[i]%d;
	    	int y = nm_indices[i]/d;
	    	int xd = (id.x+x-tpl.radius)*reduction_factor;
	    	int yd = (id.y+y-tpl.radius)*reduction_factor;
	    	int xc = xd+r;
	    	int yc = yd+r;
	    	for (int i=0;i<8;++i) {
	    		if (value&(1<<i)) {
	    			double angle = M_PI*i*22.5/180;
	    			int xstart = xc-r*cos(angle);
	    			int ystart = yc-r*sin(angle);
	    			int xend = xc+r*cos(angle);
	    			int yend = yc+r*sin(angle);

	    			cv::line(show_image, cv::Point(xstart,ystart),cv::Point(xend,yend), cv::Scalar(0,0,255),1);
	    		}
	    	}

	    }

		cv::namedWindow("gradients",1);
	    cv::imshow("gradients", show_image);
	    cv::waitKey(30);


	}


	void show(const cv::Mat& mat, const char* name)
	{
		cv::namedWindow(name,1);
		cv::imshow(name,mat);
		cv::waitKey(30);
	}


} // namespace



namespace bigg_detector {

/**
 * Determines if two rectangles intersect
 * @param a
 * @param b
 * @return boolean value indicating if the two rectangles intersect
 */
bool intersect(const cv::Rect& a, const cv::Rect& b)
{
	return ((a.x < (b.x + b.width)) && ((a.x + a.width) > b.x) &&
		((a.y + a.height) >  b.y) && (a.y < (b.y + b.height)));
}


/**
 * Computes the fraction of the intersection of two rectangles with respect to the
 * total area of the rectangles.
 * @param a
 * @param b
 * @return intersection area
 */
float rectFractOverlap(const cv::Rect& a, const cv::Rect& b)
{
	if(intersect(a, b))
	{
	    float total_area = b.height*b.width + a.width*a.height;
		int left = a.x > b.x ? a.x : b.x;
		int top = a.y > b.y ? a.y : b.y;
		int right = a.x + a.width < b.x + b.width ? a.x + a.width : b.x + b.width;
		int width = right - left;
		int bottom = a.y + a.height < b.y + b.height ? a.y + a.height : b.y + b.height;
		int height = bottom - top;
		return (2.0*height*width/(total_area+0.000001));  //Return the fraction of intersection
	}
	return 0.0;
}


/**
 * Suppress overlapping rectangle to be the rectangle with the highest score
 * @param detections vector of detections to work with
 * @param frac_overlap what fraction of overlap between 2 rectangles constitutes overlap
 */
void nonMaxSuppress(std::vector<bigg_detector::BiGGDetection>& detections, float frac_overlap)
{
	int len = (int)detections.size();

	for(int i = 0; i<len - 1; ++i) {
		for(int j = i+1; j<len; ++j) {
			float measured_frac_overlap = rectFractOverlap(detections[i].roi,detections[j].roi);
//			cout << "measured_frac_overlap (" << measured_frac_overlap << " >? )" << frac_overlap << endl;
			if(measured_frac_overlap > frac_overlap)
			{
//			    cout << "    YES!" << endl;
				if(detections[i].score >= detections[j].score) {
					std::swap(detections[j], detections[len-1]);
					len -= 1;
					j -= 1;
				}
				else {
					std::swap(detections[i], detections[len-1]);
					len -= 1;
					i -= 1;
					break;
				}
			}
		}
	}

	detections.resize(len);
}


/**
 * Suppress overlapping rectangle to be the rectangle with the highest score
 * @param detections vector of detections to work with
 * @param frac_overlap what fraction of overlap between 2 rectangles constitutes overlap
 */
void nonMaxSuppress2(std::vector<bigg_detector::BiGGDetection>& detections, float frac_overlap)
{
	int len = (int)detections.size();

	for(int i = 0; i<len - 1; ++i) {
		for(int j = i+1; j<len; ++j) {

			float raport = detections[i].score/detections[j].score;
			if (raport>1) raport = 1/raport;
			if (raport >0.95) continue;


			if (detections[i].score==detections[j].score) continue;
			float measured_frac_overlap = rectFractOverlap(detections[i].roi,detections[j].roi);
//			cout << "measured_frac_overlap (" << measured_frac_overlap << " >? )" << frac_overlap << endl;
			if(measured_frac_overlap > frac_overlap)
			{
//			    cout << "    YES!" << endl;
				if(detections[i].score >= detections[j].score) {
					std::swap(detections[j], detections[len-1]);
					len -= 1;
					j -= 1;
				}
				else {
					std::swap(detections[i], detections[len-1]);
					len -= 1;
					i -= 1;
					break;
				}
			}
		}
	}

	detections.resize(len);
}


//////////////////////////////BinarizedGradientPyramid////////////////////////////////////


BinarizedGradientPyramid::BinarizedGradientPyramid(const cv::Mat& gradientImage, int start_level_, int levels_)
{
	start_level = start_level_;
	levels = levels_;

	cv::Mat crt = gradientImage;
	for (int i=0;i<start_level+levels;++i)  {
		if (i>=start_level) {
			pyramid.push_back(crt);
		}
		if (i==(start_level+levels-1)) break;
		cv::Mat scaled;
		scaled.create(crt.rows/2, crt.cols/2, CV_8UC1);
		scaled = cv::Scalar(0,0,0);

		int rows = crt.rows - 2;
		int cols = crt.cols - 2; // "-reduction_factor_" to protect against trying to summarize past the edge of the image
		for(int Y = 0, y = 0; Y<rows; Y+=2, ++y)
		{
			uchar *s = scaled.ptr(y);
			uchar *c1 = crt.ptr(Y);
			uchar *c2 = crt.ptr(Y+1);
			for(int X = 0; X<cols; X+=2, ++s, c1+=2, c2+=2)
			{
				*s = *c1 | *(c1+1) | *c2 | *(c2+1);
			}
		}
		crt = scaled;
	}
}



///////////////////////////BinarizedGradientGrid/////////////////////////////////////


/**
 * Construct a template from a region of a gradisnt summary image.
 * @param tpl The BinarizedGradientTemplate structure.
 * @param gradSummary The single channel uchar binary gradient summary image
 * @param xc The center of a template in the gradSummary (in reduced scale), x coordinate
 * @param yc The center of a template in the gradSummary (in reduced scale), ycoordinate
 * @param r The radius of the template (in reduced scale)
 * @param reduct_factor
 */
void BinarizedGradientGrid::fillTemplateFromGradientImage(BinarizedGradientTemplate& tpl, const cv::Mat& gradSummary, int xc, int yc, int r, int level)
{
    int span = 2*r;
    tpl.radius = r;
    tpl.level = level;
    tpl.binary_gradients.assign(span*span,0); //Create the template
    tpl.match_list.clear();
    tpl.match_list.reserve(span*span);

    //Bear with me, we have to worry a bit about stepping off the iamge boundaries:
    int rows = gradSummary.rows;
    int cols = gradSummary.cols;
    //y
    int ystart = yc - r;
    int yoffset = 0; //offset before you reach the playing field
    if(ystart < 0) {yoffset = -ystart; ystart = 0;}
    int yend = yc + r;
    if(yend > rows) yend = rows;

    //x
    int xstart = xc - r;
    int xoffset = 0;//offset before you reach the playing field
    if(xstart < 0)  {xoffset = -xstart; xstart = 0;}
    int xend = xc + r;
    if(xend > cols) xend = cols;

    tpl.hist.resize(8);
    int cnt = 0;

    //Fill the binary _gradients
    for(int y =  ystart; y<yend; ++y)
    {
        const uchar *b = gradSummary.ptr<uchar>(y);
        b += xstart;
        for(int x = xstart; x<xend; ++x, ++b)
        {
            int index = (yoffset + y - ystart)*span + (xoffset + x - xstart);//If this were an image patch, this is the offset to it
            tpl.binary_gradients[index] = *b;
            if(*b) {
            	//Record where gradients are
            	tpl.match_list.push_back(index);
            }

            for (int i=0;i<8;++i) {
            	if (*b&(1<<i)) {
            		tpl.hist[i]+=1;
            		cnt++;
            	}
            }
        }
    }

    for (int i=0;i<8;++i) {
    	tpl.hist[i] /= cnt;
    }
}



void BinarizedGradientGrid::fillTemplateFromGradientImage(BinarizedGradientTemplate& tpl, const cv::Mat& gradSummary, const cv::Rect& roi, const cv::Mat& mask, const cv::Mat& img_mask, int r, int level)
{
    int span = 2*r;
//    tpl.id = crt_object_;
    tpl.radius = r;
    tpl.level = level;
    tpl.binary_gradients.assign(span*span,0); //Create the template
    tpl.match_list.clear();
    tpl.match_list.reserve(span*span);
    tpl.rect.x = tpl.rect.y = 0;
    tpl.rect.width = roi.width;
    tpl.rect.height = roi.height;

//
//    cv::namedWindow("test2",1);
//    cv::imshow("test", mask);
//    cv::waitKey(30);

    // set mask
    tpl.mask_list.resize(mask.rows*mask.cols);
    int idx = 0;
    for (int y=0;y<mask.rows;++y) {
    	for (int x=0;x<mask.cols;++x) {
    		if (mask.at<uchar>(y,x)>0) {
    			tpl.mask_list[idx++] = y*mask.cols+x;
    		}
    	}
    }
    tpl.mask_list.resize(idx);


    int reduction_factor = 1<<level;

    int xc = (roi.x + roi.width/2) / reduction_factor;
	int yc = (roi.y + roi.height/2) / reduction_factor;
	int xr = roi.width/2 / reduction_factor;
	int yr = roi.height/2 / reduction_factor;
    int rows = gradSummary.rows;
    int cols = gradSummary.cols;

//    printf("x: %d, y: %d\n", xc,yc);

    if (xr>r) xr = r;
    if (yr>r) yr = r;

    //y
    int ystart = yc - yr;
    int yoffset = r-yr; //offset before you reach the playing field
    if (ystart < 0) { yoffset = -ystart; ystart = 0; }
    int yend = yc + yr;
    if (yend > rows) yend = rows;

    //x
    int xstart = xc - xr;
    int xoffset = r-xr;//offset before you reach the playing field
    if (xstart < 0)  { xoffset = -xstart; xstart = 0; }
    int xend = xc + xr;
    if (xend > cols) xend = cols;

//    printf("xstart: %d, xend: %d, xoffset: %d, ystart: %d, yend: %d, yoffset: %d\n",
//    		xstart,xend,xoffset,ystart,yend,yoffset);
//    cv::Rect rect;
//    rect.x = xstart+xoffset;
//    rect.y = ystart+yoffset;
//    rect.width = xend-xstart-xoffset;
//    rect.height = yend-ystart-yoffset;
//    cv::Mat tpl_img = gradSummary(rect);
//    cv::namedWindow("tpl",1);
//    cv::imshow("tpl",tpl_img);
//    cv::waitKey(0);

    tpl.hist.resize(8);
//    int cnt = 0;

//
//    cv::Mat img(gradSummary.rows, gradSummary.cols, CV_8UC3);
//    showBinarizedGradient(gradSummary, img);
//    cv::namedWindow("grad_sum",1);
//    cv::imshow("grad_sum", img);
//
//    cv::namedWindow("mask",1);
//    cv::imshow("mask", img_mask);
//    cv::waitKey(0);

    //Fill the binary _gradients
    for(int y =  ystart; y<yend; ++y)
    {
        const uchar *b = gradSummary.ptr<uchar>(y);
        const uchar *m = img_mask.ptr<uchar>(y);
        b += xstart;
        m += xstart;
        for(int x = xstart; x<xend; ++x, ++b, ++m)
        {
            int index = (yoffset + y - ystart)*span + (xoffset + x - xstart);//If this were an image patch, this is the offset to it
            tpl.binary_gradients[index] = *b;
            if(*b && *m) {
            	//Record where gradients are
            	tpl.match_list.push_back(index);
            }
//            printf("Match list size: %d\n", tpl.match_list.size());

//            for (int i=0;i<8;++i) {
//            	if (*b&(1<<i)) {
//            		tpl.hist[i]+=1;
//            		cnt++;
//            	}
//            }
        }
    }

//    for (int i=0;i<8;++i) {
//    	tpl.hist[i] /= cnt;
//    }

}



//
//struct Offset
//{
//	Offset(int width) : width_(width){}
//	int operator()(int x,int y) { return y*width_+x; }
//	int width_;
//};


void BinarizedGradientGrid::detect(const cv::Mat& img, int level, const BinarizedGradientPyramid& pyr, const std::vector<cv::Point2i>& locations,
		const std::vector<BinarizedGradientTemplate>& templates, std::vector<BiGGDetection>& detections,
		float template_radius, float accept_threshold, float accept_threshold_decay)
{
//	printf("Detect on level: %d\n", level);
	float reduction_factor = float(1<<level);

	std::vector<BiGGDetection> crt_detections;
	detect(pyr[level], locations, templates, crt_detections, template_radius/reduction_factor, level, accept_threshold);

//	printf("Before nms, detections: %d\n", crt_detections.size());
//	nonMaxSuppress2(crt_detections, fraction_overlap_);
	printf("Detections: %d\n", (int)crt_detections.size());

	cv::Mat show_img;
	img.copyTo(show_img);

//	if (level==3 && crt_detections.size()>0) {
//		show_detections_on_image(show_img, crt_detections, "detections");
//	}


	if (level>pyr.start_level) {
		for (size_t i=0;i<crt_detections.size();++i) {
			std::vector<cv::Point2i> crt_locations;

			int dir[2][9] = {
					{ -1,-1,-1,0,0,0,1,1,1},
					{ -1,0,1,-1,0,1,-1,0,1}
			};
			crt_locations.resize(9);
			int x = crt_detections[i].x;
			int y = crt_detections[i].y;
			for (int k=0;k<9;++k) {
				crt_locations[k] = cv::Point2i(2*x+dir[0][k],2*y+dir[1][k]);
			}
			detect(img, level-1, pyr, crt_locations, templates[crt_detections[i].index].templates, detections, template_radius,
					accept_threshold*(1-accept_threshold_decay),accept_threshold_decay);
		}
	}
	else {
		std::copy(crt_detections.begin(), crt_detections.end(), back_inserter(detections));
	}
}


float dist(const std::vector<float>& a, const std::vector<float>& b)
{
	float result = 0;
	for (size_t i=0;i<a.size();++i) {
		result += (a[i]-b[i])*(a[i]-b[i]);
	}
	return result;
}

/**
 * Run detection for this object class.
 * @param gradientSummary
 * @param detections
 */
void BinarizedGradientGrid::detect(const cv::Mat& gradSummary,
		const std::vector<cv::Point2i> locations, const std::vector<BinarizedGradientTemplate>& templates,
		std::vector<BiGGDetection>& detections,
		float template_radius, int level, float accept_threshold)
{
//	cv::Mat respImage(gradSummary.rows, gradSummary.cols, CV_32FC1);

    int rows = gradSummary.rows;
    int cols = gradSummary.cols;
    BinarizedGradientTemplate crt_template;

    float reduction_factor = float(1<<level);

//    printf("In detect: rows: %d, cols: %d, template_radius: %g, reduction_factor: %g\n", rows, cols, template_radius, reduction_factor);

    if (locations.empty()) {
        for(int y = 5; y<rows - 5; ++y) {
        	for(int x = 5; x<cols-5; ++x) {
        		fillTemplateFromGradientImage(crt_template, gradSummary, x, y, template_radius, level);
        		for(int j = 0; j<(int)templates.size(); ++j) {
        			float res = templates[j].score(crt_template,accept_threshold,match_table,match_method_);
//        			printf("(%d, %d, %g), ",x,y,res);
    //    			respImage.at<float>(y,x) = res;
        			if(res < accept_threshold)
        			{
//        				printf("Level: %d, score: %g\n", level, res);
        				bigg_detector::BiGGDetection detection;
        				cv::Rect bbox = templates[j].rect;

        				detection.roi.x = reduction_factor*x - bbox.width/2;
        				detection.roi.y = reduction_factor*y - bbox.height/2;
        				detection.roi.width = bbox.width;
        				detection.roi.height = bbox.height;
        				detection.score = res;
//        				detection.id = templates[j].id;
        				detection.index = j;
        				detection.x = x;
        				detection.y = y;
        				detection.tpl = &templates[j];
        				detection.crt_tpl = boost::make_shared<BinarizedGradientTemplate>(crt_template);

        				detections.push_back(detection);
        			}
        		}
        	}
        }
    }
    else {
    	for (size_t i=0;i<locations.size();++i) {
    		int x = locations[i].x;
    		int y = locations[i].y;
    		fillTemplateFromGradientImage(crt_template, gradSummary, x, y, template_radius, level);
    		for(int j = 0; j<(int)templates.size(); ++j) {
    			//float hist_dist =  dist(templates[j].hist, crt_template.hist);
//    			if (hist_dist>0.01) continue;
    			float res = templates[j].score(crt_template,accept_threshold,match_table,match_method_);
//    			respImage.at<float>(y,x) = res;
//    			printf("Score: %g \n",res);
    			if(res < accept_threshold)
    			{
//    				show_matching_templates(crt_template, templates[j]);

//    				printf("Level: %d, score: %g\n", level, res);
    				bigg_detector::BiGGDetection detection;
    				cv::Rect bbox = templates[j].rect;

    				detection.roi.x = reduction_factor*x - bbox.width/2;
    				detection.roi.y = reduction_factor*y - bbox.height/2;
    				detection.roi.width = bbox.width;
    				detection.roi.height = bbox.height;
    				detection.score = res;
//    				detection.id = templates[j].id;
    				detection.index = j;
    				detection.x = x;
    				detection.y = y;
    				detection.tpl = &templates[j];
    				detection.crt_tpl = boost::make_shared<BinarizedGradientTemplate>(crt_template);

    				detections.push_back(detection);
    			}
    		}
    	}
    }

//    cv::namedWindow("resp", 1);
//    cv::imshow("resp", respImage);
//    cvWaitKey(40);

}





void BinarizedGradientGrid::flat_detect(const cv::Mat& gradSummary,
		std::vector<BiGGDetection>& detections,
		float template_radius, int level, float accept_threshold)
{

    int rows = gradSummary.rows;
    int cols = gradSummary.cols;
    BinarizedGradientTemplate crt_template;

	float reduction_factor = float(1<<level);
	float radius = template_radius/reduction_factor;

    for(int y = 5; y<rows - 5; y+=5) {
    	for(int x = 5; x<cols-5; x+=5) {
    		fillTemplateFromGradientImage(crt_template, gradSummary, x, y, radius, level);


    		for(int j = 0; j<(int)templates_.size(); ++j) {
    			float res = templates_[j].score(crt_template,accept_threshold,match_table,match_method_);
//    			respImage.at<float>(y,x) = res;
    			if(res < accept_threshold)
    			{
        				printf("Level: %d, score: %g\n", level, res);
    				bigg_detector::BiGGDetection detection;
    				cv::Rect bbox = templates_[j].rect;

    				detection.roi.x = reduction_factor*x - bbox.width/2;
    				detection.roi.y = reduction_factor*y - bbox.height/2;
    				detection.roi.width = bbox.width;
    				detection.roi.height = bbox.height;
    				detection.score = res;
//    				detection.id = templates_[j].id;
    				detection.index = j;
    				detection.x = x;
    				detection.y = y;
    				detection.tpl = &templates_[j];
    				detection.crt_tpl = boost::make_shared<BinarizedGradientTemplate>(crt_template);

    				detections.push_back(detection);
    			}
    		}
    	}
    }

}


void BinarizedGradientGrid::trainInstance(const cv::Mat& img, int level, const BinarizedGradientPyramid& pyr,
		const BinarizedGradientPyramid& mask_pyr,
		std::vector<BinarizedGradientTemplate>& templates, const cv::Rect& roi,
					const cv::Mat& mask, float template_radius, float accept_threshold)
{
	BinarizedGradientTemplate bgt;
	std::vector<BiGGDetection> detections;
	float reduction_factor = float(1<<level);
	std::vector<cv::Point2i> locations;
	locations.resize(9);
    int xc = (roi.x + roi.width/2) / reduction_factor;
	int yc = (roi.y + roi.height/2) / reduction_factor;
	int dir[2][9] = {
			{ -1,-1,-1,0,0,0,1,1,1},
			{ -1,0,1,-1,0,1,-1,0,1}
	};
	for (int k=0;k<9;++k) {
		locations[k] = cv::Point2i(xc+dir[0][k],yc+dir[1][k]);
	}
	detect(pyr[level], locations, templates, detections, template_radius/reduction_factor, level, accept_threshold);


	int template_id = -1;
	if (detections.size()==0) {
		printf("No detections, adding template: %d on level %d\n", (int)templates.size(), level);
		fillTemplateFromGradientImage(bgt, pyr[level], roi, mask, mask_pyr[level], template_radius/reduction_factor, level);

//		show_template(bgt, "template");
//		cv::waitKey(0);
		templates.push_back(bgt);
		template_id = templates.size()-1;
	}
	else {
		int max_id = 0;
		float max_score = detections[0].score;
		for (size_t i=1;i<detections.size();++i) {
			if (detections[i].score>max_score) {
				max_score = detections[i].score;
				max_id = i;
			}
		}

		printf("Template detected on level: %d, score: %g\n", level, detections[max_id].score);
		template_id = detections[max_id].index;
	}

	if (level>pyr.start_level) {
		trainInstance(img, level-1, pyr, mask_pyr, templates[template_id].templates, roi, mask, template_radius, accept_threshold);
	}
}


/* Helper circular shift functions for unsigned chars
 * assumes number of bits in an unsigned char is 8 */
 
unsigned char _rotl(unsigned char value, int shift) {
    if ((shift &= 7) == 0)
      return value;
    return (value << shift) | (value >> (8 - shift));
}
 
unsigned char _rotr(unsigned char value, int shift) {
    if ((shift &= 7) == 0)
      return value;
    return (value >> shift) | (value << (8 - shift));
}

BinarizedGradientGrid::BinarizedGradientGrid(ModelStoragePtr model_storage) : Detector(model_storage)
{
	//A somewhat inelegant way to create a unsigned char 256x256 look up table for cosine matching
	// In the binary vector, 4 positions away either direction are orthogonal (cos = 0)
	// Each adjecent position, whether right or left is a 22.5 degree missmatch since I don't consider
	// contrast (gradients go only from 0 to 180 degrees) and I binarize into bytes (180/8 = 22.5)
	// This table will be used if match_type = 1, else we'll use anding (and just disregard this table)
	match_table.resize(256);
	unsigned char a,b;
	for(unsigned int d1 = 0; d1 < 256; ++d1){
		match_table[d1].resize(256);
		for(unsigned int d2 = 0; d2 < 256; ++d2){
			a = (unsigned char)d1;
			b = (unsigned char)d2;
			if((a == 0) || (b == 0))
			{
				match_table[d1][d2] = 100; //100 => 1.0 but in binary so in the end, divide by 100
				continue;
			}
			if(a & b){
				match_table[d1][d2] = 0;
				continue;
			}
			unsigned char l,r; 
			l = _rotl(a,1);
			r = _rotr(a,1);
			if((l & b)||(r & b)){
				match_table[d1][d2] = 38; //100xSin match of 22.5 degree difference
				continue;
			}
			l = _rotl(a,2);
			r = _rotr(a,2);
			if((l & b)||(r & b)){
				match_table[d1][d2] = 71; //100xSin match of 2*22.5 degree difference
				continue;
			}
			l = _rotl(a,3);
			r = _rotr(a,3);
			if((l & b)||(r & b)){
				match_table[d1][d2] = 92; //100xSin match of 3*22.5 degree difference
				continue;
			}
			match_table[d1][d2] = 100;      //100xSin match of 90 degree difference
		}
	}
	match_table[0][0] = 0; //Call matching 0 against 0 a perfrect match (though indices are not taken there)
}

void print_tree_size(const std::vector<BinarizedGradientTemplate>& tree, int& count, int level = 1, const std::string& prefix = "")
{
	if (tree.size()==0) return;
	printf("%s Size on level %d is %d\n", prefix.c_str(), level, int(tree.size()));
	count += tree.size();
	for (size_t i=0;i<tree.size();++i) {
		print_tree_size(tree[i].templates, count, level+1, prefix+"  ");
	}

}


void extract_level_templates(int level, const std::vector<BinarizedGradientTemplate>& tree, std::vector<BinarizedGradientTemplate>& templates)
{
	for (size_t i=0;i<tree.size();++i) {
		if (tree[i].level==level) {
			templates.push_back(tree[i]);
		}
		else {
			extract_level_templates(level, tree[i].templates, templates);
		}
	}

}

template <typename T>
class SortIndex
{
public:
	SortIndex(const std::vector<T>& vec) : vec_(vec){};

	bool operator() (const size_t a, const size_t b) { return vec_[a]>vec_[b]; }

	const std::vector<T>& vec_;
};


int rand_int(int high, int low=0)
{
    return low + (int) ( double(high-low) * (std::rand() / (RAND_MAX + 1.0)));
}


void BinarizedGradientGrid::computeHashFunctions()
{
	if (templates_.size()==0) return;
	int template_size = templates_[0].binary_gradients.size();
	std::vector<float> cnt_vec(template_size*8,0);

	for (size_t i=0;i<templates_.size();++i) {
		BinarizedGradientTemplate& tpl = templates_[i];
		for (int j=0;j<template_size*8;++j) {
			int pos = j/8;
			int bit = j%8;
			if (tpl.binary_gradients[pos]&(1<<bit)) {
				cnt_vec[j]+=1;
			}
		}
	}
	for (int j=0;j<template_size*8;++j) {
		cnt_vec[j] /= templates_.size();
	}

	std::vector<int> indices(cnt_vec.size());
	for (int j=0;j<template_size*8;++j) {
		indices[j] = j;
	}
	std::sort(indices.begin(), indices.end(), SortIndex<float>(cnt_vec));

	int max_idx = 0;
	while (cnt_vec[indices[max_idx]]>0.1) {
		max_idx++;
	}

	int hash_count = 10;
	int hsize = 16;
	std::vector<int> hidx(hsize);
	hash_functions_.resize(hash_count);
	for (int i=0;i<hash_count;++i) {
		for (int j=0;j<hsize;++j) {
			hidx[j] = rand_int(indices[max_idx]);
		}
		hash_functions_[i] = new HashFunction(hidx);
		for (size_t j=0;j<templates_.size();++j) {
			hash_functions_[i]->add_value(templates_[j].binary_gradients,j);
		}
	}

//	hash_functions_[0]->add_value(templates_[0].binary_gradients,0);

	std::vector<int> indices2;

	for (int i=0;i<hash_count;++i) {
		hash_functions_[i]->get_indices(templates_[2].binary_gradients, indices2);
	}

	printf("Indices: ");
	for (size_t i=0;i<indices2.size();++i) {
		printf("%d ", indices2[i]);
	}
	printf("\n");
}

void BinarizedGradientGrid::loadModels(const std::vector<std::string>& models)
{
	names_ = models;
//	model_storage_->getModelList(getName(), names_);

	printf("Loading models\n");
	all_templates_.resize(names_.size());
	for( size_t i=0;i<names_.size();++i) {
		std::string& name = names_[i];
		printf("Loading model: %s\n", name.c_str());

		if (!model_storage_->load(name,getName(),all_templates_[i])) {
			printf("Cannot load model: %s\n", name.c_str());
		}
	}

//	int cnt = 0;
//	print_tree_size(root_templates_, cnt);
//	printf("Loaded %d models containing %d templates.\n", int(names_.size()), cnt);
//	int last_level = 2;
//	extract_level_templates(last_level, root_templates_, templates_);
//	printf("Lowest level contains %d templates.\n", int(templates_.size()));

//	computeHashFunctions();
}





void BinarizedGradientGrid::detect()
{
	// forget about previous detections
	detections_.detections.clear();
//	cv::Mat gradSummary;
//	imgToGradientSummary2(image_, gradSummary);

	cv::Mat img_gray;
	// compute the gradient summary image
	cv::Mat img = image_;
    if(img.channels() != 1) {
    	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    }
    else {
    	img_gray = img;
    }

	static cv::Mat mag;
	static cv::Mat phase;
	computeGradients(img_gray, mag, phase);

    static cv::Mat binGrad;
    static cv::Mat cleanGrad;
    binarizeGradients(mag, phase, binGrad);
    gradMorphology(binGrad, cleanGrad);

	BinarizedGradientPyramid pyr(cleanGrad,start_level_,levels_); // build two levels starting with level 2

//	for (int i=start_level_;i<start_level_+levels_;++i) {
//		cv::Mat show_gradient(pyr[i].size(), CV_8UC3);
//		showBinarizedGradient(pyr[i], show_gradient);
//		char buf[100];
//		sprintf(buf,"bg_%d.png", i);
//		cv::imwrite(buf, show_gradient);
//
//	}


	printf("Doing detection\n");

	std::vector<BiGGDetection> all_detections;
	clock_t start = clock();
	int index = 0;
	foreach (BiGGTemplateTree& root_templates, all_templates_) {
		std::vector<BiGGDetection> detections;
		std::vector<cv::Point2i> everywhere;
		printf("Using %d templates\n", (int)root_templates.size());
		detect(image_, pyr.start_level+pyr.levels-1, pyr, everywhere, root_templates, detections, template_radius_,  accept_threshold_, accept_threshold_decay_);

		foreach(BiGGDetection& detection, detections) {
			detection.name = names_[index];
			all_detections.push_back(detection);
		}

		// std::copy(detections.begin(), detections.end(), back_inserter(all_detections));
		//	flat_detect(pyr[pyr.start_level], detections, template_radius_, pyr.start_level,  accept_threshold_);
		++index;
	}
	nonMaxSuppress(all_detections, fraction_overlap_);
	double duration = double(clock()-start)/CLOCKS_PER_SEC;
	printf("Detection took: %g\n", duration);


/*    for (size_t i=0;i<detections.size();++i) {

		printf("Dist: %g\n",dist(detections[i].tpl->hist, detections[i].crt_tpl->hist));
    }
*/

    detections_.detections.resize(all_detections.size());
    for (size_t i=0;i<all_detections.size();++i) {

    	cv::Mat& mask = all_detections[i].mask;
    	const BinarizedGradientTemplate& tpl = *all_detections[i].tpl;
    	mask.create(tpl.rect.height, tpl.rect.width, CV_8UC1);
    	mask = cv::Scalar::all(0);
    	for (size_t j=0;j<tpl.mask_list.size();++j) {
    		int x = tpl.mask_list[j]%tpl.rect.width;
    		int y = tpl.mask_list[j]/tpl.rect.width;
    		mask.ptr<uchar>(y)[x] = 255;
    	}
    	//convert BiGGDetection struct  to Detection message
    	convert(all_detections[i], detections_.detections[i]);

    	const BiGGDetection& d = all_detections[i];
    	// fill in indices vector (used for point cloud segmentation)
    	for (size_t j=0;j<tpl.mask_list.size();++j) {
    		int x = tpl.mask_list[j]%tpl.rect.width;
    		int y = tpl.mask_list[j]/tpl.rect.width;
    		int index = (d.roi.y+y)*image_.cols+(d.roi.x+x);
    		detections_.detections[i].mask.indices.indices.push_back(index);
    	}
    }

}

/**
 * Starts training for a new object category model. It may allocate/initialize
 * data structures needed for training a new category.
 * @param name
 */
void BinarizedGradientGrid::startTraining(const std::string& name)
{
	crt_object_ = -1;
	for (size_t i=0;i<names_.size();++i) {
		if (names_[i]==name) {
			crt_object_ = i;
			break;
		}
	}
	if (crt_object_==-1) {
		crt_object_ = names_.size();
		names_.push_back(name);
	}
}


void show_mask(const cv::Mat& image, const cv::Rect& roi, const cv::Mat& mask)
{
	cv::Mat tmp;
	image.copyTo(tmp);
	cv::Mat im_roi = tmp(roi);
//	cv::Mat mask_roi = mask(roi);
	cv::add(im_roi,cv::Scalar(80,0,0),im_roi,mask);



	cv::namedWindow("mask",1);
	cv::imshow("mask", tmp);
	cv::waitKey(0);
}


/**
 * Trains the model on a new data instance.
 * @param name The name of the model
 * @param data Training data instance
 */
void BinarizedGradientGrid::trainInstance(const std::string& name, const TrainingData& data)
{
	printf("BiGG: training instance: %s\n", name.c_str());

	cv::Mat img_gray;
	// compute the gradient summary image
	cv::Mat img = data.image;
    if(img.channels() != 1) {
    	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    }
    else {
    	img_gray = img;
    }

	static cv::Mat mag;
	static cv::Mat phase;
	computeGradients(img_gray, mag, phase);

    static cv::Mat binGrad;
    static cv::Mat cleanGrad;
    binarizeGradients(mag, phase, binGrad);

    gradMorphology(binGrad, cleanGrad);

	BinarizedGradientPyramid pyr(cleanGrad,start_level_,levels_);

//	show_mask(data.image, data.roi, data.mask);

	cv::Mat mask = cv::Mat(img.rows, img.cols, CV_8UC1);
	mask = cv::Scalar::all(0);
	cv::Mat mask_roi = mask(data.roi);
	data.mask.copyTo(mask_roi);

	BinarizedGradientPyramid mask_pyr(mask,start_level_,levels_);

	trainInstance(img, pyr.start_level+pyr.levels-1, pyr, mask_pyr, root_templates_, data.roi, data.mask, template_radius_, accept_threshold_);
}

/**
 * Saves a trained model.
 * @param name model name
 */
void BinarizedGradientGrid::endTraining(const std::string& name)
{
	model_storage_->save(name, getName(), root_templates_);
}


/**
 * Compute magnitude and phase in degrees from single channel image
 * @param img input image
 * @param mag gradient magnitude, returned
 * @param phase gradient phase, returned
 *
 * \pre img.channels()==1 && img.rows>0 && img.cols>0
 */
void BinarizedGradientGrid::computeGradients(const cv::Mat &img, cv::Mat &mag, cv::Mat &phase)
{
	//Find X and Y gradients
	cv::Sobel(img, mag, CV_32F, 1, 0, CV_SCHARR);
	cv::Sobel(img, phase, CV_32F, 0, 1, CV_SCHARR);
	cv::cartToPolar(mag, phase, mag, phase, true); //True => in degrees not radians
}

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
void BinarizedGradientGrid::binarizeGradients(const cv::Mat &mag, const cv::Mat &phase, cv::Mat &binaryGradient)
{
	if((binaryGradient.rows != mag.rows)||(binaryGradient.cols != mag.cols))
	{
		cv::Mat newBinGrad(cv::Size(mag.cols,mag.rows),CV_8UC1);
	    binaryGradient = newBinGrad;
	}
	binaryGradient = cv::Scalar(0,0,0); // set to zero
	cv::MatConstIterator_<float> mit = mag.begin<float>(), mit_end = mag.end<float>();
	cv::MatConstIterator_<float> pit = phase.begin<float>(), pit_end = phase.end<float>();
	cv::MatIterator_<uchar> bit = binaryGradient.begin<uchar>(), bit_end = binaryGradient.end<uchar>();
	for(int i = 0; mit != mit_end; ++mit, ++pit, ++bit, ++i)
	{
	    if(*mit < magnitude_threashold_) continue;
        float angle = *pit;
		if(angle >= 180.0) angle -= 180.0; //We ignore polarity of the angle
		*bit = 1 << int(angle/(180.0/8));
	}
}

/**
 * Filter out noisy gradients via non-max suppression in a 3x3 area.
 * @param binaryGradient input binarized gradient
 * @param cleaned gradient, will be allocated if not already
 */
void BinarizedGradientGrid::gradMorphology(cv::Mat &binaryGradient, cv::Mat &cleanedGradient)
{
    int rows = binaryGradient.rows;
    int cols = binaryGradient.cols;
    //Zero the boarders -- they are unreliable
    uchar *bptrTop = binaryGradient.ptr<uchar>(0);
    uchar *bptrBot = binaryGradient.ptr<uchar>(rows - 1);
    for(int x = 0; x<cols; ++x, ++bptrTop, ++bptrBot){
        *bptrTop = 0;
        *bptrBot = 0;
    }
    for(int y = 1; y < rows - 1; ++y){
        uchar *bptr = binaryGradient.ptr<uchar>(y);
        *bptr = 0;
        bptr += cols - 1;
        *bptr = 0;
    }
    //Now do non-max suppression. NOTE. Each pixel location contains just one orientation at this point
    if((cleanedGradient.rows != rows)||(cleanedGradient.cols != cols))
    {
        binaryGradient.copyTo(cleanedGradient);
    }
    cleanedGradient = cv::Scalar(0,0,0);
    int counts[9]; //9 places since we must also count zeros ... and ignore them.  That means that bit 1 falls into counts[1], 10 into counts[2] etc
    int index[255];  //This is just a table to translate 0001->1, 0010->2, 0100->3, 1000->4 and so on
    index[0] = 0; //Fill out this table Index to increments counts, counts how many times a 1 has been shifted
    index[1] = 1;
    index[2] = 2;
    index[4] = 3;
    index[8] = 4;
    index[16] = 5;
    index[32] = 6;
    index[64] = 7;
    index[128] = 8;
    int sft[9]; //OK, bear with me now. This table will translate from offset back to shift: 1->0001, 2->0010, 3->0100 and so on.
    int mask = 1;
    for(int i = 1; i<9; ++i)
    {
        sft[i] = mask;
        mask = mask << 1;
    }
    for(int y=0; y<rows-2; ++y)
    {
        for(int i = 0; i<9; ++i) //We sweep counts across and use it to determine the orientations present in a local neighborhood
            counts[i] = 0;
        uchar *b0 = binaryGradient.ptr<uchar>(y);  //We're just going to step across 9 positions at a time, pealing off the back pointers
        uchar *b1 = binaryGradient.ptr<uchar>(y+1);
        uchar *b2 = binaryGradient.ptr<uchar>(y+2);
        uchar *c = cleanedGradient.ptr<uchar>(y+1);    //c will point to the center pixel of the "clean" image
        ++c;
        //init the back cols of the count ... remember that index translates the (single) bit position into a count of its shift from right + 1
        counts[index[*b0]] += 1; counts[index[*(b0 + 1)]] += 1;
        counts[index[*b1]] += 1; counts[index[*(b1 + 1)]] += 1;
        counts[index[*b2]] += 1; counts[index[*(b2 + 1)]] += 1;
        uchar *b02 = b0 + 2, *b12 = b1 + 2, *b22 = b2 + 2;
        for(int x=0; x<cols-2; ++x, ++b0, ++b1, ++b2, ++b02, ++b12, ++b22, ++c)
        {
            counts[index[*b02]] += 1; //add the leading edge of counts
            counts[index[*b12]] += 1;
            counts[index[*b22]] += 1;
 //           if(1)//*(b1+1)) //Don't fill in where we found no gradients
 //           {
                //find the max in count
                int maxindx = 1;  //Find the maximum count of real orientation (skip bin zero which means no orientation)
                int maxcnt = counts[1];
                for(int j = 2; j<9; ++j)
                {
                    if(counts[j] > maxcnt)
                    {
                        maxindx = j;
                        maxcnt = counts[j];
                    }
                }
                //see if we have a valid maximum
                if((maxcnt > 1)) //Only record the gradient if it's not a singleton (shot noise)
                {
                    *c = sft[maxindx];
                }
//            }
            //Peal off the back pointers
            counts[index[*b0]] -= 1;
            counts[index[*b1]] -= 1;
            counts[index[*b2]] -= 1;
        }
    }
}


///**
// * This function will OR gradients together within a given square region of diameter d.
// * No checking is done to keep this in bounds ... you must do that
// * @param grad The single channel uchar image of binarized gradients
// * @param xx The center pixel, x coordinate
// * @param yy The center pixel, y coordinate
// * @param d The diameter
// * @return The value of the central byte containing the OR'd gradients of this patch.
// */
//inline unsigned char BinarizedGradientGrid::orGradients(const cv::Mat &grad, int xx, int yy, int d)
//{
//    unsigned char c = 0;
//    for(int y = yy; y<yy+d; ++y)
//    {
//        const uchar *b = grad.ptr<uchar>(y);
//        b += xx;
//        for(int x = xx; x < xx+d; ++x, ++b)
//        {
//            c |= *b;
//        }
//    }
//    return c;
//};
//
//
///**
// * This function will produce a downsampled by logical OR'ing of the binarized gradient image.
// * The reduction factor of the resulting gradSummary image is the "reduction_factor" parameter,
// * the diameter of each summary region is "diameter" parameter
// * @param grad The single channel uchar image of binarized gradients
// * @param gradSummary This is the downsampled (by "reduction_factor") image containing the OR of
// * the binarized gradient in each patch of size "diameter"
// */
//void BinarizedGradientGrid::gradientSummaryImage(const cv::Mat &grad, cv::Mat &gradSummary, int factor)
//{
//    if(grad.empty()) {
//    	ROS_ERROR("gradientSummaryImage(): Invalid gradient image passed in.");
//    	return;
//    }
//
//    if(factor > grad.rows || factor > grad.cols) {
//    	ROS_WARN("gradientSummaryImage(): Wrong reduction_factor parameter.");
//    	factor = std::min(grad.rows, grad.cols);
//    }
//
//    // create gradSummary image if empty
//    if(gradSummary.empty() ||
//    		(gradSummary.rows != grad.rows/factor) ||
//    		(gradSummary.cols != grad.cols/factor)) {
//    	gradSummary.create(grad.rows/factor, grad.cols/factor, CV_8UC1);
//    	gradSummary = cv::Scalar(0,0,0);
//    }
//
//    int rows = grad.rows - factor;
//    int cols = grad.cols - factor; // "-reduction_factor_" to protect against trying to summarize past the edge of the image
//    for(int Y = 0, y = 0; Y<rows; Y += factor, ++y)
//    {
//        uchar *b = gradSummary.ptr<uchar>(y);
//        for(int X = 0; X<cols; X += factor, ++b)
//        {
//            *b = orGradients(grad, X, Y, factor);
//        }
//    }
//}

}

