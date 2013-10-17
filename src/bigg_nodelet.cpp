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
\author Marius Muja
**/

#include "bigg_detector/bigg_nodelet.h"
#include "bigg_detector/bigg.h"
#include "rein/io/db_model_storage.h"
#include "rein/io/fs_model_storage.h"

#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>


namespace bigg_detector {

/**
 * Sets up dynamic reconfigure callback.
 * @param nh
 */
void BiGGNodelet::initConfigureService(ros::NodeHandle& nh)
{
	static dynamic_reconfigure::Server<BiGGConfig> config_srv(nh);
	dynamic_reconfigure::Server<BiGGConfig>::CallbackType f = boost::bind(&BiGGNodelet::configCallback, this, _1, _2);
	config_srv.setCallback(f);
}


/**
 * Initializes the nodelet
 */
void BiGGNodelet::childInit(ros::NodeHandle& nh)
{
	std::string db_type;
	nh.param<std::string>("db_type", db_type, "postgresql");
	std::string connection_string;
	if (!nh.getParam("connection_string", connection_string)) {
		ROS_ERROR("Parameter 'connection_string' is missing");
	}

	// instantiate the detector
	ModelStoragePtr ms_ptr;
	if (db_type=="filesystem") {
		ms_ptr = boost::make_shared<FilesystemModelStorage>(connection_string);
	}
	else {
		ms_ptr = boost::make_shared<DatabaseModelStorage>(db_type,connection_string);
	}
	boost::shared_ptr<BinarizedGradientGrid> detector = boost::make_shared<bigg_detector::BinarizedGradientGrid>(ms_ptr);

	bool do_training;
	nh.param("do_training", do_training, false);
	if (do_training) {
		TrainablePtr trainabale_ptr = boost::static_pointer_cast<Trainable>(detector);
		trainer_server_ = boost::make_shared<TrainerServer>(boost::ref(nh), boost::ref(trainabale_ptr));
	}
	detector_ = detector;

	nh.getParam("models",models_);
	loadModels(models_);

	int start_level;
	nh.param("start_level",start_level,2);
	int levels;
	nh.param("levels",levels,2);
	detector->setStartLevel(start_level);
	detector->setLevels(levels);

}

/**
 * Callback for the configuration parameters. Automatically called when
 * a parameter is changed.
 * @param config
 * @param level
 */
void BiGGNodelet::configCallback(BiGGConfig &config, uint32_t level)
{
	float eps = 1e-5;

	if (models_ != config.models) {
		models_ = config.models;
		loadModels(models_);
	}

	boost::shared_ptr<bigg_detector::BinarizedGradientGrid> detector =
			boost::static_pointer_cast<bigg_detector::BinarizedGradientGrid>(detector_);

	int template_radius = detector->getTemplateRadius();
	if (template_radius!=config.template_radius) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the template_radius parameter to: %d.",
				detector->getName ().c_str (), config.template_radius);
		detector->setTemplateRadius(config.template_radius);
	}

	float accept_threshold = detector->getAcceptThreshold();
	if (fabs(accept_threshold - config.accept_threshold)>eps) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the accept_threshold parameter to: %g.",
				detector->getName ().c_str (), config.accept_threshold);
		detector->setAcceptThreshold(config.accept_threshold);
	}

	float accept_threshold_decay = detector->getAcceptThresholdDecay();
	if (fabs(accept_threshold_decay - config.accept_threshold_decay)>eps) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the accept_threshold_decay parameter to: %g.",
				detector->getName ().c_str (), config.accept_threshold_decay);
		detector->setAcceptThresholdDecay(config.accept_threshold_decay);
	}

	float magnitude_threshold = detector->getMagnitudeThreshold();
	if (fabs(magnitude_threshold - config.magnitude_threshold)>eps) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the magnitude_threashold parameter to: %g.",
				detector->getName ().c_str (), config.magnitude_threshold);
		detector->setMagnitudeThreshold(config.magnitude_threshold);
	}

	float fraction_overlap = detector->getFractionOverlap();
	if (fabs(fraction_overlap - config.fraction_overlap)>eps) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the fraction_overlap parameter to: %g.",
				detector->getName ().c_str (), config.fraction_overlap);
		detector->setFractionOverlap(config.fraction_overlap);
	}
	
	int match_method = detector->getMatchMethod();
	if (match_method != config.match_method) {
		NODELET_INFO ("[bigg_detector::%s::config_callback] Setting the match_method parameter to: %d.",
				detector->getName ().c_str (), config.match_method);
		detector->setMatchMethod(config.match_method);
	}

}

}

/**
 * Pluginlib declaration. This is needed for the nodelet to be dynamically loaded/unloaded
 */
typedef bigg_detector::BiGGNodelet BiGGNodelet;
PLUGINLIB_DECLARE_CLASS (bigg_detector, BiGGNodelet, BiGGNodelet, nodelet::Nodelet);
