
#include "feature_detector.h"
#include <ros/ros.h>
#include <string.h>
#include "opencv2/nonfree/features2d.hpp"
#include "vro_parameter.h"

using namespace cv;

// extern StatefulFeatureDetector* adjustedGridWrapper(cv::Ptr<DetectorAdjuster> detadj);
// extern StatefulFeatureDetector* adjusterWrapper(cv::Ptr<DetectorAdjuster> detadj, int min, int max);

FeatureDetector* create(const std::string& detectorName)
{ 
  DetectorAdjuster* detAdj = NULL;

  ROS_WARN_STREAM("Using " << detectorName << " keypoint detector.");
  if( detectorName == "SIFTGPU" ) {
    return NULL;// Will not be used
  } 
  else if(detectorName == "FAST") {
    detAdj = new DetectorAdjuster("FAST", 20);
  }
  else if(detectorName == "SURF" || detectorName == "SURF128") {
    detAdj = new DetectorAdjuster("SURF", 200);
  }
  else if(detectorName == "SIFT") {
    detAdj = new DetectorAdjuster("SIFT", 0.04, 0.0001);
  }
  else if(detectorName == "ORB") {
    detAdj = new DetectorAdjuster("AORB", 20);
  } 
  else {
    ROS_ERROR("Unsupported Keypoint Detector. Using GridDynamicORB as fallback.");
    return create("GridDynamicORB");
  }
  assert(detAdj != NULL);

  /*ParameterServer* params = ParameterServer::instance();
  // bool gridWrap = (params->get<int>("detector_grid_resolution") > 1);
  // bool dynaWrap = (params->get<int>("adjuster_max_iterations") > 0);

  if(dynaWrap && gridWrap){
    return adjustedGridWrapper(detAdj);
  }
  else if(dynaWrap){
    int min = params->get<int>("max_keypoints");
    int max = min * 1.5; //params->get<int>("max_keypoints");
    return adjusterWrapper(detAdj, min, max);
  }
  else return detAdj;*/
  return detAdj; 
}


FeatureDetector* myCreateDetector( const std::string& detectorName)
{
  DetectorAdjuster* detAdj = NULL;

  ROS_WARN_STREAM(" feature_detector.cpp: Using " << detectorName << " keypoint detector.");
  if( detectorName == "SIFTGPU" ) {
    return NULL;// Will not be used
  } 
  else if(detectorName == "FAST") {
    // detAdj = new DetectorAdjuster("FAST", 20);
    detAdj = new CDetectorWrapper("FAST", 20);
  }
  else if(detectorName == "SURF" || detectorName == "SURF128") {
    // detAdj = new DetectorAdjuster("SURF", 200);
    detAdj = new CDetectorWrapper("SURF", 200);
  }
  else if(detectorName == "SIFT") 
  {
    // ros::NodeHandle nh("~"); 
    int returnedFeatures, nOctaveLayers; 
    double threshold, inv_OctaveLayers; 
    // nh.param<int>("sift_num_features", returnedFeatures, 0); 
    // nh.param<int>("sift_octave_layers", nOctaveLayers, 5); 
    // nh.param<double>("sift_contrast_threshold", threshold, 0.04);
    CParams* pP = CParams::Instance();
    returnedFeatures = pP->m_sift_num_features; 
    nOctaveLayers = pP->m_sift_octave_layers; 
    threshold = pP->m_sift_contrast_threshold; 
    inv_OctaveLayers = 1./(double)(nOctaveLayers*2);

    // detAdj = new DetectorAdjuster("SIFT", 0.04, 0.0001);
    detAdj = new CDetectorWrapper("SIFT", returnedFeatures, nOctaveLayers, threshold * inv_OctaveLayers); // according to the sift  matlab version
  }
  else if(detectorName == "ORB") {
    // detAdj = new DetectorAdjuster("AORB", 20);
    detAdj = new CDetectorWrapper("AORB", 20);
  } 
  else {
    ROS_ERROR("Unsupported Keypoint Detector. Using GridDynamicORB as fallback.");
    // return createDetector("GridDynamicORB");
    return create("GridDynamicORB");
  }
  assert(detAdj != NULL);

  /*
  ParameterServer* params = ParameterServer::instance();
  bool gridWrap = (params->get<int>("detector_grid_resolution") > 1);
  bool dynaWrap = (params->get<int>("adjuster_max_iterations") > 0);

  if(dynaWrap && gridWrap){
    return adjustedGridWrapper(detAdj);
  }
  else if(dynaWrap){
    int min = params->get<int>("max_keypoints");
    int max = min * 1.5; //params->get<int>("max_keypoints");
    return adjusterWrapper(detAdj, min, max);
  }
  else return detAdj;*/
  return detAdj; 
}

DescriptorExtractor* createDescriptorExtractor( const string& descriptorType ) 
{
  DescriptorExtractor* extractor = 0;
  if( !descriptorType.compare( "SIFT" ) ) {
    extractor = new SiftDescriptorExtractor();/*( double magnification=SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION(), bool isNormalize=true, bool recalculateAngles=true, int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES, int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS, int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE, int angleMode=SIFT::CommonParams::FIRST_ANGLE )*/
  }
  else if( !descriptorType.compare( "BRIEF" ) ) {
    extractor = new BriefDescriptorExtractor();
  }
  else if( !descriptorType.compare( "BRISK" ) ) {
    extractor = new cv::BRISK();/*brisk default: (int thresh=30, int octaves=3, float patternScale=1.0f)*/
  }
  else if( !descriptorType.compare( "FREAK" ) ) {
    extractor = new cv::FREAK();
  }
  else if( !descriptorType.compare( "SURF" ) ) {
    extractor = new SurfDescriptorExtractor();/*( int nOctaves=4, int nOctaveLayers=2, bool extended=false )*/
  }
  else if( !descriptorType.compare( "SURF128" ) ) {
    extractor = new SurfDescriptorExtractor();/*( int nOctaves=4, int nOctaveLayers=2, bool extended=false )*/
    extractor->set("extended", 1);
  }
#if CV_MAJOR_VERSION > 2 || CV_MINOR_VERSION >= 3
  else if( !descriptorType.compare( "ORB" ) ) {
    extractor = new OrbDescriptorExtractor();
  }
#endif
  else if( !descriptorType.compare( "SIFTGPU" ) ) {
    ROS_DEBUG("%s is to be used as extractor, creating SURF descriptor extractor as fallback.", descriptorType.c_str());
    extractor = new SurfDescriptorExtractor();/*( int nOctaves=4, int nOctaveLayers=2, bool extended=false )*/
  }
  else {
    ROS_ERROR("No valid descriptor-matcher-type given: %s. Using SURF", descriptorType.c_str());
    extractor = createDescriptorExtractor("SURF");
  }
  ROS_ERROR_COND(extractor == 0, "No extractor could be created");
  return extractor;
}



CDetectorWrapper::CDetectorWrapper(const char* detector_name, double initial_thresh, double min_thresh, double max_thresh, double increase_factor, double decrease_factor):
  DetectorAdjuster(detector_name, initial_thresh, min_thresh, max_thresh, increase_factor, decrease_factor),
  nReturnedFeatures_(0), 
  nOctaveLayers_(3),
  contrastThreshold_(0.04)
{
}

CDetectorWrapper::CDetectorWrapper(const char* detector_name, int nReturnedFeatures, int nOctaveLayers, double contrastThreshold):
  DetectorAdjuster(detector_name), 
  nReturnedFeatures_(nReturnedFeatures),
  nOctaveLayers_(nOctaveLayers),
  contrastThreshold_(contrastThreshold)
{
}

CDetectorWrapper::~CDetectorWrapper(){}

void CDetectorWrapper::detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask ) const
{
  Ptr<FeatureDetector> detector; 
  if(strcmp(detector_name_, "SIFT") == 0)
  {
    detector = new SiftFeatureDetector( nReturnedFeatures_/*max_features*/, nOctaveLayers_ /*default lvls/octave*/, contrastThreshold_);
    detector->detect(image, keypoints, mask);
    return ;
  }

  // as rgbdslam do 
  DetectorAdjuster::detectImpl(image, keypoints, mask);
}


