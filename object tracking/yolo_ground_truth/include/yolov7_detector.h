#ifndef YOLOV7_DETECTOR_H
#define YOLOV7_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>
#include <glob.h>
#include <libgen.h>
#include <vector>

typedef struct Bbox
{
	int x;
	int y;
	int h;
	int w;
	float score;
	int type_number;
}Bbox;

extern std::vector<std::string> detect_class_name;

torch::jit::script::Module init_network(std::string model_pb, std::string config_path);

std::vector<std::vector<Bbox> > detect_bbox_from_image(torch::jit::script::Module &module, cv::Mat &frame);

cv::Mat draw_bbox_on_frame(cv::Mat &frame, std::vector<std::vector<Bbox> > &result);

#endif