#include <iostream>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "yolov7_detector.h"

using namespace std;
using namespace cv;

//-----------------------------------------------------------------------
void save_target_2dbbox_as_txt(std::vector<std::vector<Bbox>>& result, std::string& outputTXT_path)
{
    std::ofstream f(outputTXT_path);
    if (!f.is_open()) {
        // 处理文件无法打开的情况
        return;
    }
    int num = 0;
    // cls是类别数，cls：1-3，人、自行车、汽车
    for (int cls = 0; cls < result.size(); ++cls) {
        // 读取一类的所有box
        auto& box = result[cls];
        for (int i = 0; i < box.size(); i++) // 遍历这一类
        {
            int x = box[i].x;
            int y = box[i].y;
            int h = box[i].h;
            int w = box[i].w;
            // 算主对角线两点坐标
            int x1 = x - w / 2;
            int y1 = y - h / 2;
            int x2 = x + w / 2;
            int y2 = y + h / 2;
            f << "0 "<< x1 << " " << y1 <<" " << x2 << " " << y2 << "\n";  
                       
            num++;
        }
    }
    f.close();
}



// Function to calculate the 3D bounding boxes
int main(int argc, char* argv[]) {
    // Check if the input path is provided
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " path/to/img_or_video" << std::endl;
        return -1;
    }
    // Root path
    std::string rootPath = "../"; // Default in the ../ file
    // Input path for img or video
    std::string inputPath = argv[1]; // Get the input path from the command-line argument
    //fing file name and extension
    size_t lastDotPos = inputPath.find_last_of('.');
    size_t lastSlashPos = inputPath.find_last_of('/');
    std::string file_Name = inputPath.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
    std::string extension = inputPath.substr(lastDotPos);

    //初始化yolov7，模型与配置
    // ================= Init Yolov7 detector ==========================
    auto model = init_network("../model/traced_model.pt", "../config");
    // Open the video file
    cv::VideoCapture videoCapture(inputPath);
    // Check if the video file was opened successfully
    if (!videoCapture.isOpened())
    {
        std::cerr << "Error: Unable to open the video file." << std::endl;
        return -1;
    }
    // Get the frame dimensions of the video
    int frameWidth =  static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    // Output video size
    std::cout << "video_height: " << frameHeight << std::endl;
    std::cout << "video_width: " << frameWidth << std::endl;
    // Create a VideoWriter object to save the processed video        
    std::string outputVideoPath = rootPath + "output/" + file_Name + "_result" + extension;
    
    cv::VideoWriter videoWriter;
    int fourcc = static_cast<int>(videoCapture.get(cv::CAP_PROP_FOURCC));
    double fps = videoCapture.get(cv::CAP_PROP_FPS);        
    std::cout<<"video_fps: "<<fps<<endl;
    cv::Size outputSize(frameWidth, frameHeight);
    videoWriter.open(outputVideoPath, fourcc, fps, outputSize);    

    if (!videoWriter.isOpened())
        {
            std::cerr << "Error: Unable to create the VideoWriter." << std::endl;
            return -1;
        }
    else cout<<"video open successfully"<<endl;
    // Process each frame in the video
    int frameCount = 0;
    
    
    while(true)
    {    
        cv::Mat frame;
        videoCapture >> frame; // Read the next frame
        // Check if the frame was read successfully
        if (frame.empty())
            break;
        // Increase the frame count
        frameCount++;
//=====================================开始==================================================
        // Yolov7 object detection
        // 读一帧camera，检测三类，存于bboxes
        std::vector<std::vector<Bbox> > bboxes = detect_bbox_from_image(model, frame);
            
        std::cout << "new frame:" << std::endl;
        // 输出bboxes前三个类的信息，类别名+数量。人、自行车、汽车。
        for(int i = 0; i < 3; ++i)
        {
            std::cout << detect_class_name[i] << " : " << bboxes[i].size() << std::endl;
        }   
        std::string output_txt_path = rootPath + "outputTXTyolo/" + file_Name +"yolo_"+std::to_string(frameCount) +".txt";
        save_target_2dbbox_as_txt(bboxes,output_txt_path);

        // draw bbox on the image
        // 遍历每个类别再遍历每个框，画红框，
        cv::Mat showImg = draw_bbox_on_frame(frame, bboxes);
        cv::imshow("image", showImg);
        

            
            
            // cv::Mat show_;
            // 改变大小
            // cv::resize(showImg, show_, cv::Size(640, 480), cv::INTER_NEAREST);
            // cv::imshow("img", show_);
            
        
        
//================================下面不用管==================================================
        //控制按键。
        int key=cv::waitKey(1) & 0xff;
        if(key=='q')
            break;
        // cv::waitKey(0);
    }
    



    // Release the VideoWriter and VideoCapture objects
    videoWriter.release();
    videoCapture.release();
    // cap.release();
    cv::destroyAllWindows();
    sleep(0.5);
    return 0;
}
