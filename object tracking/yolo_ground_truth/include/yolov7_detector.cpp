#include "DBSCAN.h"
#include "yolov7_detector.h"

#define COLOR 176
// 检测类别名称，摩托车这里没用上
std::vector<std::string> detect_class_name = {"person", "bicycle", "car", "motorbike"};

using namespace std;
using namespace cv;
using namespace torch::indexing;

//在cluster维度进行排序
bool compc(point a, point b)
{
	return a.cluster < b.cluster;
}

// 将灰度图渲染成彩色图
torch::Tensor decode_color_labels(torch::Tensor labels)
{
	at::Tensor decode_mask = torch::zeros({ 3, labels.sizes()[0], labels.sizes()[1] }).toType(torch::kU8);

	// 为高层次的渲染方法，把图像分为大类
	//int color_map[40][3] = {
	//	{128, 64, 128} ,{128, 64, 128}, {70, 70, 70}, {70, 70, 70}, {70, 70, 70},
	//	{153, 153, 153}, {153, 153, 153}, {153, 153, 153}, {107, 142, 35}, {107, 142, 35},
	//	{70, 130, 180}, {220, 20, 60}, {220, 20, 60},
	//	{0, 0, 142}, {0, 0, 142}, {0, 0, 142}, {0, 0, 142}, {0, 0, 142}, {0, 0, 142}
	//};//存储0-8的RGB值

	//细化把图片染成19种颜色
	int color_map[40][3] = {
		{128, 64 , 128}, {244, 35 , 232}, {70 , 70 , 70 }, {102, 102, 156}, {190, 153, 153},
		{153, 153, 153}, {250, 170, 30 }, {220, 220, 0  }, {107, 142, 35 }, {152, 251, 152},
		{70 , 130, 180}, {220, 20 , 60 }, {255, 0  , 0  }, {0  , 0  , 142},
		{0  , 0  , 70 }, {0  , 60 , 100}, {0  , 80 , 100}, {0  , 0  , 230}, {119, 11 , 32 }
	};


	for (int j = 0; j < 30; j++)
	{
		auto x = (labels == j);
		decode_mask[0] += ((labels == j).toType(torch::kU8)) * color_map[j][0];
		decode_mask[1] += ((labels == j).toType(torch::kU8)) * color_map[j][1];
		decode_mask[2] += ((labels == j).toType(torch::kU8)) * color_map[j][2];
	}
	return decode_mask;
}


//输入instance和车道线输出合成后的图片
torch::Tensor colorplus(torch::Tensor instance, torch::Tensor zebra)
{
	auto h = instance.sizes()[0], w = instance.sizes()[1];
	auto plus = at::zeros_like(instance);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (instance[i][j][0].item<int>() + instance[i][j][1].item<int>() + instance[i][j][2].item<int>() != 0)
			{
				zebra[i][j][0] = instance[i][j][0];
				zebra[i][j][1] = instance[i][j][1];
				zebra[i][j][2] = instance[i][j][2];
			}
		}
	}
	return zebra;
}

//把灰度的标签图转化成可以人眼分辨的彩色图,便于保存观察
cv::Mat convert(torch::Tensor tensor_image_gray)
{
	auto pred = decode_color_labels(tensor_image_gray);

	auto R1 = pred[0].to(at::kCPU);
	auto G1 = pred[1].to(at::kCPU);
	auto B1 = pred[2].to(at::kCPU);
	auto pred1 = pred.permute({ 1, 2, 0 }).toType(torch::kU8);
	int frame_h, frame_w;
	frame_h = pred1.size(0);
	frame_w = pred1.size(1);
	cv::Mat R(cv::Size(frame_w, frame_h), CV_8UC1);
	cv::Mat G(cv::Size(frame_w, frame_h), CV_8UC1);
	cv::Mat B(cv::Size(frame_w, frame_h), CV_8UC1);
	memcpy(R.data, R1.data_ptr(), R1.numel() * sizeof(unsigned char));
	memcpy(G.data, G1.data_ptr(), G1.numel() * sizeof(unsigned char));
	memcpy(B.data, B1.data_ptr(), B1.numel() * sizeof(unsigned char));
	Mat imgs[3] = { R,G,B };
	Mat result;
	cv::merge(imgs, 3, result);
	return result;
}


// 对一个string进行分割，返回分割后的vector给res这个参数，用法Stringsplit(s1, '\\', strList);
//void Stringsplit(string str, const const char split, vector<string>& res)
//{
//	istringstream iss(str);	// 输入流
//	string token;			// 接收缓冲区
//	while (getline(iss, token, split))	// 以split为分隔符
//	{
//		res.push_back(token);
//	}
//}

//(待实现) 给定一个图像和车的bottom和roof的各四个点，画出三维框的效果
void draw_3d(torch::Tensor, int* bottom, int* roof)
{

	return;
}


//(已实现)
//directions方向数组(从文件中读取,如[40, -30],一般限速为整数)
//x_car和y_car,车的中心位置
//speed车速,在本程序中采用随机生成大小和方向的方式
//img为含车道线的灰度图,其中本程序中只利用到了中间的车道分割线
int orientation_test(int* directions, int linenum, float speed, Bbox box, at::Tensor img)
{
	int width = img.sizes()[0];
	int height = img.sizes()[1];
	int rnum = 0;
	int lnum = 0;

	//cout << "box:" << box.x  << "   " << box.y  << "   " << width << endl;

	//cout << img[width - 1][height - 1] << endl; 

	for (int i = box.y; i > 0; i--)
	{
		if (img[i - 1][box.x].item<int>() == 176 && img[i][box.x].item<int>() != 176)
		{
			lnum += 1;
		}
	}

	for (int i = box.y + 1; i < width; i++)
	{
		if (img[i - 1][box.x].item<int>() == 176 && img[i][box.x].item<int>() != 176)
		{
			rnum += 1;
		}
	}
	if (lnum + rnum != linenum)
		return 0;

	int chaosu = (abs(directions[lnum]) - abs(speed)) < 0;
	int nixing = (speed * directions[lnum] > 0);
	return 10 * chaosu + nixing;

}

// car 则令flag = 255, bicycle 则令 flag = 128
/*
int yuejie_judge(int ymax_car, int ymin_car, int xmax_car, at::Tensor tensor_image, int flag = 255)
{

	int num_inter = 50;
	int weigui_point_car = 0;
	//point interval((bottom_right_car.x - bottom_left_car.x) * 1.0 / num_inter, (bottom_right_car.y - bottom_left_car.y) * 1.0 / num_inter, 0);
	point interval(0, (ymax_car - ymin_car) * 1.0 / num_inter, 0);

	for (int j = 0; j <= num_inter; j++)
	{
		int x = xmax_car;
		int y = ymin_car + j * interval.y;
		//车运行在自行车道路上或者自行车运动在车道路上则判断违规
		if (tensor_image[x][y].item<int>() == flag)//128为车行道路，255为自行车道路
		{
			weigui_point_car += 1;
		}
	}
	return weigui_point_car;
}*/

// car 则令flag = 255, bicycle 则令 flag = 128
int yuejie_judge(int xmin_car, int xmax_car, int ymax_car, at::Tensor tensor_image, int flag = 255)
{//这里的x应该是纵坐标  ymax_car指横坐标

	int num_inter = 50;
	int weigui_point_car = 0;
	//point interval((bottom_right_car.x - bottom_left_car.x) * 1.0 / num_inter, (bottom_right_car.y - bottom_left_car.y) * 1.0 / num_inter, 0);
	point interval((xmax_car - xmin_car) * 1.0 / num_inter, 0, 0);

	for (int j = 0; j <= num_inter; j++)
	{
		int x = xmin_car + j * interval.x;
		x = min((long long)max(0, x), (long long)tensor_image.sizes()[1] - 1);
		int y = ymax_car;
		y = min((long long)max(0, y), (long long)tensor_image.sizes()[0] - 1);
		//车运行在自行车道路上或者自行车运动在车道路上则判断违规
		//cout << "y:" << y << "   " << "x:" << x << endl;
		if (tensor_image[y][x].item<int>() == flag)//128为车行道路，255为自行车道路
		{
			weigui_point_car += 1;
		}
		//tensor_image[x][y] = 200;
	}
	//Mat mat=tensor2Mat(tensor_image);
	//cv::imwrite("./test/"+to_string(xmin_car)+".png", mat);
	return weigui_point_car;
}







//常用的类型转换
int string2int(string SS)
{
	int num = 0;
	istringstream ss(SS);
	ss >> num;
	return num;
}

cv::Mat tensor2Mat(torch::Tensor& i_tensor) {
	int height = i_tensor.size(0), width = i_tensor.size(1);
	//i_tensor = i_tensor.to(torch::kF32);
	i_tensor = i_tensor.to(torch::kCPU);
	cv::Mat o_Mat(cv::Size(width, height), CV_8U, i_tensor.data_ptr());
	return o_Mat;
}

at::Tensor Mat2tensor(Mat input_img)//只支持单个图片的Mat2tensor
{
	torch::Tensor tensor_image = torch::from_blob(input_img.data,
		{ input_img.rows, input_img.cols }, torch::kByte);
	return tensor_image;
}

float iou(Bbox box1, Bbox box2)
{

	int min_x = max(box1.x - box1.h / 2, box2.x - box2.h / 2);  // 找出左上角坐标哪个大
	int max_x = min(box1.x + box1.h / 2, box2.x + box2.h / 2);  // 找出右上角坐标哪个小

	int min_y = max(box1.y - box1.w / 2, box2.y - box2.w / 2);
	int max_y = min(box1.y + box1.w / 2, box2.y + box2.w / 2);

	if (min_x >= max_x || min_y >= max_y) // 如果没有重叠
		return 0;
	float over_area = (max_x - min_x) * (max_y - min_y);  // 计算重叠面积
	float area_a = box1.h * box1.w;
	float area_b = box2.h * box2.w;
	float iou = over_area / (area_a + area_b - over_area);
	return iou;
}

bool cmp(Bbox box1, Bbox box2)
{
	return (box1.score > box2.score);
}
/*
(1) 获取当前目标类别下所有bbx的信息
(2) 将bbx按照confidence从高到低排序,并记录当前confidence最大的bbx
(3) 计算最大confidence对应的bbx与剩下所有的bbx的IOU,移除所有大于IOU阈值的bbx
(4) 对剩下的bbx，循环执行(2)和(3)直到所有的bbx均满足要求（即不能再移除bbx）
*/
vector<Bbox> nms(vector<Bbox>& vec_boxs, float threshold) //为iou_thres
{
	vector<Bbox>  res;
	while (vec_boxs.size() > 0)
	{
		res.push_back(vec_boxs[0]);
		int index = 1;
		while (index < vec_boxs.size())
		{
			float iou_value = iou(vec_boxs[0], vec_boxs[index]);
			if (iou_value > threshold)
				vec_boxs.erase(vec_boxs.begin() + index);
			else
				index++;
		}
		//sort(vec_boxs.begin(), vec_boxs.end(), cmp);
		//res.push_back(vec_boxs[0]);
		//for (int i = 0; i < vec_boxs.size() - 1; i++)
		//{
		//	float iou_value = iou(vec_boxs[0], vec_boxs[i + 1]);
		//	cout << endl <<"iou:" << iou_value ;
		//	if (iou_value > threshold)
		//	{
		//		
		//		vec_boxs.erase(vec_boxs.begin() + i + 1);
		//	}
		//}
		vec_boxs.erase(vec_boxs.begin());  // res 已经保存，所以可以将最大的删除了

	}
	return res;
}

//zebra_gray是车道(线!)
//zebra_area是标出了行驶区域
//auto size = cv::Size(512, 256);//这个尺寸是模型输入的最佳尺寸
auto size = cv::Size(512, 512);

int label_car = 1;
int label_bicycle = 1;
int threshold_car = 100;
int threshold_bicycle = 30;
float conf_thres = 0.25;
float iou_thres = 0.45;
int expand = 20;

int direction[10] = { 0 }, linenum = 0;
torch::Tensor tensor_image;
at::Tensor tensor_zebra;
torch::DeviceType device_type;

// init the model according to the model path
torch::jit::script::Module init_network(std::string model_pb, std::string config_path)
{
	auto zebra_gray_path = config_path + "/label.png";
	auto zebra_gray = imread(zebra_gray_path, 0);
	auto zebra_area_path = config_path + "/area.png";
	auto zebra_area = imread(zebra_area_path, 0);
	auto color_zebra = imread(config_path + "/color_zebra.png");

	// 把区域和车道分割线的语义图转化成512*256方便判断
	cv::resize(zebra_gray, zebra_gray, size, cv::INTER_NEAREST);
	tensor_zebra = Mat2tensor(zebra_gray);
	cv::resize(zebra_area, zebra_area, size, cv::INTER_NEAREST);
	tensor_image = Mat2tensor(zebra_area);//把车道所在区域转化成tensor
	cout << "tensor_image.shape:" << tensor_image.sizes() << endl;

	ifstream infile(config_path + "/speed.txt");
	string buf;
	while (getline(infile, buf))//逐行读入文档中的数据
		direction[linenum++] = string2int(buf);

	// init network
	device_type = at::kCUDA;
	torch::Device device(device_type);

	auto model = torch::jit::load(model_pb);
	model.to(device_type);

	//--------------------------------------------预热
	for (int i = 0; i < 2; i++)
	{
		// 初始化随机数生成器
    	cv::RNG rng(cv::getTickCount());
		// 创建一个随机图像
		cv::Mat randomImage(size, CV_8UC3);

		for (int y = 0; y < size.height; ++y) {
			for (int x = 0; x < size.width; ++x) {
				// 生成随机颜色
				cv::Vec3b color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				randomImage.at<cv::Vec3b>(y, x) = color;
			}
		}

		torch::Tensor tensor_image1 = torch::from_blob(randomImage.data,
			{ randomImage.rows, randomImage.cols,3 }, torch::kByte);
		tensor_image1 = tensor_image1.permute({ 2,0,1 }).toType(torch::kFloat).unsqueeze(0).toType(torch::kFloat).to(device_type);
		model.to(at::kCUDA);
		tensor_image1.to(at::kCUDA);
		tensor_image1 = tensor_image1 / 255.0;
		cout << tensor_image1.sizes() << endl;

		clock_t sss = clock();
		auto prediction = model.forward({ tensor_image1 });
	}

	return model;
}

/*********************************************************************************************************/
/* 函 数:     detect_bbox_from_image													                                           */
/* 描 述:     检测图像中的目标，输出边界框                                                               */
/* 输 入:     module: 神经网络模型, frame: 当前帧图像                                                    */
/* 输 出:     长度为3的列表：                                                                            */
/*                下标0: 行人检测结果, 1: 非机动车检测结果, 2: 机动车检测结果                            */
/*            每一类目标检测结果为一个Bbox列表，其中每一个Bbox元素包含属性x, y, w, h ,score, type_num    */
/*                x, y: 目标中点在原图像中的位置, x为列, y为行                                           */
/*                w, h: 目标在原图像中的宽和高, w为列宽, h为行高                                         */
/*                score: 目标概率                                                                        */
/*                type_num: 目标类型(1: 行人, 2: 非机动车, 3: 机动车)                                    */
/*********************************************************************************************************/
std::vector<std::vector<Bbox> > detect_bbox_from_image(torch::jit::script::Module &module, cv::Mat &frame)
{
	int rows = frame.rows;
	int cols = frame.cols;
	// cout << "rows: " <<rows<<endl;//720
	// cout << "cols: " <<cols<<endl;//1280
	cv::Mat image_transfomed;
	cv::resize(frame, image_transfomed, size, cv::INTER_NEAREST);
	// cout << "frame size: " << frame.size().width<<endl;
	
	torch::Tensor tensor_image1 = torch::from_blob(image_transfomed.data,
		{ image_transfomed.rows, image_transfomed.cols,3 }, torch::kByte);
	tensor_image1 = tensor_image1.permute({ 2,0,1 }).toType(torch::kFloat).unsqueeze(0).toType(torch::kFloat).to(device_type);


	module.to(at::kCUDA);
	tensor_image1.to(at::kCUDA);
	tensor_image1 = tensor_image1 / 255.0;

	auto prediction = module.forward({ tensor_image1 });

	torch::Tensor output = prediction.toTensor();			//(1,25200,85)
	int nc = output.sizes()[2] - 5;							//80
	auto xc = output.index({ 0, "...", 4 });				//(25200)
	int num_box = xc.sizes()[0];							//25200
	auto vec = torch::where(xc > conf_thres)[0];			//(53)
	int num = vec.sizes()[0];								//53

	std::vector<Bbox> box_person_1;
	std::vector<Bbox> box_bicycle_2;
	std::vector<Bbox> box_car_3;
	//下面这个定义没用上
	// std::vector<Bbox> box_motorbike_4;

	//对当前图片的所有框进行分类放进不同类别的vector里
	for (int i = 0; i < num; i++)
	{
		at::Tensor real_vec = output[0][vec[i]];//当前待处理的向量
		auto now_vec = real_vec.index({ "...", Slice(5) });
		auto maxValue = (torch::max)(now_vec, 0);
		int index = std::get<1>(maxValue).item<int>() + 1;

		Bbox box_to_push;
		box_to_push.x = real_vec[0].item<int>();
		box_to_push.y = real_vec[1].item<int>();
		box_to_push.h = real_vec[2].item<int>();
		box_to_push.w = real_vec[3].item<int>();
		box_to_push.score = real_vec[4].item<float>();
		box_to_push.type_number = index;
   
    	// 调整比例,bbox存放真实比例
    	box_to_push.x *= (double) cols / size.width;// y=y*1280/512
		box_to_push.y *= (double) rows / size.height;// x=x*720/512
		box_to_push.h *= (double) cols / size.width;
		box_to_push.w *= (double) rows / size.height;
    	swap(box_to_push.h, box_to_push.w);

		if (index == 1)
		{
			box_person_1.push_back(box_to_push);
		}
		else if (index == 2)
		{
			box_bicycle_2.push_back(box_to_push);
		}
		else if (index == 3)
		{
			box_car_3.push_back(box_to_push);
		}
	}
	
	std::vector<std::vector<Bbox> > ret;
	ret.push_back(nms(box_person_1, iou_thres));
	ret.push_back(nms(box_bicycle_2, iou_thres));
	ret.push_back(nms(box_car_3, iou_thres));

	return ret;
}

cv::Mat draw_bbox_on_frame(cv::Mat &frame, vector<vector<Bbox> > &result)
{
	cv::Mat ret = frame;//1280*720
	cv::Size frame_size = frame.size();
	// cls是类别数
	for(int cls = 0; cls < result.size(); ++cls)
	{
		// 读取一类的所有box
		auto &box = result[cls];
		for (int i = 0; i < box.size(); i++)
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
			// 输出计算的坐标值
			// cout<<"坐标值"<<endl;
			// cout << "yolo 计算结果：";
			// std::cout<< x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
			// 左上、右下，绘制红色边框
			cv::rectangle(ret, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);

		}
	}

	return ret;
}
