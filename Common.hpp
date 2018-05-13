#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
const int IMAGE_DOWNSAMPLE = 4;							// 对图像进行降采样的比率， 为了加快处理速度
const double FOCAL_LENGTH = 5616/ IMAGE_DOWNSAMPLE;		// 根据相似三角形原理， 焦距也会跟着图片的缩小而缩小
const double MAX_3D_REPROJECTION_ERROR = 3.0;			// 3D点重投影误差
const double MAX_2D_REPROJECTION_ERROR = 3.0;			// 2D点重投影误差
const double MIN_PARALLAX_DEGREE = 2.0;					// 三角测量角度阈值


using namespace std;
using namespace cv;

typedef int image_t;
typedef int point2d_t;
typedef int point3d_t;
struct Frame
{
	image_t image_id;
	string image_name;
	Mat image;

	vector<Point2d> image_pts;
	Mat image_pts_desc;

	Mat R;
	Mat t;

	vector<point2d_t> pt_3ds_indicator;


	Frame(){}
	Frame(image_t image_id) : image_id(image_id){};
	Frame(image_t image_id, string image_name) : image_id(image_id), image_name(image_name){};
};


struct TrackElement
{
	TrackElement(){}
	TrackElement(image_t image_id, point2d_t point2d_idx) : image_id(image_id), point2d_idx(point2d_idx){}
	image_t image_id;
	point2d_t point2d_idx;
};


struct Track
{
	Point3d p3d;
	vector<TrackElement> elements;
	bool aviliable = true;
	
};


struct SFM
{

	Mat K;
	vector<Frame> frames;
	vector<vector<DMatch>> matches;
	vector<Track> tracks;
	vector<bool> is_new;

	size_t Length()
	{
		size_t inlier_num = 0;
		for(size_t i = 0; i < tracks.size(); ++i)
		{
			inlier_num += tracks[i].aviliable;
		}
		return inlier_num;
	}

	void Update()
	{
		
		for(auto& track : tracks)
		{
			if(track.aviliable)
				continue;

			for(auto& element : track.elements)
			{
				Frame& frame = frames[element.image_id];
				frame.pt_3ds_indicator[element.point2d_idx] = -1;
			}
		}
	}
};

const int IMAGE_USE = 23;
const std::string IMAGE_DIR = "/home/anton/Desktop/templeRing/";
const std::vector<std::string> IMAGES = {
"IMG_2331.JPG",
"IMG_2332.JPG",
"IMG_2333.JPG",
"IMG_2334.JPG",
"IMG_2335.JPG",
"IMG_2336.JPG",
"IMG_2337.JPG",
"IMG_2338.JPG",
"IMG_2339.JPG",
"IMG_2340.JPG",
"IMG_2341.JPG",
"IMG_2342.JPG",
"IMG_2343.JPG",
"IMG_2344.JPG",
"IMG_2345.JPG",
"IMG_2346.JPG",
"IMG_2347.JPG",
"IMG_2348.JPG",
"IMG_2349.JPG",
"IMG_2350.JPG",
"IMG_2351.JPG",
"IMG_2352.JPG",
"IMG_2353.JPG",
"IMG_2354.JPG",
/* "IMG_2355.JPG", */
"IMG_2356.JPG",
"IMG_2357.JPG",
"IMG_2358.JPG",
"IMG_2359.JPG",
"IMG_2360.JPG",
"IMG_2361.JPG",
"IMG_2362.JPG",
"IMG_2363.JPG",
"IMG_2364.JPG",
"IMG_2365.JPG",
"IMG_2366.JPG",
"IMG_2367.JPG",
"IMG_2368.JPG",
"IMG_2369.JPG",
"IMG_2370.JPG",
"IMG_2371.JPG",
"IMG_2372.JPG",
"IMG_2373.JPG",
"IMG_2374.JPG",
"IMG_2375.JPG",
"IMG_2376.JPG",
"IMG_2377.JPG",
"IMG_2378.JPG",
"IMG_2379.JPG",
"IMG_2380.JPG",
"IMG_2381.JPG",
"IMG_2382.JPG",
"IMG_2383.JPG",
"IMG_2384.JPG",
"IMG_2385.JPG",
"IMG_2386.JPG",
"IMG_2387.JPG",
"IMG_2388.JPG",
"IMG_2389.JPG",
"IMG_2390.JPG",
"IMG_2391.JPG",
"IMG_2392.JPG",
"IMG_2393.JPG",
"IMG_2394.JPG",
"IMG_2395.JPG",
"IMG_2396.JPG",
"IMG_2397.JPG",
"IMG_2398.JPG",
"IMG_2399.JPG",
"IMG_2400.JPG",
"IMG_2401.JPG",
};

#endif // __COMMON_HPP__
