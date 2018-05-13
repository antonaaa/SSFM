#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <mutex>


#include "Common.hpp"
using namespace std;
using namespace cv;


void ExtractFeature(Frame& frame);
void ComputeMatch(SFM& sfm, int i);
void Initialize(SFM& sfm);
void ShowPointCloud(SFM& sfm);
void BundleAdjustment(SFM& sfm, int st, int ed);
void AddMoreViewToReconstruction(SFM& sfm, int ith);
void PostFilter(SFM& sfm);
void SaveForPMVS(const SFM& sfm);

int main()
{
	SFM sfm;

	// step1 提取特征
	for(size_t i = 0; i < IMAGE_USE; ++i)
	{
		Frame frame(i, IMAGE_DIR + IMAGES[i]);
		frame.image = imread(frame.image_name);
		resize(frame.image, frame.image, frame.image.size() / IMAGE_DOWNSAMPLE);

		ExtractFeature(frame);

		sfm.frames.push_back(frame);
	}

	// step2 计算匹配	
	for(size_t i = 0; i < sfm.frames.size() - 1; ++i)
	{
		ComputeMatch(sfm, i);
	}

	// step3 初始化
	Initialize(sfm);
	BundleAdjustment(sfm, 0, 1);
	ShowPointCloud(sfm);

	// step4 添加更多的图片进行重建
	for(size_t i = 2; i < sfm.frames.size(); ++i)
	{
		cout << "current add " << i << "th image " << endl;
		AddMoreViewToReconstruction(sfm, i);
		BundleAdjustment(sfm, 0, i);
		cout << endl;
		cout << endl;
		PostFilter(sfm);
		BundleAdjustment(sfm, 0, i);
		/* ShowPointCloud(sfm); */
		/* ShowPointCloud(sfm); */
	}

	SaveForPMVS(sfm);
}
































// step1 提取特征
void keyPoints2Points(const vector<KeyPoint>& kps, vector<Point2d>& pts)
{
	for(size_t i = 0; i < kps.size(); ++i)
	{
		Point2d p(kps[i].pt.x, kps[i].pt.y);
		pts.push_back(p);
	}
}

void ExtractFeature(Frame& frame)
{

	Ptr<AKAZE> feature = AKAZE::create();
	vector<KeyPoint> kpts;
	feature->detect(frame.image, kpts);
	feature->compute(frame.image, kpts, frame.image_pts_desc);
	keyPoints2Points(kpts, frame.image_pts);


	frame.pt_3ds_indicator.resize(frame.image_pts.size());

	for(size_t i = 0; i < frame.image_pts.size(); ++i)
	{
		frame.pt_3ds_indicator[i] = -1;
	}
}




// step2 计算匹配
void ShowMatch(Frame& frame1, Frame& frame2, vector<DMatch> matches)
{
	namedWindow("image", WINDOW_NORMAL);
	Mat canvas = frame1.image.clone();
	canvas.push_back(frame2.image.clone());

	for(size_t i = 0; i < matches.size(); ++i)
	{
		point2d_t queryIdx = matches[i].queryIdx;
		point2d_t trainIdx = matches[i].trainIdx;
		
		line(canvas, frame1.image_pts[queryIdx], frame2.image_pts[trainIdx] + Point2d(0, frame1.image.rows), Scalar(0, 0, 255), 2);
	}

	imshow("image", canvas);
	waitKey(1);
}
void ComputeMatch(SFM& sfm, int i)
{

	Frame& frame1 = sfm.frames[i];
	Frame& frame2 = sfm.frames[i + 1];

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> initialMatches;
	vector<DMatch> matches;

	matcher->knnMatch(frame1.image_pts_desc, frame2.image_pts_desc, initialMatches, 2);

	int good_matches = 0;
	for(auto& m : initialMatches)
	{
		if(m[0].distance < 0.7 * m[1].distance)
		{
			good_matches++;
			matches.push_back(m[0]);
		}
	}

	sfm.matches.emplace_back(std::move(matches));

		
	assert(good_matches >= 50);
	cout << "Feature matching : " << frame1.image_id << " " << frame2.image_id << " ==> " << good_matches << "/" << initialMatches.size() << endl;

	ShowMatch(frame1, frame2, sfm.matches[i]);

}

// step3 初始化

bool HasPositiveDepth(Point3d p3d, const Mat& R, const Mat& t)
{
	Mat pt = Mat::zeros(3, 1, CV_64F);
	pt.at<double>(0, 0) = p3d.x;
	pt.at<double>(1, 0) = p3d.y;
	pt.at<double>(2, 0) = p3d.z;
	pt = R * pt + t;
	if(pt.at<double>(2, 0) > std::numeric_limits<double>::epsilon())
		return true;
	return false;
}
void RemoveWorldPtsByVisiable(const vector<Point3d> point3ds, 
							  const Mat& R1,
							  const Mat& t1,
							  const Mat& R2,
							  const Mat& t2,
							  Mat& mask)
{
	
	int total_num = countNonZero(mask);
	int outlier_num = 0;
	for(int i = 0; i < mask.rows; ++i)
	{
		if(mask.at<uchar>(i, 0) == 0)
			continue;

		if(not HasPositiveDepth(point3ds[i], R1, t1) || not HasPositiveDepth(point3ds[i], R2, t2))
		{
			mask.at<uchar>(i, 0) = 0;
			outlier_num += 1;
		}
	}

	if(outlier_num > 0)
		cout << "remove " << outlier_num << " outliers outof " << total_num << " with not visiable" << endl;
}


double GetReProjectionError(const Point3d& p3d, 
							const Point2d& pt, 
							const Mat& R, 
							const Mat& t, 
							const Mat& K)
{
	Mat rvec;
	Rodrigues(R, rvec);
	vector<Point2d> projPt;
	vector<Point3d> _p3d;
	_p3d.push_back(p3d);
	projectPoints(_p3d, rvec, t, K, noArray(), projPt);
	// 2 norm
	double err = norm(pt - projPt[0]);
	return err;
}
void RemoveWorldPtsByReprojectionError(const vector<Point3d> point3ds,
									   const vector<Point2d> aligned_pts1,
									   const Mat& R1,
									   const Mat& t1,
									   const vector<Point2d> aligned_pts2,
									   const Mat& R2,
									   const Mat& t2,
									   const Mat& K,
									   Mat& mask,
									   double threshold_in_pixles = MAX_3D_REPROJECTION_ERROR)
{
	
	
	int total_num = countNonZero(mask);
	int outlier_num = 0;
	for(int i = 0; i < mask.rows; ++i)
	{
		if(mask.at<uchar>(i, 0) == 0)
			continue;

		double error1 = GetReProjectionError(point3ds[i], aligned_pts1[i], R1, t1, K);
		double error2 = GetReProjectionError(point3ds[i], aligned_pts2[i], R2, t2, K);
		if(error1 > threshold_in_pixles || error2 > threshold_in_pixles)
		{
			mask.at<uchar>(i, 0) = 0;
			outlier_num += 1;
		}
	}

	if(outlier_num > 0)
		cout << "remove " << outlier_num << " outliers outof " << total_num << " with repeojecion error bigger than " << (threshold_in_pixles) << endl;
}


double CalculateParallaxAngle(const Point3d& proj_center1,
							  const Point3d& proj_center2,
							  const Point3d& p3d)
{
	// 余弦定理
	// cosA = (b^2 + c^2 - a^2) / 2bc
	const double baseline = norm(proj_center1 - proj_center2);
	const double ray1 = norm(p3d - proj_center1);
	const double ray2 = norm(p3d - proj_center2);

	const double angle = std::abs(
			std::acos((ray1 * ray1 + ray2 * ray2 - baseline * baseline) / (2 * ray1 * ray2))
	);

	if(ceres::IsNaN(angle))
	{
		return 0;
	}
	else
	{
		return std::min(angle, M_PI - angle) * 180 / M_PI;
	}

}
							  
void RemoveWorldPtsByParallaxAngle(const vector<Point3d> point3ds,
								   const Mat& R1,
								   const Mat& t1,
								   const Mat& R2,
								   const Mat& t2,
								   Mat& mask,
								   double threshold_in_angle_degree = MIN_PARALLAX_DEGREE)
{
	
	int total_num = countNonZero(mask);
	int outlier_num = 0;
	for(int i = 0; i < mask.rows; ++i)
	{
		if(mask.at<uchar>(i, 0) == 0)
			continue;

		// 光心在世界坐标系下的坐标
		Mat O1 = -R1.t() * t1;
		Mat O2 = -R2.t() * t2;

		Point3d proj_center1(O1.at<double>(0, 0), O1.at<double>(1, 0), O1.at<double>(2, 0));
		Point3d proj_center2(O2.at<double>(0, 0), O2.at<double>(1, 0), O2.at<double>(2, 0));

		double angle = CalculateParallaxAngle(proj_center1, proj_center2, point3ds[i]);

		if(angle < threshold_in_angle_degree)
		{
			mask.at<uchar>(i, 0) = 0;
			outlier_num += 1;
		}
	}

	if(outlier_num > 0)
		cout << "remove " << outlier_num << " outliers outof " << total_num << " with paralax angle less than " << (threshold_in_angle_degree) << endl;
}
								  
void GetAlignPointsFromMatch(const vector<Point2d>& pts1, 
							 const vector<Point2d>& pts2, 
							 const vector<DMatch>& matches, 
							 vector<Point2d>& aligned_pts1,
							 vector<Point2d>& aligned_pts2)
{
	for(size_t i = 0; i < matches.size(); ++i)
	{
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx;

		aligned_pts1.push_back(pts1[queryIdx]);
		aligned_pts2.push_back(pts2[trainIdx]);
	}
}
void Initialize(SFM& sfm)
{
	assert(sfm.frames.size() >= 2);
	
	Frame& frame1 = sfm.frames[0];
	Frame& frame2 = sfm.frames[1];
	vector<DMatch>& matches = sfm.matches[0];

	vector<Point2d> aligned_pts1;
	vector<Point2d> aligned_pts2;
	
	GetAlignPointsFromMatch(frame1.image_pts, frame2.image_pts, matches, aligned_pts1, aligned_pts2);

	
	double cx = frame1.image.size().width / 2.0;
	double cy = frame1.image.size().height / 2.0;
	Mat K = Mat::eye(3, 3, CV_64F);
	K.at<double>(0, 0) = FOCAL_LENGTH;
	K.at<double>(1, 1) = FOCAL_LENGTH;
	K.at<double>(0, 2) = cx;
	K.at<double>(1, 2) = cy;

	sfm.K = K;


	Mat mask;
	// 恢复位姿

	Mat E = findEssentialMat(aligned_pts1, aligned_pts2, K, RANSAC, 0.995, MAX_2D_REPROJECTION_ERROR, mask);
	frame1.R = Mat::eye(3, 3, CV_64F);
	frame1.t = Mat::zeros(3, 1, CV_64F);
	recoverPose(E, aligned_pts1, aligned_pts2, K, frame2.R, frame2.t, mask);

	// 投影矩阵
	Mat P1, P2;
	hconcat(K * frame1.R, K * frame1.t, P1);
	hconcat(K * frame2.R, K * frame2.t, P2);

	// 三角测量
	Mat X;
	vector<Point3d> point3ds;
	triangulatePoints(P1, P2, aligned_pts1, aligned_pts2, X);
	convertPointsFromHomogeneous(X.t(), point3ds);


	// 检查3d点的
	// 可见性
	RemoveWorldPtsByVisiable(point3ds, frame1.R, frame1.t, frame2.R, frame2.t, mask);
	// 重投影误差
	RemoveWorldPtsByReprojectionError(point3ds, aligned_pts1, frame1.R, frame1.t, aligned_pts2, frame2.R, frame2.t, sfm.K, mask); 
	// 视差角
	RemoveWorldPtsByParallaxAngle(point3ds, frame1.R, frame1.t, frame2.R, frame2.t, mask);

	// 更新tracks
	for(int i = 0; i < mask.rows; ++i)
	{
		if(mask.at<uchar>(i, 0) == 0)
			continue;

		Track track;
		track.p3d = point3ds[i];
		point2d_t queryIdx = matches[i].queryIdx;
		point2d_t trainIdx = matches[i].trainIdx;

		// 建立track
		track.elements.emplace_back(frame1.image_id, queryIdx);
		track.elements.emplace_back(frame2.image_id, trainIdx);

		sfm.tracks.emplace_back(std::move(track));
		sfm.is_new.push_back(false);

		point3d_t point3d_idx = sfm.tracks.size() - 1;
		frame1.pt_3ds_indicator[queryIdx] = point3d_idx;
		frame2.pt_3ds_indicator[trainIdx] = point3d_idx;
	}
	cout << "initialize triangulate : " << frame1.image_id << " " << frame2.image_id << " ==> " << countNonZero(mask)<< " landmarks" << endl;
}







// step4 
void GetCorresponding2d3dPoints(SFM& sfm, int ith, int jth, vector<Point2d>& p2ds, vector<Point3d>& p3ds, vector<size_t>& p2d_idxs, vector<point3d_t>& p3d_idxs)
{
	assert(ith + 1 == jth);
	assert(jth < sfm.frames.size());
	const Frame& frame1 = sfm.frames[ith];
	const Frame& frame2 = sfm.frames[jth];
	const auto& matches = sfm.matches[ith];

	for(size_t i = 0; i < matches.size(); ++i)
	{
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx;

		if(frame1.pt_3ds_indicator[queryIdx] == -1)
			continue;


		point3d_t point3d_idx = frame1.pt_3ds_indicator[queryIdx];
		p2ds.push_back(frame2.image_pts[trainIdx]);
		p3ds.push_back(sfm.tracks[point3d_idx].p3d);

		p2d_idxs.push_back(i);
		p3d_idxs.push_back(point3d_idx);
	}
}


void AddMorePoints(SFM& sfm, int ith, const vector<Point2d>& aligned_pts1, const vector<Point2d>& aligned_pts2, const vector<DMatch>& matches)
{
	Frame& frame1 = sfm.frames[ith - 1];
	Frame& frame2 = sfm.frames[ith];



	Mat P1, P2;	
	hconcat(sfm.K * frame1.R, sfm.K * frame1.t, P1);
	hconcat(sfm.K * frame2.R, sfm.K * frame2.t, P2);

	Mat X;
	vector<Point3d> point3ds;
	triangulatePoints(P1, P2, aligned_pts1, aligned_pts2, X);
	convertPointsFromHomogeneous(X.t(), point3ds);

	Mat mask = Mat::ones(point3ds.size(), 1, CV_8U);
	// 可见性
	RemoveWorldPtsByVisiable(point3ds, frame1.R, frame1.t, frame2.R, frame2.t, mask);
	// 重投影误差
	RemoveWorldPtsByReprojectionError(point3ds, aligned_pts1, frame1.R, frame1.t, aligned_pts2, frame2.R, frame2.t, sfm.K, mask); 
	// 视差角
	RemoveWorldPtsByParallaxAngle(point3ds, frame1.R, frame1.t, frame2.R, frame2.t, mask);
	int inlier_num = 0;
	for(size_t i = 0; i < point3ds.size(); ++i)
	{
		
		if(mask.at<uchar>(i, 0) == 0)
			continue;
		inlier_num += 1;
		/* double error1 = GetReProjectionError(point3ds[i], aligned_pts1[i], sfm.frames[ith - 1].R, sfm.frames[ith - 1].t, sfm.K); */
		/* double error2 = GetReProjectionError(point3ds[i], aligned_pts2[i], sfm.frames[ith].R, sfm.frames[ith].t, sfm.K); */

		/* if(error1 > MAX_2D_REPROJECTION_ERROR || error2 > MAX_2D_REPROJECTION_ERROR) */
			/* continue; */

		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx;
		Track track;
		track.p3d = point3ds[i];
		track.elements.emplace_back(frame1.image_id, queryIdx);
		track.elements.emplace_back(frame2.image_id, trainIdx);

		sfm.tracks.emplace_back(std::move(track));
		sfm.is_new.push_back(true);
		point3d_t point3d_idx = sfm.tracks.size() - 1;

		frame1.pt_3ds_indicator[queryIdx] = point3d_idx;
		frame2.pt_3ds_indicator[trainIdx] = point3d_idx;
	}
	cout << "add more points: " << frame1.image_id << " " << frame2.image_id << " ==> " << inlier_num << " point3ds" << endl;
}
void AddMoreViewToReconstruction(SFM& sfm, int ith)
{
	
	Frame& frame1 = sfm.frames[ith - 1];
	Frame& frame2 = sfm.frames[ith];
	vector<DMatch>& matches = sfm.matches[ith - 1];

	vector<Point2d> p2ds;
	vector<Point3d> p3ds;
	vector<size_t> p2d_idxs;
	vector<point3d_t> p3d_idxs;
	GetCorresponding2d3dPoints(sfm, ith - 1, ith, p2ds, p3ds, p2d_idxs, p3d_idxs);

	printf("p2ds.size() = %lu, p3ds.size() = %lu\n", p2ds.size(), p3ds.size());

	Mat mask;
	Mat rvec, tvec;
	// PnP求解位姿
	solvePnPRansac(p3ds, p2ds, sfm.K, Mat(), rvec, tvec, true , 100, 8.0, 0.995, mask);
	
	Rodrigues(rvec, sfm.frames[ith].R);
	sfm.frames[ith].t = tvec;


	// 更新旧的track
	for(int i = 0; i < mask.rows; ++i)
	{
		int inlier_idx = mask.at<uchar>(i, 0);
		point3d_t point3d_idx = p3d_idxs[inlier_idx];
		point2d_t point2d_idx = matches[p2d_idxs[inlier_idx]].trainIdx;
		//TODO 检查新增加的这个点的
		// 可见性
		// 重投影误差
		// 角度
		/* printf("%d %d \n", point3d_idx, sfm.tracks.size()); */
		/* printf("%d\n", point3d_idx); */
		sfm.tracks[point3d_idx].elements.emplace_back(frame2.image_id, point2d_idx);
	}


	vector<Point2d> aligned_pts1;
	vector<Point2d> aligned_pts2;
	vector<DMatch> aligned_matches;
	size_t m = 0;
	for(size_t i = 0; i < matches.size(); ++i)
	{
		if(m < p2d_idxs.size() && i == p2d_idxs[m])
		{
			m++;
			continue;
		}
		point2d_t queryIdx = matches[i].queryIdx;
		point2d_t trainIdx = matches[i].trainIdx;
		aligned_pts1.push_back(frame1.image_pts[queryIdx]);
		aligned_pts2.push_back(frame2.image_pts[trainIdx]);

		aligned_matches.push_back(matches[i]);

	}

	printf("aligned_pts1.size() = %lu, aligned_pts2.size() = %lu\n", aligned_pts1.size(), aligned_pts2.size());
	
	AddMorePoints(sfm, ith, aligned_pts1, aligned_pts2, aligned_matches);

}


void RemoveWorldPtsByReprojectionError(SFM& sfm, double threshold_in_pixels = MAX_3D_REPROJECTION_ERROR)
{
	/* threshold_in_pixels *= threshold_in_pixels; */

	int aviliable_num = sfm.Length();
	int outlier_num = 0;
	for(auto& track : sfm.tracks)
	{
		if(not track.aviliable)
			continue;

		for(auto& element : track.elements)
		{
			image_t image_id = element.image_id;
			point2d_t point2d_idx = element.point2d_idx;
			const Frame& frame = sfm.frames[image_id];
			double error = GetReProjectionError(track.p3d, frame.image_pts[point2d_idx], frame.R, frame.t, sfm.K);

			// 大于阈值
			if(error > threshold_in_pixels)
			{
				track.aviliable = false;
				outlier_num += 1;
				break;
			}
		}
	}

	sfm.Update();

	if(outlier_num > 0)
		cout << "post filter remove " << outlier_num << " outliers outof " << aviliable_num << " with reprojection error bigger than " << (threshold_in_pixels) << endl;
}


void PostFilter(SFM& sfm)
{
	RemoveWorldPtsByReprojectionError(sfm);	
}




// step5

void SaveForPMVS(const SFM& sfm)
{

	system("mkdir -p root/visualize");
	system("mkdir -p root/txt");
	system("mkdir -p root/models");

	ofstream option("root/options.txt");

	option << "timages  -1 " << 0 << " " << (sfm.frames.size() - 1) << endl;;
	option << "oimages 0" << endl;
	option << "level 1" << endl;

	option.close();

	for (size_t i=0; i < sfm.frames.size(); i++) {

		char str[256];
		Mat R = sfm.frames[i].R;
		Mat t = sfm.frames[i].t;
		Mat P;
		hconcat(sfm.K * R, sfm.K * t, P);

		/* sprintf(str, "cp -f %s/%s root/visualize/%04d.jpg", IMAGE_DIR.c_str(), IMAGES[i].c_str(), (int)i); */
		/* system(str); */

		sprintf(str, "root/visualize/%04d.jpg", (int)i);
		imwrite(str, sfm.frames[i].image);


		sprintf(str, "root/txt/%04d.txt", (int)i);
		ofstream out(str);

		out << "CONTOUR" << endl;

		for (int j=0; j < 3; j++) {
			for (int k=0; k < 4; k++) {
				out<< P.at<double>(j, k) << " ";
			}
			out << endl;
		}
	}

	cout << endl;
	cout << "You can now run pmvs2 on the results eg. PATH_TO_PMVS_BINARY/pmvs2 root/ options.txt" << endl;

}









// Bundle Adjustment

void initLogging()
{
	google::InitGoogleLogging("SFM Bundle Adjustment!!!");
}

std::once_flag  initLoggingFlag;

struct SimpleReprojectError
{
	SimpleReprojectError(double observed_x, double observed_y) :
		observed_x(observed_x), observed_y(observed_y){}

	bool operator()(const double* const camera,
					const double* const point,
					const double* const focal,
						  double* residuals) const
	{
		double p[3];
		//对点point施加camera[0,1,2]所对应的旋转，　结果存储在p中
		ceres::AngleAxisRotatePoint(camera, point, p);
		//加上平移
		p[0] += camera[3];
		p[1] += camera[4];
		p[2] += camera[5];

		//齐次坐标归一化
		const double xp = p[0] / p[2];
		const double yp = p[1] / p[2];

		const double predicted_x = focal[0] * xp;
		const double predicted_y = focal[1] * yp;

		residuals[0] = predicted_x - double(observed_x);
		residuals[1] = predicted_y - double(observed_y);
		return true;
	}


	static ceres::CostFunction* Create(const double observed_x, const double observed_y)
	{
		//2 表示残差项的维度为2
		//6	表示camera的维度为6
		//3 表示point的维度为3
		//1 表示focal的维度为2
		return (new ceres::NumericDiffCostFunction<SimpleReprojectError, ceres::CENTRAL, 2, 6, 3, 2>(
				new SimpleReprojectError(observed_x, observed_y)));
	}
	double observed_x;
	double observed_y;
};

void BundleAdjustment(SFM& sfm, int st, int ed)
{
	std::call_once(initLoggingFlag, initLogging);

	ceres::Problem  problem;
	typedef cv::Matx<double, 1, 6> CameraVector;
	std::vector<CameraVector> cameraPose6d;

	double focal[2] = {sfm.K.at<double>(0, 0), sfm.K.at<double>(1, 1)};

	//位姿
	for(int i = st; i < ed + 1; ++i)
	{
		double angleAxis[3];
		/* ceres::RotationMatrixToAngleAxis<double>(sfm.frames[i].R.ptr<double>(), angleAxis); */
		Mat rvec;
		Rodrigues(sfm.frames[i].R, rvec);
		angleAxis[0] = rvec.at<double>(0, 0);
		angleAxis[1] = rvec.at<double>(1, 0);
		angleAxis[2] = rvec.at<double>(2, 0);
		cameraPose6d.push_back(CameraVector(
					angleAxis[0],
					angleAxis[1],
					angleAxis[2],
					sfm.frames[i].t.at<double>(0, 0),
					sfm.frames[i].t.at<double>(1, 0),
					sfm.frames[i].t.at<double>(2, 0)));
	}



	std::vector<cv::Vec3d> point3ds(sfm.tracks.size());
	int m = 0;
	for(size_t i = 0; i < sfm.tracks.size(); ++i)
	{
		if(not sfm.tracks[i].aviliable)
			continue;
		auto& track = sfm.tracks[i];
		point3ds[m] = Vec3d(track.p3d.x, track.p3d.y, track.p3d.z);

		for(const auto& element : track.elements)
		{
			image_t image_id = element.image_id;
			point2d_t point2d_idx = element.point2d_idx;
			Point2d p2d = sfm.frames[image_id].image_pts[point2d_idx];
			p2d.x -= sfm.K.at<double>(0, 2);
			p2d.y -= sfm.K.at<double>(1, 2);
			ceres::CostFunction* cost_function = SimpleReprojectError::Create(p2d.x, p2d.y);
			problem.AddResidualBlock(cost_function, NULL, cameraPose6d[image_id].val, point3ds[m].val, focal);
		}
		m++;
	}
	problem.SetParameterBlockConstant(focal);
	/* problem.SetParameterBlockConstant(cameraPose6d[0].val); */

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 1000;
	/* options.eta = 1e-2; */
	/* options.max_solver_time_in_seconds = 10; */
	/* options.logging_type = ceres::LoggingType::SILENT; */

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	/* std::cout << summary.FullReport() << std::endl; */
	if(not (summary.termination_type == ceres::CONVERGENCE))
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
		return;
	}
	else
	{	
		std::cout << std::endl
		<< "Bundle Adjustment statistics (approximated RMSE):\n"
		<< " #residuals: " << summary.num_residuals << "\n"
		<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2/ summary.num_residuals) << "\n"
		<< " Final RMSE: " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
		<< " Time (s): " << summary.total_time_in_seconds << "\n"
		<< std::endl;
	}


	//更新motion
	for(int i = st; i < ed + 1; ++i)
	{
		double R[9] = {0};
		ceres::AngleAxisToRotationMatrix(cameraPose6d[i].val, R);
		Mat rvec = (Mat_<double>(3, 1) << cameraPose6d[i](0), cameraPose6d[i](1), cameraPose6d[i](2));
		Rodrigues(rvec, sfm.frames[i].R);

		sfm.frames[i].t.at<double>(0, 0) = cameraPose6d[i](3);
		sfm.frames[i].t.at<double>(1, 0) = cameraPose6d[i](4);
		sfm.frames[i].t.at<double>(2, 0) = cameraPose6d[i](5);
	}
	// 更新structure
	m = 0;
	for(size_t i = 0; i < sfm.tracks.size(); ++i)
	{
		if(not sfm.tracks[i].aviliable)
			continue;

		auto& track = sfm.tracks[i];
		track.p3d.x = point3ds[m][0];
		track.p3d.y = point3ds[m][1];
		track.p3d.z = point3ds[m][2];
		m += 1;
	}

}









// 点云可视化
void ShowPointCloud(const vector<Point3d>& points, const vector<Vec3b>& colors)
{

	viz::WCloud a(points, colors);
	
	
	//创建窗口
	viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setBackgroundColor(viz::Color::black());
	//添加坐标系
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	myWindow.showWidget("Mesh", a);


	while(!myWindow.wasStopped())
	{
		myWindow.spinOnce(0, true);
	}
}
void ShowPointCloud(SFM& sfm)
{
	vector<Point3d> point3ds;
	vector<Vec3b> colors;
	for(size_t i = 0; i < sfm.tracks.size(); ++i)
	{
		auto& track = sfm.tracks[i];

		if(not track.aviliable) 
			continue;
		
		point3ds.push_back(track.p3d);
		if(sfm.is_new[i])
			colors.push_back(Vec3b(0, 0, 255));
		else
		{
			int r = 0;
			int g = 0;
			int b = 0;
			for(auto& element : sfm.tracks[i].elements)
			{
				image_t image_id = element.image_id;
				point2d_t point2d_idx = element.point2d_idx;

				Point2d pt = sfm.frames[image_id].image_pts[point2d_idx];
				b += sfm.frames[image_id].image.at<Vec3b>(pt)[0];
				g += sfm.frames[image_id].image.at<Vec3b>(pt)[1];
				r += sfm.frames[image_id].image.at<Vec3b>(pt)[2];
			}
			r /= sfm.tracks[i].elements.size();
			b /= sfm.tracks[i].elements.size();
			g /= sfm.tracks[i].elements.size();
			colors.push_back(Vec3b(b, g, r));
		}
	}
	if(point3ds.size() == 0)
		return;
	ShowPointCloud(point3ds, colors);

	/* sfm.updateIsNew(); */
	for(size_t i = 0; i < sfm.is_new.size(); ++i)
	{
		sfm.is_new[i] = false;
	}
}


