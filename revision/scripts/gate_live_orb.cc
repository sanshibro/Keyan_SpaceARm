// Streaming, headless wrapper around the frozen native ORB-SLAM3 RGB-D library.
// Input: FRAME <id> <exposure_seconds> <rgb_file> <uint16_depth_file>; QUIT.
// Output: POSE <id> <CSV row>. All other native stdout is redirected to stderr.
#include "System.h"
#include "Map.h"
#include "MapPoint.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <cstdio>

int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"usage: gate_live_orb vocabulary settings native.csv\n";return 2;}
 std::cout.rdbuf(std::cerr.rdbuf());
 cv::FileStorage settings(argv[2],cv::FileStorage::READ);
 const double fx=(double)settings["Camera1.fx"],fy=(double)settings["Camera1.fy"];
 const double cx=(double)settings["Camera1.cx"],cy=(double)settings["Camera1.cy"];
 std::ofstream out(argv[3]);if(!out)return 2;
 ORB_SLAM3::System slam(argv[1],argv[2],ORB_SLAM3::System::RGBD,false);
 out<<"t,state,map_id,matched,features,reproj_px,track_ms,occluded";
 for(int k=0;k<12;k++)out<<",r"<<k;out<<"\n";out.flush();
 std::printf("READY\n");std::fflush(stdout);
 std::string line;
 while(std::getline(std::cin,line)){
  if(line=="QUIT")break;
  std::istringstream in(line);std::string tag,rgbpath,depthpath;long id;double t;
  if(!(in>>tag>>id>>t>>rgbpath>>depthpath)||tag!="FRAME"){std::printf("ERROR malformed\n");std::fflush(stdout);continue;}
  cv::Mat rgb=cv::imread(rgbpath,cv::IMREAD_COLOR),depth=cv::imread(depthpath,cv::IMREAD_UNCHANGED);
  if(rgb.empty()||depth.type()!=CV_16UC1){std::printf("ERROR image\n");std::fflush(stdout);return 3;}
  const auto begin=std::chrono::steady_clock::now();
  const auto Tcw=slam.TrackRGBD(rgb,depth,t);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  const auto key=slam.GetTrackedKeyPointsUn();const auto points=slam.GetTrackedMapPoints();
  std::map<unsigned long,int> maps;std::vector<double> residual;
  for(size_t i=0;i<std::min(key.size(),points.size());++i){
   auto *point=points[i];if(!point||point->isBad()||point->Observations()<1||!point->GetMap())continue;
   const auto local=Tcw*point->GetWorldPos();if(local.z()<=.0001)continue;
   ++maps[point->GetMap()->GetId()];
   residual.push_back(std::hypot(fx*local.x()/local.z()+cx-key[i].pt.x,fy*local.y()/local.z()+cy-key[i].pt.y));
  }
  long map=-1;int most=0;for(const auto &p:maps)if(p.second>most){map=p.first;most=p.second;}
  double median=NAN;if(!residual.empty()){auto mid=residual.begin()+residual.size()/2;std::nth_element(residual.begin(),mid,residual.end());median=*mid;}
  bool black=cv::countNonZero(rgb.reshape(1))==0;
  std::ostringstream row;row<<std::setprecision(10)<<t<<','<<slam.GetTrackingState()<<','<<map<<','<<most<<','<<key.size()<<','<<median<<','<<ms<<','<<black;
  const auto pose=Tcw.matrix();for(int i=0;i<3;i++)for(int j=0;j<4;j++)row<<','<<pose(i,j);
  out<<row.str()<<"\n";out.flush();
  std::printf("POSE %ld %s\n",id,row.str().c_str());std::fflush(stdout);
 }
 slam.Shutdown();out.close();return 0;
}

