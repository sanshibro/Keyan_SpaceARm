// Headless instrumentation around the installed ORB-SLAM3 RGB-D library.
#include "System.h"
#include "Map.h"
#include "MapPoint.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <thread>

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: orb vocabulary settings dataset natural|occluded output.csv\n";
        return 2;
    }
    const std::string root = argv[3], condition = argv[4];
    std::ifstream frames(root + "/frames.txt");
    std::ofstream out(argv[5]);
    if (!frames || !out || (condition != "natural" && condition != "occluded")) return 2;
    cv::FileStorage settings(argv[2], cv::FileStorage::READ);
    const double fx = (double)settings["Camera1.fx"], fy = (double)settings["Camera1.fy"];
    const double cx = (double)settings["Camera1.cx"], cy = (double)settings["Camera1.cy"];
    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);
    out << "t,state,map_id,matched,features,reproj_px,track_ms,occluded";
    for (int k=0;k<12;k++) out << ",r" << k;
    out << "\n" << std::setprecision(10);
    std::string rgbName, depthName;
    double t; int n=0;
    while (frames >> t >> rgbName >> depthName) {
        cv::Mat rgb = cv::imread(root + "/" + rgbName, cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(root + "/" + depthName, cv::IMREAD_UNCHANGED);
        if (rgb.empty() || depth.type()!=CV_16UC1) return 3;
        const bool black = condition=="occluded" && t>=31 && t<35;
        if (black) { rgb.setTo(0); depth.setTo(0); }
        const auto begin=std::chrono::steady_clock::now();
        const auto Tcw=slam.TrackRGBD(rgb,depth,t);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        const auto key=slam.GetTrackedKeyPointsUn();
        const auto points=slam.GetTrackedMapPoints();
        std::map<unsigned long,int> maps;
        std::vector<double> residual;
        for (size_t i=0;i<std::min(key.size(),points.size());++i) {
            auto *point=points[i];
            if (!point || point->isBad() || point->Observations()<1 || !point->GetMap()) continue;
            const auto local=Tcw*point->GetWorldPos();
            if (local.z()<=0.0001) continue;
            ++maps[point->GetMap()->GetId()];
            residual.push_back(std::hypot(fx*local.x()/local.z()+cx-key[i].pt.x,
                                          fy*local.y()/local.z()+cy-key[i].pt.y));
        }
        long map=-1; int most=0;
        for (const auto &pair:maps) if (pair.second>most) {map=pair.first;most=pair.second;}
        double median=NAN;
        if (!residual.empty()) {
            auto mid=residual.begin()+residual.size()/2;
            std::nth_element(residual.begin(),mid,residual.end()); median=*mid;
        }
        out << t << ',' << slam.GetTrackingState() << ',' << map << ',' << most << ',' << key.size()
            << ',' << median << ',' << ms << ',' << black;
        const auto pose=Tcw.matrix();
        for (int i=0;i<3;++i) for (int j=0;j<4;++j) out << ',' << pose(i,j);
        out << '\n';
        if (++n%25==0) out.flush();
        // Give the stock local mapping thread its native inter-frame time.
        std::this_thread::sleep_until(begin+std::chrono::milliseconds(40));
    }
    out.close(); slam.Shutdown();
    return n?0:3;
}
