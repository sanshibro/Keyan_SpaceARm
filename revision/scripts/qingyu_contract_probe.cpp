// Exercise the installed SDK on an isolated ROS master, without a serial node.
#include <qingyu_api_sdk/client.hpp>
#include <qingyu_api/QingyuCommand.h>
#include <ros/ros.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>

int main(int argc,char** argv) {
  const char* master=std::getenv("ROS_MASTER_URI");
  if(!master || std::string(master)!="http://127.0.0.1:11483") {
    std::cerr<<"Probe requires its isolated localhost master on 11483\n";return 2;
  }
  if(argc!=3){std::cerr<<"input CSV, output CSV required\n";return 2;}
  ros::init(argc,argv,"paper_qingyu_contract_probe");ros::NodeHandle nh;
  std::ofstream output(argv[2]);
  output<<"receive_time_s,theta1,phi1,theta2,phi2,theta3,phi3,enable,cmd_mode,control_mode\n";
  int received=0;
  auto sub=nh.subscribe<qingyu_api::QingyuCommand>("/paper_revision_probe/command",20,
    [&](const qingyu_api::QingyuCommand::ConstPtr& m){
      output<<std::setprecision(12)<<ros::WallTime::now().toSec()<<","<<m->theta1<<","<<m->phi1
        <<","<<m->theta2<<","<<m->phi2<<","<<m->theta3<<","<<m->phi3<<","<<int(m->enable)
        <<","<<int(m->cmd_mode)<<","<<int(m->control_mode)<<"\n";output.flush();received++;
    });
  qingyu_api_sdk::ClientOptions options;
  options.node_name="paper_qingyu_contract_probe_sdk";
  options.command_topic="/paper_revision_probe/command";
  options.pose_topic="/paper_revision_probe/pose";
  options.serial_node_name="/paper_revision_probe/no_serial_backend";
  qingyu_api_sdk::Client client(options);
  if(!client.start()){std::cerr<<client.lastError()<<"\n";return 3;}
  for(int i=0;i<200 && sub.getNumPublishers()==0;i++){ros::spinOnce();ros::WallDuration(.01).sleep();}
  if(sub.getNumPublishers()==0){std::cerr<<"No isolated publisher\n";return 4;}
  std::ifstream input(argv[1]);std::string line;std::getline(input,line);int sent=0;
  while(std::getline(input,line)) {
    if(line.empty())continue;
    for(char &c:line)if(c==',')c=' ';
    std::istringstream row(line);qingyu_api_sdk::JointCommand cmd;int enable;
    if(!(row>>cmd.theta1>>cmd.phi1>>cmd.theta2>>cmd.phi2>>cmd.theta3>>cmd.phi3>>enable))return 5;
    cmd.enable=enable!=0;
    if(!client.moveJoints(cmd)){std::cerr<<client.lastError()<<"\n";return 6;}
    ++sent;
    for(int i=0;i<200 && received<sent;i++){ros::spinOnce();ros::WallDuration(.005).sleep();}
    if(received!=sent){std::cerr<<"Missing command message\n";return 7;}
  }
  client.stop();std::cout<<"SDK commands sent="<<sent<<" received="<<received<<"\n";
  return sent>0 && sent==received?0:8;
}
