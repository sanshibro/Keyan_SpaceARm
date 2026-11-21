// Same conservative component/cylinder certificates as M1, batched over poses.
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" void map_batch(int n,const double*robot,int no,const double*obstacles,int nq,const double*states,double*out){
 for(int k=0;k<nq;++k){const double*s=states+6*k;double c=std::cos(s[3]),sn=std::sin(s[3]);
  double lo[3]={1e9,1e9,1e9},hi[3]={-1e9,-1e9,-1e9};
  std::vector<double> transformed;transformed.resize(n*10);
  for(int i=0;i<n;++i){const double*p=robot+10*i;double*q=transformed.data()+10*i;
   q[0]=s[0]+c*p[0]-sn*p[1];q[1]=s[1]+sn*p[0]+c*p[1];q[2]=s[2]+p[2];
   q[3]=std::abs(c)*p[3]+std::abs(sn)*p[4];q[4]=std::abs(sn)*p[3]+std::abs(c)*p[4];q[5]=p[5];
   if(p[9]<0){q[3]=q[4]=-p[9];}else if(p[9]>0){q[3]=q[4]=q[5]=p[9];}
   for(int j=6;j<10;++j)q[j]=p[j];for(int j=0;j<3;++j){lo[j]=std::min(lo[j],q[j]-q[j+3]-q[6]);hi[j]=std::max(hi[j],q[j]+q[j+3]+q[6]);}}
  std::vector<int>near;for(int j=0;j<no;++j){const double*o=obstacles+11*j;bool keep=true;for(int a=0;a<3;++a)if(std::max(o[a]-o[a+3]-hi[a],lo[a]-o[a]-o[a+3])>.30)keep=false;if(keep)near.push_back(j);}
  out[2*k]=out[2*k+1]=.30;
  for(int i=0;i<n;++i){const double*p=transformed.data()+10*i;int group=(int)p[7];
   for(int j:near){const double*o=obstacles+11*j;double dx=p[0]-o[0],dy=p[1]-o[1],dz=p[2]-o[2];
    double x=std::abs(dx)-p[3]-o[3],y=std::abs(dy)-p[4]-o[4],z=std::abs(dz)-p[5]-o[5];
    if(std::max({x,y,z})-p[6]>=out[2*k+group])continue;
    double ex=std::max(0.,x),ey=std::max(0.,y),ez=std::max(0.,z);double gap=std::sqrt(ex*ex+ey*ey+ez*ez)+std::min(0.,std::max({x,y,z}));
    if(o[9]>0){double ax=o[6],ay=o[7],az=o[8],axial=dx*ax+dy*ay+dz*az,ux=dx-axial*ax,uy=dy-axial*ay,uz=dz-axial*az,rr=std::sqrt(ux*ux+uy*uy+uz*uz);
      gap=std::max(gap,std::abs(axial)-o[10]-(p[9]>0?p[9]:p[3]*std::abs(ax)+p[4]*std::abs(ay)+p[5]*std::abs(az)));
      if(rr>1e-12)gap=std::max(gap,rr-o[9]-(p[9]>0?p[9]:(p[3]*std::abs(ux)+p[4]*std::abs(uy)+p[5]*std::abs(uz))/rr));}
    out[2*k+group]=std::min(out[2*k+group],gap-p[6]);
   }
  }
 }
}
