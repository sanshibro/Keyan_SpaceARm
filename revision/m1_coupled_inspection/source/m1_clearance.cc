// Conservative Euclidean clearance certificate for unions of component bounds.
#include <algorithm>
#include <cmath>
#include <limits>
extern "C" void gallery_clearance(int n, const double *robot, int nobs, const double *obstacles,
                 const double *state, double *out, int *pair) {
  const double c=std::cos(state[3]), s=std::sin(state[3]);
  out[0]=out[1]=10.;
  for(int i=0;i<n;++i){
    const double *p=robot+10*i;
    const int group=static_cast<int>(p[7]);
    const double cx=state[0]+c*p[0]-s*p[1],cy=state[1]+s*p[0]+c*p[1],cz=state[2]+p[2];
    const double hx=std::abs(c)*p[3]+std::abs(s)*p[4],hy=std::abs(s)*p[3]+std::abs(c)*p[4],hz=p[5];
    for(int j=0;j<nobs;++j){
      const double *o=obstacles+j*11;
      const double dx=cx-o[0],dy=cy-o[1],dz=cz-o[2];
      const double x=std::abs(dx)-hx-o[3], y=std::abs(dy)-hy-o[4], z=std::abs(dz)-hz-o[5];
      if(std::max({x,y,z})-p[6]>=out[group])continue;
      const double ex=std::max(0.,x),ey=std::max(0.,y),ez=std::max(0.,z);
      double d=std::sqrt(ex*ex+ey*ey+ez*ez)+std::min(0.,std::max({x,y,z}));
      if(o[9]>0){
        const double ax=o[6],ay=o[7],az=o[8],axial=dx*ax+dy*ay+dz*az;
        double ux=dx-axial*ax,uy=dy-axial*ay,uz=dz-axial*az;
        const double rr=std::sqrt(ux*ux+uy*uy+uz*uz);
        const double sep=std::abs(axial)-o[10]-(p[9]>0 ? p[9] : hx*std::abs(ax)+hy*std::abs(ay)+hz*std::abs(az));
        d=std::max(d,sep);
        if(rr>1e-12)d=std::max(d,rr-o[9]-(p[9]>0 ? p[9] : (hx*std::abs(ux)+hy*std::abs(uy)+hz*std::abs(uz))/rr));
      }
      d-=p[6];
      if(d<out[group]){out[group]=d;pair[group*2]=static_cast<int>(p[8]);pair[group*2+1]=j;}
    }
  }
}
