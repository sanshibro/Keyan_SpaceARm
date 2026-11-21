// Local analytic obstacle SDF sampled on an immutable rectangular subgrid.
// Values select candidate poses; the accepted spline has a separate CAD proof.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

extern "C" void local_field(int no,const double* obs,const double* origin,double step,
                            int nx,int ny,int nz,float cap,float* field) {
    std::fill(field,field+nx*ny*nz,cap);
    const int shape[]={nx,ny,nz};
    for(int i=0;i<no;++i) {
        const double* o=obs+11*i;
        int lo[3],hi[3];
        for(int d=0;d<3;++d) {
            lo[d]=std::max(0,int(std::floor((o[d]-o[3+d]-cap-origin[d])/step)));
            hi[d]=std::min(shape[d]-1,int(std::ceil((o[d]+o[3+d]+cap-origin[d])/step)));
        }
        for(int x=lo[0];x<=hi[0];++x)for(int y=lo[1];y<=hi[1];++y) {
            double dx=origin[0]+x*step-o[0],dy=origin[1]+y*step-o[1];
            for(int z=lo[2];z<=hi[2];++z) {
                double dz=origin[2]+z*step-o[2],d;
                if(o[9]>0) {
                    double dot=dx*o[6]+dy*o[7]+dz*o[8];
                    double radial=std::sqrt(std::max(0.,dx*dx+dy*dy+dz*dz-dot*dot))-o[9];
                    double axial=std::abs(dot)-o[10];
                    double a=std::max(0.,radial),b=std::max(0.,axial);
                    d=std::sqrt(a*a+b*b)+std::min(0.,std::max(radial,axial));
                } else {
                    double a=std::abs(dx)-o[3],b=std::abs(dy)-o[4],c=std::abs(dz)-o[5];
                    double h=std::min(0.,std::max({a,b,c}));
                    a=std::max(0.,a);b=std::max(0.,b);c=std::max(0.,c);
                    d=std::sqrt(a*a+b*b+c*c)+h;
                }
                float& f=field[(x*ny+y)*nz+z];f=std::min(f,static_cast<float>(d));
            }
        }
    }
}
extern "C" void pose_field(int n,const double* pts,int nq,const double* q,
    const double* origin,double step,int nx,int ny,int nz,const float* field,double cap,
    double threshold,double* out) {
    double inv=1./step;
    #pragma omp parallel for num_threads(2) if(nq>=256) schedule(static)
    for(int j=0;j<nq;++j) {
        const double* s=q+6*j;double cy=std::cos(s[3])*inv,sy=std::sin(s[3])*inv,m=cap;
        double x0=(s[0]-origin[0])*inv,y0=(s[1]-origin[1])*inv,z0=(s[2]-origin[2])*inv;
        for(int k=0;k<n;++k) {
            const double* p=pts+3*k;
            double x=x0+cy*p[0]-sy*p[1],y=y0+sy*p[0]+cy*p[1],z=z0+inv*p[2];
            if(x<0 || x>=nx-1 || y<0 || y>=ny-1 || z<0 || z>=nz-1){m=-1.;break;}
            int a=int(x),b=int(y),c=int(z);
            double u=x-a,v=y-b,w=z-c;
            const float* t=field+(a*ny+b)*nz+c;
            int s1=nz,s2=ny*nz;
            double val=(1-u)*((1-v)*(t[0]+w*(t[1]-t[0]))+v*(t[s1]+w*(t[s1+1]-t[s1])))+
                        u*((1-v)*(t[s2]+w*(t[s2+1]-t[s2]))+v*(t[s2+s1]+w*(t[s2+s1+1]-t[s2+s1])));
            m=std::min(m,val);if(m<threshold)break;
        }
        out[j]=m;
    }
}
