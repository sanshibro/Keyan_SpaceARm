// Dense proof from native link transforms and obstacles in one causal snapshot.
#include "m2_self.cc"

struct Box { V c, h; };
static Box world_box(const double* pose,const double* local) {
    Box b{{pose[0],pose[1],pose[2]}, {}};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        b.c[i]+=pose[3+3*i+j]*local[j];
        b.h[i]+=std::abs(pose[3+3*i+j])*local[j+3];
    }
    return b;
}
static double box_distance2(const Box&a,const Box&b) {
    double ss=0;
    for(int k=0;k<3;++k) {
        double e=std::max(0.,std::abs(a.c[k]-b.c[k])-a.h[k]-b.h[k]);
        ss+=e*e;
    }
    return ss;
}

extern "C" void path_clearances(int nr,int no,const int* types,const double* sizes,
    const double* boxes,const int* offsets,const int* counts,const double* vertices,
    const int* groups,const double* obstacle,const int* internal,int np,
    const double* origin,double radius,const double* caps,int nq,const double* moving,
    double* result,int* witness) {
    int n=nr+no;
    std::vector<double> pose(12*n);
    std::copy(obstacle,obstacle+12*no,pose.begin()+12*nr);
    Shapes shapes{types,sizes,pose.data(),vertices,offsets,counts};
    std::vector<Box> bounds(n);
    for(int i=nr;i<n;++i) bounds[i]=world_box(pose.data()+12*i,boxes+6*i);
    std::vector<int> near;
    // A previous nearest pair is only an upper-bound warm start.
    int warm_a[4]={-1,-1,-1,-1},warm_b[4]={-1,-1,-1,-1};
    for(int row=0;row<nq;++row) {
        std::copy(moving+12*nr*row,moving+12*nr*(row+1),pose.begin());
        double* out=result+row*5;int* pair=witness+row*8;
        std::fill(pair,pair+8,-1);double upper[4];
        for(int k=0;k<4;++k) upper[k]=out[k]=caps[k];
        V lo{1e9,1e9,1e9},hi{-1e9,-1e9,-1e9};double far2=0.;
        for(int i=0;i<nr;++i) {
            bounds[i]=world_box(pose.data()+12*i,boxes+6*i);
            const auto& b=bounds[i];double dd=0.;
            for(int j=0;j<3;++j) {
                lo[j]=std::min(lo[j],b.c[j]-b.h[j]);hi[j]=std::max(hi[j],b.c[j]+b.h[j]);
                const double off=std::abs(b.c[j]-origin[j])+b.h[j];dd+=off*off;
            }
            far2=std::max(far2,dd);
        }
        out[4]=radius-std::sqrt(far2);
        auto visit=[&](int i,int j,int k) {
            if(box_distance2(bounds[i],bounds[j])>=upper[k]*upper[k])return;
            if(upper[k]<=0.)return;
            auto d=distance(shapes,i,j,1.e-5);
            if(d[0]<out[k]) {out[k]=d[0];pair[k*2]=i;pair[k*2+1]=j;}
            if(d[1]<upper[k]) {upper[k]=d[1];warm_a[k]=i;warm_b[k]=j;}
        };
        for(int k=0;k<4;++k)if(warm_a[k]>=0)visit(warm_a[k],warm_b[k],k);
        const Box robot{mul(add(lo,hi),.5),mul(sub(hi,lo),.5)};
        double cap=std::max(caps[0],caps[1]);near.clear();
        for(int j=nr;j<n;++j)if(box_distance2(robot,bounds[j])<cap*cap)near.push_back(j);
        for(int j:near)for(int i=0;i<nr;++i)visit(i,j,groups[i]);
        for(int j=0;j<np;++j)visit(internal[3*j],internal[3*j+1],internal[3*j+2]);
    }
}
