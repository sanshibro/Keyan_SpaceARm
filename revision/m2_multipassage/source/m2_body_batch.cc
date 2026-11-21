// Rigid-body pose batch using the same nominal support certificate as ConvexMap.
#include "m2_self.cc"
#include <map>

extern "C" void rigid_batch(int n, int nr, const int* type, const double* size,
 const double* local, const double* boxes, const int* offset, const int* count,
 const double* vertices, int nq, const double* states, double cap, double threshold, double* result) {
 struct Bounds { V c,h; };
 auto bounds=[&](const double* p, int i) {
    Bounds b{{p[0],p[1],p[2]},{}};
    for(int j=0;j<3;++j) for(int k=0;k<3;++k) {
        b.c[j]+=p[3+j*3+k]*boxes[i*6+k];
        b.h[j]+=std::abs(p[3+j*3+k])*boxes[i*6+3+k];
    }
    return b;
 };
 struct Rotated { std::vector<double> pose; std::vector<Bounds> geom; std::vector<std::vector<V>> points; V lo,hi; };
 std::map<double,Rotated> cache;
 std::vector<Bounds> ob(n-nr);
 for(int j=nr;j<n;++j)ob[j-nr]=bounds(local+12*j,j);
 std::vector<double> pose(local,local+n*12);
 Shapes shapes{type,size,pose.data(),vertices,offset,count};
 std::vector<std::vector<V>> base_points(nr);
 if(threshold>0.) {
    static const int dirs[][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{1,-1,-1},{-1,1,1}};
    for(int i=0;i<nr;++i) {
        std::vector<V> ends;
        for(auto& a:dirs) {
            V v=shapes.raw_support(i,V{double(a[0]),double(a[1]),double(a[2])});bool duplicate=false;
            for(const auto& w:ends){V d=sub(v,w);if(dot(d,d)<1e-18){duplicate=true;break;}}
            if(!duplicate)ends.push_back(v);
        }
        V mid{};for(const auto& point:ends)mid=add(mid,point);mid=mul(mid,1./ends.size());
        auto& points=base_points[i];points.push_back(mid);points.insert(points.end(),ends.begin(),ends.end());
        if(norm(bounds(local+12*i,i).h)>.05)
            for(const auto& point:ends)points.push_back(mul(add(mid,point),.5));
    }
 }
 // Distances from actual contained robot points to the map primitive are
 // feasible upper witnesses. When below the threshold they rule a pose out.
 auto obstacle_d2=[&](const V& p,int j) {
    const double* op=local+12*j;const double* sz=size+3*j;
    V r{p[0]-op[0],p[1]-op[1],p[2]-op[2]};
    if(type[j]==6) { // ConvexMap registers snapshot boxes on world axes.
        double d2=0.;for(int a=0;a<3;++a){double t=std::max(0.,std::abs(r[a])-sz[a]);d2+=t*t;}return d2;
    }
    double axial=r[0]*op[5]+r[1]*op[8]+r[2]*op[11];
    double radial=std::sqrt(std::max(0.,dot(r,r)-axial*axial));
    double e=std::max(0.,radial-sz[0]),h=std::max(0.,std::abs(axial)-sz[1]);return e*e+h*h;
 };
 struct Pair { double squared; int a,b; };
 std::vector<int> near;
 std::vector<Pair> pairs;
 for(int k=0;k<nq;++k) {
    const double* q=states+k*6;
    auto [it,inserted]=cache.try_emplace(q[3]);
    auto& rot=it->second;
    if(inserted) {
        rot.pose.resize(12*nr);rot.geom.resize(nr);rot.points.resize(nr);
        rot.lo={1e100,1e100,1e100};rot.hi={-1e100,-1e100,-1e100};
        double c=std::cos(q[3]),s=std::sin(q[3]);
        for(int i=0;i<nr;++i) {
            const double* src=local+12*i;double* p=rot.pose.data()+12*i;
            p[0]=c*src[0]-s*src[1];p[1]=s*src[0]+c*src[1];p[2]=src[2];
            for(int col=0;col<3;++col) {
                p[3+col]=c*src[3+col]-s*src[6+col];
                p[6+col]=s*src[3+col]+c*src[6+col];p[9+col]=src[9+col];
            }
            const auto b=bounds(p,i);rot.geom[i]=b;
            if(threshold>0) {
                auto& points=rot.points[i];points.reserve(base_points[i].size());
                for(const auto& v:base_points[i])points.push_back({c*v[0]-s*v[1],s*v[0]+c*v[1],v[2]});
            }
            for(int j=0;j<3;++j) {
                rot.lo[j]=std::min(rot.lo[j],b.c[j]-b.h[j]);rot.hi[j]=std::max(rot.hi[j],b.c[j]+b.h[j]);
            }
        }
    }
    std::copy(rot.pose.begin(),rot.pose.end(),pose.begin());
    near.clear();pairs.clear();double upper=cap,lower=cap;
    for(int j=0;j<n-nr;++j) {
        const auto& b=ob[j];bool touches=true;
        for(int a=0;a<3;++a)
            if(b.c[a]-b.h[a]>q[a]+rot.hi[a]+cap || b.c[a]+b.h[a]<q[a]+rot.lo[a]-cap) {touches=false;break;}
        if(touches)near.push_back(j);
    }
    bool ruled_out=false;const double threshsq=std::pow(std::max(0.,threshold-1e-8),2);
    for(int i=0;i<nr && !ruled_out;++i) {
        for(int a=0;a<3;++a)pose[12*i+a]+=q[a];
        const auto& robot=rot.geom[i];
        for(int j:near) {
            double ds=0.;
            for(int a=0;a<3;++a) {
                double e=std::max(0.,std::abs(robot.c[a]+q[a]-ob[j].c[a])-robot.h[a]-ob[j].h[a]);
                ds+=e*e;
            }
            if(ds<=cap*cap)pairs.push_back({ds,i,nr+j});
            if(threshold>0. && ds<=threshsq) {
                for(const auto& off:rot.points[i]) {
                    V pt{off[0]+q[0],off[1]+q[1],off[2]+q[2]};
                    if(obstacle_d2(pt,nr+j)<threshsq){ruled_out=true;break;}
                }
                if(ruled_out)break;
            }
        }
    }
    if(ruled_out){result[k]=-INFINITY;continue;}
    std::sort(pairs.begin(),pairs.end(),[](const Pair& a,const Pair& b){return a.squared<b.squared;});
    for(const auto& pair:pairs) {
        if(pair.squared>upper*upper)break;
        const auto d=distance(shapes,pair.a,pair.b,1e-5);
        lower=std::min(lower,d[0]);upper=std::min(upper,d[1]);
        if(threshold>0. && upper<threshold-1e-8){lower=-INFINITY;break;}
        if(upper<=0)break;
    }
    result[k]=lower;
 }
}
