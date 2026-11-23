#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/physx/articulation_link_component.h>
#include <sapien/entity.h>
#include <sapien/sapien_renderer/camera_component.h>
#include <sapien/sapien_renderer/sapien_renderer_system.h>
#include <mutex>
#include <cmath>
#include <cstdio>
namespace py=pybind11;
using sapien::physx::PhysxArticulation;
using sapien::physx::PhysxArticulationJoint;
using sapien::physx::PhysxSystem;
using Joint=PhysxArticulationJoint*;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
using F=py::array_t<float,py::array::c_style|py::array::forcecast>;

py::tuple control(PhysxArticulation* left,PhysxArticulation* right,
 std::vector<Joint> la,std::vector<Joint> ra,std::vector<Joint> lg,std::vector<Joint> rg,
 PhysxSystem* physics,py::function check_success,
 D lp,D lv,D rp,D rv,F lgr,F rgr,D lintent,D rintent,
 int ln,int rn,bool lf,bool rf,double lvalue,double rvalue) {
 if(!left||!right||!physics)throw std::runtime_error("Null native physics reference");
 auto cpu=dynamic_cast<sapien::physx::PhysxSystemCpu*>(physics);
 if(ln<=0||rn<=0||!cpu)throw std::runtime_error("Expected positive CPU physics paths");
 auto lpa=lp.unchecked<2>(),lva=lv.unchecked<2>(),rpa=rp.unchecked<2>(),rva=rv.unchecked<2>();
 auto lga=lgr.unchecked<2>(),rga=rgr.unchecked<2>();auto li=lintent.unchecked<1>(),ri=rintent.unchecked<1>();
 if((lf&&(lp.shape(0)<ln||lv.shape(0)<ln||lp.shape(1)!=(int)la.size()||lv.shape(1)!=(int)la.size())) ||
    (rf&&(rp.shape(0)<rn||rv.shape(0)<rn||rp.shape(1)!=(int)ra.size()||rv.shape(1)!=(int)ra.size())) ||
    lgr.shape(0)!=ln||rgr.shape(0)!=rn||lgr.shape(1)!=(int)lg.size()*2||rgr.shape(1)!=(int)rg.size()*2||lintent.size()!=ln||rintent.size()!=rn)
   throw std::runtime_error("Native control shape mismatch");
 int i=0,j=0,ticks=0;bool success=false;
 {
 py::gil_scoped_release release;
 while(i<ln||j<rn){
   // The state inputs of gravity/Coriolis do not change between drive setters.
   // Preserve their exact force values, compute once before this physical tick.
   left->setQf(left->computePassiveForce(true,true));right->setQf(right->computePassiveForce(true,true));
   if(i<ln&&double(i)/ln<=double(j)/rn){
     if(lf)for(size_t k=0;k<la.size();++k){la[k]->setDriveTargetPosition(float(lpa(i,k)));la[k]->setDriveTargetVelocity(float(lva(i,k)));}
     for(size_t k=0;k<lg.size();++k){lg[k]->setDriveTargetPosition(lga(i,2*k));lg[k]->setDriveTargetVelocity(lga(i,2*k+1));}
     lvalue=li(i);++i;
   }
   if(j<rn&&double(j)/rn<=double(i)/ln){
     if(rf)for(size_t k=0;k<ra.size();++k){ra[k]->setDriveTargetPosition(float(rpa(j,k)));ra[k]->setDriveTargetVelocity(float(rva(j,k)));}
     for(size_t k=0;k<rg.size();++k){rg[k]->setDriveTargetPosition(rga(j,2*k));rg[k]->setDriveTargetVelocity(rga(j,2*k+1));}
     rvalue=ri(j);++j;
   }
   cpu->sapien::physx::PhysxSystemCpu::step();++ticks;
   {
     py::gil_scoped_acquire acquire;
     success=py::cast<bool>(check_success(lvalue,rvalue));
   }
   if(success)break;
 }
 }
 return py::make_tuple(success,i,j,ticks,lvalue,rvalue);
}

template<class T>T clip(T x,T low,T high){if(x<=low)return low;if(x>=high)return high;return x;}
py::tuple grippers(D commands,F initial,D factors,D biases,double scale0,double scale1){
 auto c=commands.unchecked<1>();auto start=initial.unchecked<1>();auto f=factors.unchecked<1>(),b=biases.unchecked<1>();
 int n=commands.size(),joints=initial.size();
 if(joints<1||factors.size()!=joints||biases.size()!=joints||scale0==scale1)throw std::runtime_error("Invalid original gripper calibration");
 F rows({n,2*joints});D intent(n);auto out=rows.mutable_unchecked<2>();auto goal=intent.mutable_unchecked<1>();
 std::vector<float> current(joints);for(int j=0;j<joints;++j)current[j]=start(j);
 {
 py::gil_scoped_release release;
 for(int i=0;i<n;++i){
   double command=clip(c(i),0.,1.);goal(i)=command;
   double real=clip((double(current[0])-scale0)/(scale1-scale0),0.,1.);
   if(command-real>.1)command=real+.1;
   double normalized=scale0+command*(scale1-scale0);
   for(int j=0;j<joints;++j){
     double target=normalized*f(j)+b(j);
     // Frozen NumPy1.26 value-based promotion: small np.float64 scalar minus
     // a float32 array remains float32, as do clip and multiplication by0.05.
     float position=float(target);float velocity=clip(position-current[j],-1.f,1.f)*.05f;
     out(i,2*j)=position;out(i,2*j+1)=velocity;current[j]=position;
   }
 }
 }
 return py::make_tuple(rows,intent);
}
// Keep Vulkan render operations serialized while releasing Python's GIL.
// Scene lifetime/global cache clearing is protected by the Python SceneGate.
std::mutex renderMutex;
void render_step(sapien::sapien_renderer::SapienRendererSystem* system){
 if(!system)throw std::runtime_error("Null render system");
 py::gil_scoped_release release;std::lock_guard<std::mutex> guard(renderMutex);
 system->step();
}
void take_pictures(std::vector<sapien::sapien_renderer::SapienRenderCameraComponent*> cameras){
 for(auto camera:cameras)if(!camera)throw std::runtime_error("Null camera reference");
 py::gil_scoped_release release;std::lock_guard<std::mutex> guard(renderMutex);
 for(auto camera:cameras)camera->takePicture();
}
PYBIND11_MODULE(cx_native_generic_control,m){m.def("control",&control);m.def("grippers",&grippers);m.def("render_step",&render_step);m.def("take_pictures",&take_pictures);}

