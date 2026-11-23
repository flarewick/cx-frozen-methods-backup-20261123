#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/physx/articulation_link_component.h>
#include <sapien/physx/physx_system.h>
namespace py=pybind11;
using A=sapien::physx::PhysxArticulation;
using J=sapien::physx::PhysxArticulationJoint*;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
using F=py::array_t<float,py::array::c_style|py::array::forcecast>;
py::tuple control(A* left,A* right,std::vector<J> la,std::vector<J> ra,std::vector<J> lg,std::vector<J> rg,
 sapien::physx::PhysxSystem* system,D lp,D lv,D rp,D rv,F gl,F gr,D li,D ri,int begin,int end,double lvalue,double rvalue,bool record){
 auto cpu=dynamic_cast<sapien::physx::PhysxSystemCpu*>(system);
 if(!cpu||!left||!right||la.size()!=6||ra.size()!=6||begin<0||end<begin)throw std::runtime_error("Unexpected collection control contract");
 if(lp.ndim()!=2||lv.ndim()!=2||rp.ndim()!=2||rv.ndim()!=2||gl.ndim()!=2||gr.ndim()!=2||li.ndim()!=1||ri.ndim()!=1||lp.shape(1)!=6||lv.shape(1)!=6||rp.shape(1)!=6||rv.shape(1)!=6||lp.shape(0)!=lv.shape(0)||rp.shape(0)!=rv.shape(0)||gl.shape(1)!=(int)lg.size()*2||gr.shape(1)!=(int)rg.size()*2||li.size()!=gl.shape(0)||ri.size()!=gr.shape(0))throw std::runtime_error("Collection input shape mismatch");
 auto lpa=lp.unchecked<2>(),lva=lv.unchecked<2>(),rpa=rp.unchecked<2>(),rva=rv.unchecked<2>();auto gla=gl.unchecked<2>(),gra=gr.unchecked<2>();auto lia=li.unchecked<1>(),ria=ri.unchecked<1>();
 py::array_t<float> dense({record?end-begin:0,14});auto out=dense.mutable_unchecked<2>();
 {
 py::gil_scoped_release release;
 auto force=[&](){left->setQf(left->computePassiveForce(true,true));right->setQf(right->computePassiveForce(true,true));};
 for(int i=begin;i<end;++i){
  if(i<lp.shape(0)){force();for(int k=0;k<6;++k){la[k]->setDriveTargetPosition(float(lpa(i,k)));la[k]->setDriveTargetVelocity(float(lva(i,k)));}}
  if(i<gl.shape(0)){force();for(size_t k=0;k<lg.size();++k){lg[k]->setDriveTargetPosition(gla(i,2*k));lg[k]->setDriveTargetVelocity(gla(i,2*k+1));}lvalue=lia(i);}
  if(i<rp.shape(0)){force();for(int k=0;k<6;++k){ra[k]->setDriveTargetPosition(float(rpa(i,k)));ra[k]->setDriveTargetVelocity(float(rva(i,k)));}}
  if(i<gr.shape(0)){force();for(size_t k=0;k<rg.size();++k){rg[k]->setDriveTargetPosition(gra(i,2*k));rg[k]->setDriveTargetVelocity(gra(i,2*k+1));}rvalue=ria(i);}
  if(record){for(int k=0;k<6;++k){out(i-begin,k)=la[k]->getDriveTargetPosition()(0);out(i-begin,7+k)=ra[k]->getDriveTargetPosition()(0);}out(i-begin,6)=float(lvalue);out(i-begin,13)=float(rvalue);}
  cpu->sapien::physx::PhysxSystemCpu::step();
 }
 }
 return py::make_tuple(dense,lvalue,rvalue);
}
PYBIND11_MODULE(cx_collection_control,m){m.def("control",&control);}
