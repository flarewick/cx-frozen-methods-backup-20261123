#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/physx/articulation_link_component.h>
#include <cmath>
#include <cstdint>
namespace py=pybind11;
using Joint=sapien::physx::PhysxArticulationJoint*;
using Doubles=py::array_t<double,py::array::c_style|py::array::forcecast>;
py::dict capture(std::vector<Joint> arms,std::vector<Joint> fingers,std::vector<std::int64_t> counts,Doubles logical){
 if(arms.size()!=12||counts.size()!=2||counts[0]<0||counts[1]<0||counts[0]+counts[1]!=(std::int64_t)fingers.size()||logical.ndim()!=1||logical.size()!=2)
  throw std::runtime_error("Original six-per-arm/two-gripper telemetry shape required");
 for(auto* j:arms)if(!j)throw std::runtime_error("Missing arm joint");
 for(auto* j:fingers)if(!j)throw std::runtime_error("Missing gripper joint");
 py::array_t<double> arm_p(12),arm_v(12),grip_p(fingers.size()),grip_v(fingers.size()),command(2);
 py::array_t<std::int64_t> joint_counts(2);
 auto ap=arm_p.mutable_unchecked<1>(),av=arm_v.mutable_unchecked<1>(),gp=grip_p.mutable_unchecked<1>(),gv=grip_v.mutable_unchecked<1>(),cmd=command.mutable_unchecked<1>();
 auto count=joint_counts.mutable_unchecked<1>();auto input=logical.unchecked<1>();
 for(int i=0;i<2;++i){if(!std::isfinite(input(i)))throw std::runtime_error("Nonfinite gripper command");cmd(i)=input(i);count(i)=counts[i];}
 {
 py::gil_scoped_release release;
 auto read=[](Joint j){auto p=j->getDriveTargetPosition();auto v=j->getDriveTargetVelocity();
  if(p.size()!=1||v.size()!=1||!std::isfinite(p(0))||!std::isfinite(v(0)))throw std::runtime_error("Expected one finite commanded drive DOF");
  return std::make_pair(double(p(0)),double(v(0)));};
 for(std::size_t i=0;i<arms.size();++i){auto values=read(arms[i]);ap(i)=values.first;av(i)=values.second;}
 for(std::size_t i=0;i<fingers.size();++i){auto values=read(fingers[i]);gp(i)=values.first;gv(i)=values.second;}
 }
 py::dict result;result["arm_position_target"]=arm_p;result["arm_velocity_target"]=arm_v;
 result["gripper_position_target"]=grip_p;result["gripper_velocity_target"]=grip_v;
 result["gripper_logical_command"]=command;result["gripper_joint_counts"]=joint_counts;return result;
}
PYBIND11_MODULE(cx_tta_command_telemetry,m){m.def("capture",&capture);}
