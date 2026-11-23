#include <pybind11/pybind11.h>
#include <sapien/physx/physx_system.h>
namespace py=pybind11;
void step(sapien::physx::PhysxSystem* system){
 auto cpu=dynamic_cast<sapien::physx::PhysxSystemCpu*>(system);
 if(!cpu)throw std::runtime_error("Collection requires the original CPU PhysX system");
 py::gil_scoped_release release;
 cpu->sapien::physx::PhysxSystemCpu::step();
}
PYBIND11_MODULE(cx_collection_step,m){m.def("step",&step);}
