#include <pybind11/pybind11.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/rigid_component.h>
#include <sapien/entity.h>
namespace py=pybind11;
py::list vector3(sapien::Vec3 const &v){py::list x;for(float q:{v.x,v.y,v.z})x.append(py::float_(double(q)));return x;}
py::list contacts(py::list original_contacts){
 py::list result;
 // Use the public Python API to obtain the contact list and entity names.
 // Only the public Contact/ContactPoint value structs are read natively;
 // never depend on private PhysxSystem layout from installed SDK headers.
 for(auto item:original_contacts){
  auto contact=py::cast<sapien::physx::Contact*>(item);
  py::dict row;py::list bodies,points;
  for(auto body:item.attr("bodies"))bodies.append(body.attr("entity").attr("name"));
  for(auto const &point:contact->points){py::dict p;p["position"]=vector3(point.position);p["normal"]=vector3(point.normal);p["impulse"]=vector3(point.impulse);p["separation"]=py::float_(double(point.separation));points.append(p);}
  row["bodies"]=bodies;row["points"]=points;result.append(row);
 }
 return result;
}
PYBIND11_MODULE(cx_benefit_contacts,m){m.def("contacts",&contacts);}
