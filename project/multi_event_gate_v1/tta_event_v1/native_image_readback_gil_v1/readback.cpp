#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/sapien_renderer/camera_component.h>
#include <thread>
#include <chrono>
namespace py=pybind11;
using Camera=sapien::sapien_renderer::SapienRenderCameraComponent;
py::array wrap(sapien::CpuArrayHandle const &image,py::object owner) {
  if(image.type!="f4" || image.shape.size()!=3 || image.shape[2]!=4 || image.strides.size()!=3 || !image.ptr)
    throw std::runtime_error("Require actual float32 HWC RGBA buffer");
  std::vector<py::ssize_t> shape(image.shape.begin(),image.shape.end()),strides(image.strides.begin(),image.strides.end());
  return py::array(py::dtype::of<float>(),shape,strides,image.ptr,owner);
}
PYBIND11_MODULE(cx_tta_original_image_readback,m) {
  m.def("registered_camera_type",[](){
    auto *info=py::detail::get_type_info(typeid(Camera),false);
    if(!info)throw std::runtime_error("Original SAPIEN Camera ABI not registered");
    return py::reinterpret_borrow<py::object>((PyObject*)info->type);
  });
  m.def("read",[](std::vector<Camera*> cameras){
    if(cameras.empty())throw py::value_error("Require original captured cameras");
    std::vector<sapien::CpuArrayHandle> images;images.reserve(cameras.size());
    {
      py::gil_scoped_release release;
      for(auto *camera:cameras){
        if(!camera)throw std::runtime_error("Null camera");
        images.push_back(camera->getImage("Color"));
      }
    }
    py::list result;
    for(size_t i=0;i<images.size();++i)result.append(wrap(images[i],py::cast(cameras[i],py::return_value_policy::reference)));
    return result;
  });
  m.def("CPU_view_fixture",[](py::array value){
    auto b=value.request();
    if(b.format!=py::format_descriptor<float>::format())throw py::value_error("Float32 fixture required");
    sapien::CpuArrayHandle image;
    image.shape.assign(b.shape.begin(),b.shape.end());image.strides.assign(b.strides.begin(),b.strides.end());image.type="f4";image.ptr=b.ptr;
    return wrap(image,value);
  });
  m.def("CPU_wait_fixture",[](int milliseconds){
    if(milliseconds<0 || milliseconds>1000)throw py::value_error("Bounded CPU fixture only");
    py::gil_scoped_release release;std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  });
}
