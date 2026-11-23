#include <pybind11/pybind11.h>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace py=pybind11;
struct Token{std::string text;double number=0;bool numeric=false;};
struct Capture{
 std::vector<Token> tokens;std::unordered_map<std::string,std::string> strings;std::unordered_set<PyObject*> active;py::object quote;
 Capture():quote(py::module_::import("json.encoder").attr("encode_basestring_ascii")){tokens.reserve(65536);}
 void text(std::string s){tokens.push_back({std::move(s),0,false});}
 void string(py::handle obj){
  py::object bytes=py::reinterpret_steal<py::object>(PyUnicode_AsEncodedString(obj.ptr(),"utf-8","surrogatepass"));if(!bytes)throw py::error_already_set();
  std::string key=py::cast<std::string>(bytes);auto found=strings.find(key);
  if(found==strings.end())found=strings.emplace(key,py::cast<std::string>(quote(obj))).first;text(found->second);
 }
 void value(py::handle obj,int depth=0){
  if(depth>512)throw py::value_error("Record nesting exceeds bounded schema");
  if(obj.is_none()){text("null");return;}if(PyBool_Check(obj.ptr())){text(obj.ptr()==Py_True?"true":"false");return;}
  if(PyFloat_Check(obj.ptr())){double x=PyFloat_AsDouble(obj.ptr());if(!std::isfinite(x))throw py::value_error("Out of range float values are not JSON compliant");tokens.push_back({{},x,true});return;}
  if(PyLong_Check(obj.ptr())){text(py::cast<std::string>(py::str(obj)));return;}
  if(PyUnicode_Check(obj.ptr())){string(obj);return;}
  bool dict=PyDict_Check(obj.ptr()),list=PyList_Check(obj.ptr()),tuple=PyTuple_Check(obj.ptr());
  if(!dict&&!list&&!tuple)throw py::type_error("Unsupported record value type");
  if(!active.insert(obj.ptr()).second)throw py::value_error("Circular reference detected");
  if(dict){text("{");Py_ssize_t pos=0;PyObject *k,*v;bool first=true;
   while(PyDict_Next(obj.ptr(),&pos,&k,&v)){if(!PyUnicode_Check(k))throw py::type_error("Record keys must be strings");if(!first)text(", ");first=false;string(py::handle(k));text(": ");value(py::handle(v),depth+1);}text("}");
  }else{text("[");Py_ssize_t n=list?PyList_GET_SIZE(obj.ptr()):PyTuple_GET_SIZE(obj.ptr());for(Py_ssize_t i=0;i<n;i++){if(i)text(", ");value(py::handle(list?PyList_GET_ITEM(obj.ptr(),i):PyTuple_GET_ITEM(obj.ptr(),i)),depth+1);}text("]");}
  active.erase(obj.ptr());
 }
};
py::str dumps(py::handle record){
 Capture c;c.value(record);std::string out;
 {py::gil_scoped_release release;out.reserve(c.tokens.size()*10);
  for(auto const &token:c.tokens){if(!token.numeric){out+=token.text;continue;}char buffer[64];auto result=std::to_chars(buffer,buffer+64,token.number,std::chars_format::general,std::numeric_limits<double>::max_digits10);if(result.ec!=std::errc{})throw std::runtime_error("Float encoding failed");std::string_view s(buffer,result.ptr-buffer);out.append(s);if(s.find_first_of(".eE")==std::string_view::npos)out+=".0";}
 }
 return py::str(out);
}
PYBIND11_MODULE(cx_benefit_json,m){m.def("dumps",&dumps);}
