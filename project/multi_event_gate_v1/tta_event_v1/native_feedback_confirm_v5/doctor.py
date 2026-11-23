"""Original GPU doctor with its certificate destination isolated to TTA outputs."""
from pathlib import Path
import ast,hashlib,inspect,json,sys
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_feedback_confirm_v5'
sys.path.insert(0,str(P/'deployment'));import gpu_doctor
O.mkdir(parents=True,exist_ok=True);assert not(O/'gpu-doctor-6.json').exists()
original=inspect.getsource(gpu_doctor.main)
needle='(container.WORK / ("gpu-doctor-" + str(args.gpu) + ".json")).write_text'
assert original.count(needle)==1
source=original.replace(needle,'(CERT_ROOT / ("gpu-doctor-" + str(args.gpu) + ".json")).write_text')
namespace=dict(gpu_doctor.__dict__,CERT_ROOT=O)
exec(compile(ast.parse(source),str(Path(__file__).resolve()),'exec'),namespace)
namespace['main']()
(O/'doctor_source_binding.json').write_text(json.dumps(dict(original_source_sha256=hashlib.sha256(Path(gpu_doctor.__file__).read_bytes()).hexdigest(),only_certificate_destination_changed=True,certificate=str(O/'gpu-doctor-6.json')))+'\n')
