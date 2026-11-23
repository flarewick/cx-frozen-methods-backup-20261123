"""Bounded, single-GPU validation for the user-owned container."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import user_container as container


SERVER_RULES = Path("/home/qiufw/服务器使用注意事项.md")
EXPECTED_RULES_SHA256 = "ba46e7a5a8ca832941742b3b2a764036e83236d95b4ee0226a73ce916dc5b4db"


def verify_server_rules():
    actual = hashlib.sha256(SERVER_RULES.read_bytes()).hexdigest()
    if actual != EXPECTED_RULES_SHA256:
        raise RuntimeError("Server rules changed; reread and update resource checks before GPU use")
    print("SERVER_RULES_VERIFIED", str(SERVER_RULES), actual, flush=True)


def query(arguments):
    return subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                          check=True, text=True, stdout=subprocess.PIPE, timeout=3).stdout


def gpu_snapshot(index):
    row = next(csv.reader(io.StringIO(query([
        "-i", str(index), "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu"
    ]))))
    return dict(index=int(row[0]), uuid=row[1].strip(), pci=row[2].strip().lower(),
                memory_mib=int(row[3]), sm_percent=int(row[4]))


def check_admission(info):
    verify_server_rules()
    from gpu_policy import inventory,require_admission
    fleet=inventory();require_admission(fleet,info['uuid'])
    print('GPU_FLEET_ADMISSION',json.dumps(fleet),flush=True)


def driver_options(info):
    cmd = ["--tmpfs", "/run", "--dir", "/run/cxdriver", "--dir", "/dev/dri"]
    libraries = subprocess.run(["ldconfig", "-p"], check=True, text=True, stdout=subprocess.PIPE).stdout
    for line in libraries.splitlines():
        if " => " not in line or "x86-64" not in line:
            continue
        name, path = line.split()[0], line.split(" => ")[1].strip()
        if any(part in name for part in ("libnvidia-", "libcuda.so", "libGLX_nvidia",
                                         "libEGL_nvidia", "libGLESv1_CM_nvidia",
                                         "libGLESv2_nvidia", "libnvcuvid")):
            cmd += ["--ro-bind", str(Path(path).resolve()), "/run/cxdriver/" + name]
    for source, target in [
        ("/usr/share/vulkan/icd.d/nvidia_icd.json", "/run/cxdriver/nvidia_icd.json"),
        ("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "/run/cxdriver/10_nvidia.json"),
    ]:
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        cmd += ["--ro-bind", source, target]
    nodes = ["/dev/nvidia" + str(info["index"]), "/dev/nvidiactl",
             "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]
    selected_pci = info["pci"].split(":")
    selected_pci = selected_pci[0][-4:] + ":" + ":".join(selected_pci[1:])
    for node in Path("/sys/class/drm").iterdir():
        if node.name.startswith(("card", "renderD")) and (node / "device").exists():
            if (node / "device").resolve().name.lower() == selected_pci:
                nodes.append("/dev/dri/" + node.name)
    for node in nodes:
        if Path(node).exists():
            cmd += ["--dev-bind", node, node]
    for key, value in {
        "CUDA_VISIBLE_DEVICES": info["uuid"],
        "NVIDIA_VISIBLE_DEVICES": info["uuid"],
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "LD_LIBRARY_PATH": "/run/cxdriver:/usr/local/cuda/lib64",
        "VK_ICD_FILENAMES": "/run/cxdriver/nvidia_icd.json",
        "VK_DRIVER_FILES": "/run/cxdriver/nvidia_icd.json",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/run/cxdriver/10_nvidia.json",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    }.items():
        cmd += ["--setenv", key, value]
    return cmd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    if not (container.WORK / "stage-bootstrap.ok").is_file():
        raise RuntimeError("Complete environment build and its CPU doctor before GPU validation")
    info = gpu_snapshot(args.gpu)
    check_admission(info)
    print("GPU_SMOKE_ADMISSION", info, flush=True)
    program = (
        "import torch,runpy,sys; "
        "assert torch.cuda.device_count()==1, 'Expected exactly one visible GPU'; "
        "sys.argv=['/opt/cx_deployment/doctor.py']; "
        "runpy.run_path('/opt/cx_deployment/doctor.py',run_name='__main__')"
    )
    cmd = container.command(["python", "-c", program], mount_project=True)
    split = cmd.index("--")
    cmd[split:split] = driver_options(info)
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        # A doctor is a short diagnostic; terminate it before the five-minute
        # sustained-workload thresholds could apply. Never create filler load.
        code = process.wait(timeout=240)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if code:
        raise SystemExit(code)
    certificate = {"uuid": info["uuid"], "index": info["index"],
                   "build_mtime_ns": (container.WORK / "stage-bootstrap.ok").stat().st_mtime_ns,
                   "tested_at_unix": time.time()}
    (container.WORK / ("gpu-doctor-" + str(args.gpu) + ".json")).write_text(json.dumps(certificate) + "\n")
    print("GPU_DOCTOR_PASSED", flush=True)


if __name__ == "__main__":
    main()
