"""User-owned Ubuntu/CUDA container for hosts without Docker or sudo.

Uses the installed bubblewrap; does not change host packages or kernel settings.
The frozen Dockerfile/bootstrap and transferred sources remain authoritative.
"""
import argparse
import hashlib
import json
import os
import pwd
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORK = HERE / ".work"
ROOTFS = WORK / "rootfs"
CANONICAL = Path("/data/cj/robo_pi/250010033")
ARTIFACTS = PROJECT.parent / "artifacts"


def environment():
    env = {
        "PATH": "/opt/cx_venv/bin:/opt/python311/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "LANG": "C.UTF-8",
        "PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "DEBIAN_FRONTEND": "noninteractive",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MAX_JOBS": "2",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def command(inner, writable=False, mount_project=False):
    if not (ROOTFS / ".cx-base-extracted").is_file():
        raise FileNotFoundError("CUDA base image is not extracted")
    temporary = WORK / ("build-tmp" if writable else "run-tmp")
    temporary.mkdir(exist_ok=True)
    cmd = [
        "bwrap", "--unshare-user", "--uid", "0" if writable else str(os.getuid()),
        "--gid", "0" if writable else str(os.getgid()),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent", "--new-session", "--clearenv",
        "--bind" if writable else "--ro-bind", str(ROOTFS), "/",
        "--proc", "/proc", "--dev", "/dev", "--ro-bind", "/sys", "/sys",
        "--bind", str(temporary), "/tmp",
        "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf",
        "--ro-bind", "/etc/hosts", "/etc/hosts",
    ]
    env = environment()
    if not writable:
        env["HOME"] = "/tmp"
        env["USER"] = env["LOGNAME"] = pwd.getpwuid(os.getuid()).pw_name
    if mount_project:
        canonical_project = CANONICAL / "VLA-Adapter/cx_new_direction_guide_TTA"
        cmd += ["--bind", str(PROJECT), str(canonical_project)]
        for rel in ("RoboTwin/assets", "checkpoints/pi05_motus_robotwin2/motus_robotwin2/1000000"):
            source = ARTIFACTS / "rootfs/qfw/robotwin" / rel
            if not source.is_dir():
                raise FileNotFoundError(source)
            cmd += ["--ro-bind", str(source), str(CANONICAL / "qfw/robotwin" / rel)]
        cmd += ["--chdir", str(canonical_project)]
    for key, value in env.items():
        cmd += ["--setenv", key, value]
    return cmd + ["--", *inner]


def execute(inner, writable=False, mount_project=False):
    result = subprocess.run(command(inner, writable, mount_project))
    if result.returncode:
        raise RuntimeError("User container command exited with status " + str(result.returncode))


def stage(name, inner):
    marker = WORK / ("stage-" + name + ".ok")
    if marker.exists():
        print("STAGE_ALREADY_DONE", name, flush=True)
        return
    print("STAGE_START", name, flush=True)
    execute(inner, writable=True)
    marker.write_text("completed\n")
    print("STAGE_DONE", name, flush=True)


def build():
    if not (ROOTFS / ".cx-base-extracted").exists():
        raise FileNotFoundError("CUDA base image is not extracted")
    # Provision this optional shared-font directory under the single mapped UID.
    # This avoids a package postinst requesting an unmapped staff GID.
    (ROOTFS / "usr/local/share/fonts").mkdir(parents=True, exist_ok=True)
    sources = ROOTFS / "etc/apt/sources.list"
    source_text = sources.read_text()
    source_text = source_text.replace("http://archive.ubuntu.com/ubuntu", "https://mirrors.tuna.tsinghua.edu.cn/ubuntu")
    source_text = source_text.replace("http://security.ubuntu.com/ubuntu", "https://mirrors.tuna.tsinghua.edu.cn/ubuntu")
    sources.write_text(source_text)
    (ROOTFS / "etc/apt/apt.conf.d/99cx-userns").write_text(
        'APT::Sandbox::User "root";\nAcquire::Retries "2";\n'
    )
    # Avoid starting services inside this single-user build container.
    policy = ROOTFS / "usr/sbin/policy-rc.d"
    policy.write_text("#!/bin/sh\nexit 101\n")
    policy.chmod(0o755)
    stage("system-packages", ["bash", "-euc", """apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates curl git pkg-config rsync ffmpeg vulkan-tools libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncursesw5-dev uuid-dev libgdbm-dev libnss3-dev libgl1 libegl1 libglvnd0 libgles2 libopengl0 libx11-6 libglib2.0-0 libvulkan1 libglfw3 cuda-nvcc-12-6=12.6.85-1 cuda-cudart-dev-12-6=12.6.77-1 cuda-cccl-12-6=12.6.77-1
rm -rf /var/lib/apt/lists/*
"""])
    py = json.loads((HERE / "python-source.json").read_text())
    stage("python3119", ["bash", "-euc", f"""curl -fL --retry 3 --connect-timeout 20 --max-time 240 {py['url']} -o /tmp/python.tar.xz
echo '{py['sha256']}  /tmp/python.tar.xz' | sha256sum -c -
tar -xJf /tmp/python.tar.xz -C /tmp
cd /tmp/Python-3.11.9
LDFLAGS='-Wl,-rpath,/opt/python311/lib' ./configure --prefix=/opt/python311 --enable-shared
make -j4
make install
/opt/python311/bin/python3.11 -m venv /opt/cx_venv
"""])
    destination = ROOTFS / "opt/cx_deployment"
    destination.mkdir(parents=True, exist_ok=True)
    context = ARTIFACTS / "docker_context"
    for name in ("requirements.runtime.txt", "environment.json", "install_overlay.py", "doctor.py", "bootstrap.sh"):
        shutil.copy2(HERE / name, destination / name)
    marker = WORK / "stage-runtime-sources.ok"
    if not marker.exists():
        print("STAGE_START runtime-sources", flush=True)
        with tarfile.open(context / "runtime.tar.gz") as archive:
            archive.extractall(ROOTFS, filter="data")
        marker.write_text("completed\n")
        print("STAGE_DONE runtime-sources", flush=True)
    for target in (
        "data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA",
        "data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/assets",
        "data/cj/robo_pi/250010033/qfw/robotwin/checkpoints/pi05_motus_robotwin2/motus_robotwin2/1000000",
    ):
        (ROOTFS / target).mkdir(parents=True, exist_ok=True)
    bootstrap_was_ready = (WORK / "stage-bootstrap.ok").exists()
    stage("bootstrap", ["bash", "/opt/cx_deployment/bootstrap.sh"])
    if bootstrap_was_ready:
        # Deployment-only support fixes need not recompile unchanged CuRobo.
        execute(["python", "/opt/cx_deployment/install_overlay.py"], writable=True)
        execute(["python", "/opt/cx_deployment/doctor.py", "--cpu"], writable=True)
    print("USER_CONTAINER_BUILD_PASSED", flush=True)


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("build", "exec", "doctor-cpu", "preflight", "preflight-tta"))
    parser.add_argument("inner", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "doctor-cpu":
        execute(["python", "/opt/cx_deployment/doctor.py", "--cpu"], mount_project=True)
    elif args.action in ("preflight", "preflight-tta"):
        import run as frozen_run
        frozen_args = argparse.Namespace(artifacts=ARTIFACTS, entry=args.action, gpu=0, image="cx-user-rootfs")
        docker_command = frozen_run.command(frozen_args, args.inner)
        inner = docker_command[docker_command.index(frozen_args.image) + 1:]
        execute(inner, mount_project=True)
    else:
        inner = args.inner[1:] if args.inner[:1] == ["--"] else args.inner
        if not inner:
            parser.error("exec requires a command")
        execute(inner, mount_project=True)


if __name__ == "__main__":
    main()
