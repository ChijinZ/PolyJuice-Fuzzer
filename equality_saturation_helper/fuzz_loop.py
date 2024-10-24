# this script is used as a parent process for mitigating the memory leak of DL frameworks/compilers
import logging
import subprocess
import os
import argparse
import time
import shutil
from subprocess import TimeoutExpired
import signal
import psutil

TIME_FOR_SUBPROCESS = 1850


# PARALLEL = 5


def clean_per_circle():
    inductor_cache_path = "/dev/shm/torchinductor"
    # shm_usage = psutil.disk_usage('/dev/shm')
    # if shm_usage.free > 10 * 1024 * 1024 * 1024:
    #     os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_path
    # else:
    #     if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
    #         del os.environ["TORCHINDUCTOR_CACHE_DIR"]
    if os.path.exists("/tmp/torchinductor_root"):
        shutil.rmtree("/tmp/torchinductor_root")
    if os.path.exists("/root/.cache/hidet"):
        shutil.rmtree("/root/.cache/hidet")
    if os.path.exists(inductor_cache_path):
        shutil.rmtree(inductor_cache_path)


def kill_a_proc(proc: subprocess.Popen) -> int:
    return_code = proc.poll()
    start_time = time.time()
    while return_code is None:  # it is still alive
        if time.time() - start_time < 2:
            proc.terminate()
        else:
            proc.send_signal(signal.SIGKILL)
        return_code = proc.poll()
    print(f"spend {time.time() - start_time} second to kill the proc")
    return return_code


def main():
    parser = argparse.ArgumentParser(description="fuzzing loop helper")
    parser.add_argument("-o", "--output_path", type=str, help="directory path of output path", required=True)
    parser.add_argument("-s", "--script_path", type=str, help="path of fuzzing script", required=True)
    parser.add_argument("-m", "--model_type", type=str, help="torch/onnx/tensorflow", required=True)
    parser.add_argument("-b", "--backend_type", type=str, help="backend", required=True)
    parser.add_argument("-p", "--parallel", type=int, help="how many cores", required=True)
    parser.add_argument("-t", "--backend_target", type=str, help="cpu/gpu", default="cpu")
    parser.add_argument("-r", "--running-time", type=int, help="time for running (s), -1 for never stop, default -1",
                        default=-1)
    parser.add_argument("-a", "--appended_args", type=str, help="appended arguments for the script", default="")
    args = parser.parse_args()
    script_path = args.script_path
    output_path = args.output_path
    model_type = args.model_type
    backend_type = args.backend_type
    parallel = args.parallel
    backend_target = args.backend_target
    running_time = args.running_time
    appended_args = [i for i in args.appended_args.split(" ")] if args.appended_args != "" else []
    if os.path.exists(output_path):
        logging.error(f"{output_path} already exists, please remove it before we can start the fuzzing process")
        return

    os.makedirs(output_path, exist_ok=True)

    time_for_start_all = time.time()
    session_id = 0
    while True:
        clean_per_circle()
        if running_time != -1 and time.time() - time_for_start_all > running_time:
            print("fuzzing is over due to timeout")
            break

        print(f"new cycle, session id start with {session_id}, parallel num is {parallel}")
        procs = []
        start_time = time.time()
        for _ in range(parallel):
            fuzzer_output_dir = os.path.join(output_path, f"fuzz_report_{session_id}")
            log_output_path = os.path.join(output_path, f"fuzz_log_{session_id}")
            with open(log_output_path, "w+") as f:
                cmd = ["python3", script_path, f"fuzz.time={TIME_FOR_SUBPROCESS - 50}s", "cmp.oracle=null",
                       "cmp.oracle=null",
                       f"model.type={model_type}",
                       f"backend.type={backend_type}", f"backend.target={backend_target}",
                       f"fuzz.root={fuzzer_output_dir}",
                       "debug.viz=true", "mgen.max_nodes=10"] + appended_args
                session_id += 1
                proc = subprocess.Popen(cmd, stdout=f, stderr=f)
                procs.append(proc)

        print(f"wait for the termination of {len(procs)} processes, time: {time.asctime(time.localtime(time.time()))}")
        used = [False for _ in range(len(procs))]
        jump_out = False
        while True:
            if used.count(True) == len(used):
                break

            if (time.time() - start_time) > TIME_FOR_SUBPROCESS:
                print(f"time to terminate. time: {time.asctime(time.localtime(time.time()))}")

            for index, proc in enumerate(procs):
                if used[index]:
                    continue

                if jump_out:
                    return_code = kill_a_proc(proc)
                    print(f"process {index} has been terminated due to jump out; exit code: {return_code}")
                    used[index] = True

                try:
                    code = proc.wait(5)
                    print(f"process {index} has been normally terminated; exit code: {code}")
                    used[index] = True
                except TimeoutExpired as _:
                    if (time.time() - start_time) > TIME_FOR_SUBPROCESS and proc.poll() is None:
                        code = kill_a_proc(proc)
                        print(f"process {index} has been force terminated; exit code: {code}")
                        used[index] = True
                except BaseException as e:
                    print(f"terminated by {e}, start killing the processes")
                    jump_out = True

        print("all processes are terminated")
        if jump_out:
            break
    print("all done")


if __name__ == '__main__':
    main()
