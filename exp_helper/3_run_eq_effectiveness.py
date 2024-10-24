import subprocess
import shutil
import os
import copy

POLYJUICE_ROOT_PATH = "/root/polyjuice/"
TMP_FUZZ_ROOT = "/tmp/fuzz_report"

OUTPUT_DIR = "/root/3_eq_effectiveness/"

NOISE_REDIRECTION = subprocess.STDOUT


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subprocess.run(["rm", "-rf", TMP_FUZZ_ROOT + "*"])

    new_env = copy.deepcopy(os.environ)
    new_env["OUTPUT_DIR"] = OUTPUT_DIR
    new_env["SAVE_EQ_IS_WRONG"] = "false"
    new_env["EQ_HELPER_PATH"] = "/root/polyjuice/libdl_compiler_fuzzer_helper.so"
    new_env["TVM_LOG_DEBUG"] = "DEFAULT=0"
    new_env["TVM_HOME"] = "/root/compilers/apache-tvm-src"
    new_env["PYTHONPATH"] = "/root/compilers/apache-tvm-src/python:/root/polyjuice"
    new_env["LD_LIBRARY_PATH"] = "/root/compilers/apache-tvm-src/new_build"

    ablation_study_script_path = POLYJUICE_ROOT_PATH + "nnsmith/cli/ablation_exp_run.py"
    p = subprocess.run(
        ["python3", ablation_study_script_path, "mgen.rank_choices=[3,3]", "model.type=onnx", "backend.type=tvm",
         "backend.target=cpu", f"fuzz.root={TMP_FUZZ_ROOT}", "debug.viz=true", "mgen.max_nodes=10"],
        # stdout=NOISE_REDIRECTION, stderr=NOISE_REDIRECTION,
        env=new_env, cwd=POLYJUICE_ROOT_PATH)
    if p.returncode != 0:
        print(f"ablation study failed. return code: {p.returncode}")
    else:
        print("ablation study runs finished successfully")
    print("processing results...")

    process_diff_script_path = os.path.join(POLYJUICE_ROOT_PATH, "exp_helper/process_diff_effect.py")
    ablation_exp_csv_path = os.path.join(OUTPUT_DIR, "ablation_exp.csv")

    p = subprocess.run(
        ["python3", process_diff_script_path, ablation_exp_csv_path],
        # stdout=NOISE_REDIRECTION, stderr=NOISE_REDIRECTION,
        env=new_env, cwd=POLYJUICE_ROOT_PATH)
    if p.returncode != 0:
        print(f"processing results failed. return code: {p.returncode}")
    else:
        print("processing results finished successfully")

    print(f"the result should be in {os.path.join(OUTPUT_DIR, 'table4.csv')}")


if __name__ == '__main__':
    main()
