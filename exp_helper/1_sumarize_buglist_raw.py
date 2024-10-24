import os
import pandas as pd

RAW_BUGLIST_PATH = "/root/raw_data/raw_data/tab3_raw_data/polyjuice_buglist.csv"
OUTPUT_DIR_PATH = "/root/raw_data/raw_data_results"


def main():
    with open(RAW_BUGLIST_PATH, "r") as f:
        buglist = f.readlines()

    res_dic = {
        "Torch Inductor": {"reported": 0, "confirmed": 0, "fixed": 0},
        "TensorRT": {"reported": 0, "confirmed": 0, "fixed": 0},
        "ONNXRuntime": {"reported": 0, "confirmed": 0, "fixed": 0},
        "TVM": {"reported": 0, "confirmed": 0, "fixed": 0},
        "TF XLA": {"reported": 0, "confirmed": 0, "fixed": 0},
        "Hidet": {"reported": 0, "confirmed": 0, "fixed": 0},
        "EinNet": {"reported": 0, "confirmed": 0, "fixed": 0},
    }

    for bug_report in buglist[1:]:
        col = bug_report.split(",")
        assert len(col) == 3
        compiler_name = col[0].split("#")[0]
        status = col[2].strip()
        if status == "reported":
            res_dic[compiler_name]["reported"] += 1
        elif status == "confirmed":
            res_dic[compiler_name]["reported"] += 1
            res_dic[compiler_name]["confirmed"] += 1
        elif status == "fixed":
            res_dic[compiler_name]["reported"] += 1
            res_dic[compiler_name]["confirmed"] += 1
            res_dic[compiler_name]["fixed"] += 1
        else:
            raise AssertionError(f"Unknown status: {status}")

    res_list = []
    for compiler_name, status_dic in res_dic.items():
        res_list.append([compiler_name, status_dic["reported"], status_dic["confirmed"], status_dic["fixed"]])

    output_path = os.path.join(OUTPUT_DIR_PATH, "tab3.csv")

    pd.DataFrame(data=res_list, columns=["Compiler", "Reported", "Confirmed", "Fixed"]).to_csv(
        output_path,
        index=False)


if __name__ == '__main__':
    main()
