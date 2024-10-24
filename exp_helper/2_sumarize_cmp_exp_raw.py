import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn3_unweighted

from exp_helper.bug_analysis import analyze_crash, analyze_silent

RAW_EXP_DIR = "/root/raw_data/raw_data/fig10_raw_data"

OUTPUT_DIR_PATH = "/root/raw_data/raw_data_results"


def main():
    polyjuice_tvm_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_eq_fuzz_tvm_0")
    polyjuice_xla_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_eq_fuzz_xla_0")
    nnsmith_tvm_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_nnsmith_tvm_0")
    nnsmith_xla_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_nnsmith_xla_0")
    mtdlcomp_tvm_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_mtdlcomp_tvm_0")
    mtdlcomp_xla_dir_path = os.path.join(RAW_EXP_DIR, "fuzz_report_mtdlcomp_xla_0")

    polyjuice_tvm_crash_set = analyze_crash(polyjuice_tvm_dir_path)
    polyjuice_xla_crash_set = analyze_crash(polyjuice_xla_dir_path)

    nnsmith_tvm_crash_set = analyze_crash(nnsmith_tvm_dir_path)
    nnsmith_xla_crash_set = analyze_crash(nnsmith_xla_dir_path)

    mtdlcomp_tvm_crash_set = analyze_crash(mtdlcomp_tvm_dir_path)
    mtdlcomp_xla_crash_set = analyze_crash(mtdlcomp_xla_dir_path)

    polyjuice_tvm_diff_set, polyjuice_tvm_eq_diff_set = analyze_silent(polyjuice_tvm_dir_path)
    polyjuice_xla_diff_set, polyjuice_xla_eq_diff_set = analyze_silent(polyjuice_xla_dir_path)

    nnsmith_tvm_diff_set, nnsmith_tvm_eq_diff_set = analyze_silent(nnsmith_tvm_dir_path)
    nnsmith_xla_diff_set, nnsmith_xla_eq_diff_set = analyze_silent(nnsmith_xla_dir_path)

    mtdlcomp_tvm_diff_set, mtdlcomp_tvm_eq_diff_set = analyze_silent(mtdlcomp_tvm_dir_path)
    mtdlcomp_xla_diff_set, mtdlcomp_xla_eq_diff_set = analyze_silent(mtdlcomp_xla_dir_path)

    # matplotlib.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # order: NNSmith, MT-DLComp, PolyJuice
    v = venn3_unweighted(subsets=({
        '100': len(
            nnsmith_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set) - (
                    nnsmith_tvm_crash_set & polyjuice_tvm_crash_set)
        ),
        '001': len(
            polyjuice_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    polyjuice_tvm_crash_set & mtdlcomp_tvm_crash_set) - (
                    polyjuice_tvm_crash_set & nnsmith_tvm_crash_set)
        ),
        '010': len(
            mtdlcomp_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    mtdlcomp_tvm_crash_set & nnsmith_tvm_crash_set)
        ),
        '110': len(nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '101': len(nnsmith_tvm_crash_set & polyjuice_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '011': len(mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '111': len(nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[0][0])
    axes[0][0].set_title("TVM Crash/Exception Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    v = venn3_unweighted(subsets=({
        '100': len(
            nnsmith_xla_crash_set - (nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set) - (
                    nnsmith_xla_crash_set & mtdlcomp_xla_crash_set) - (
                    nnsmith_xla_crash_set & polyjuice_xla_crash_set)),
        '001': len(
            polyjuice_xla_crash_set - (nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set) - (
                    polyjuice_xla_crash_set & mtdlcomp_xla_crash_set) - (
                    polyjuice_xla_crash_set & nnsmith_xla_crash_set)),
        '010': len(
            mtdlcomp_xla_crash_set - (nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set) - (
                    mtdlcomp_xla_crash_set & polyjuice_xla_crash_set) - (
                    mtdlcomp_xla_crash_set & nnsmith_xla_crash_set)),
        '110': len(nnsmith_xla_crash_set & mtdlcomp_xla_crash_set - (
                nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set)),
        '101': len(nnsmith_xla_crash_set & polyjuice_xla_crash_set - (
                nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set)),
        '011': len(mtdlcomp_xla_crash_set & polyjuice_xla_crash_set - (
                nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set)),
        '111': len(nnsmith_xla_crash_set & mtdlcomp_xla_crash_set & polyjuice_xla_crash_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[0][1])
    axes[0][1].set_title("XLA Crash/Exception Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    v = venn3_unweighted(subsets=({
        '100': len(nnsmith_tvm_diff_set),
        '001': len(polyjuice_tvm_eq_diff_set),
        '010': len(mtdlcomp_tvm_diff_set),
        '110': len(nnsmith_tvm_diff_set & mtdlcomp_tvm_diff_set),
        '101': len(nnsmith_tvm_diff_set & polyjuice_tvm_diff_set),
        '011': len(mtdlcomp_tvm_diff_set & polyjuice_tvm_diff_set),
        '111': len(nnsmith_tvm_diff_set & mtdlcomp_tvm_diff_set & polyjuice_tvm_diff_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[1][0])
    axes[1][0].set_title("TVM Mis-Compilation Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    v = venn3_unweighted(subsets=({
        '100': len(nnsmith_xla_diff_set),
        '001': len(polyjuice_xla_eq_diff_set),
        '010': len(mtdlcomp_xla_diff_set),
        '110': len(nnsmith_xla_diff_set & mtdlcomp_xla_diff_set),
        '101': len(nnsmith_xla_diff_set & polyjuice_xla_diff_set),
        '011': len(mtdlcomp_xla_diff_set & polyjuice_xla_diff_set),
        '111': len(nnsmith_xla_diff_set & mtdlcomp_xla_diff_set & polyjuice_xla_diff_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[1][1])
    axes[1][1].set_title("XLA Mis-Compilation Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    # plt.subplots_adjust(wspace=0.05)
    # plt.subplots_adjust(hspace=0.00)
    figure_fig = plt.gcf()  # 'get current figure'

    output_path = os.path.join(OUTPUT_DIR_PATH, "fig10.pdf")
    if os.path.exists(output_path):
        os.remove(output_path)
    figure_fig.savefig(output_path,
                       format='pdf',
                       dpi=1000,
                       bbox_inches='tight',
                       pad_inches=0)
    # plt.show()


if __name__ == '__main__':
    main()
