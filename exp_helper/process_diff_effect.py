import pandas as pd
from scipy import stats
import sys
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)


def process_diff_effect(path):
    df = pd.read_csv(path)

    naive_edit_dis_mean = df["edit_res_1"].mean()
    polyjuice_edit_dis_mean = df["edit_res_2"].mean()

    naive_lcs_dis_mean = df["lcs_res_1"].mean()
    polyjuice_lcs_dis_mean = df["lcs_res_2"].mean()

    df_tmp = df[df['edit_res_1'] != 0]
    df2 = df_tmp[df_tmp['lcs_res_1'] != 0]

    edit_dis_impr = ((df2['edit_res_2'] - df2['edit_res_1']) / df2['edit_res_1'])
    lcs_dis_impr = ((df2['lcs_res_2'] - df2['lcs_res_1']) / df2['lcs_res_1'])

    edit_dis_avg_impr = edit_dis_impr.mean()
    lcs_dis_avg_impr = lcs_dis_impr.mean()

    edit_dis_median_impr = edit_dis_impr.median()
    lcs_dis_median_impr = lcs_dis_impr.median()

    edit_dis_impr_std = edit_dis_impr.std()
    lcs_dis_impr_std = lcs_dis_impr.std()

    edit_dis_impr_stderr = edit_dis_impr.sem()
    lcs_dis_impr_stderr = lcs_dis_impr.sem()

    res_df = pd.DataFrame(data=[
        ["{0:.2f}%".format(lcs_dis_avg_impr * 100), lcs_dis_impr_stderr, "{0:.2f}%".format(edit_dis_avg_impr * 100),
         edit_dis_impr_stderr]],
        columns=["lcs_dis_avg_impr", "lcs_dis_impr_stderr", "edit_dis_avg_impr",
                 "edit_dis_impr_stderr", ])
    print(res_df)

    output_dir_path = os.environ.get("OUTPUT_DIR")
    if output_dir_path is not None:
        csv_output_path = os.path.join(output_dir_path, "table4.csv")
        res_df.to_csv(csv_output_path, index=False)

    # print({
    #     "edit_dis_avg_impr": edit_dis_avg_impr, "lcs_dis_avg_impr": lcs_dis_avg_impr,
    #     "edit_dis_impr_stderr": edit_dis_impr_stderr, "lcs_dis_impr_stderr": lcs_dis_impr_stderr})


if __name__ == '__main__':
    path = sys.argv[1]
    process_diff_effect(path)
