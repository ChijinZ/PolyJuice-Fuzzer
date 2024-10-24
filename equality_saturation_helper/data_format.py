import pickle
import os
import torch

root_path = "/workspace/modified_nnsmith/triage_bugs/infini_11_27/fuzz_report_cpu_11_27/cpu_transpose"
torch_model = os.path.join(root_path, "t1", "model.pth")
data_file = os.path.join(root_path, "0.pickle")

idx = 0
key_list = []
print(pickle.load(open(data_file, 'rb')))
for k, v in pickle.load(open(data_file, 'rb')).items():
    print(f"data_{idx} = np.random.normal(5, 1, size={v.shape}).astype(np.{v.dtype})")
    key_list.append(k)
    idx += 1

# dict_str = "input_data_0 = ["
# for i in range(idx):
#     dict_str += f"data_{i},"
# dict_str += "]"
# print(dict_str)

dict_str = "input_dict_0 = {"
for i in range(idx):
    dict_str += f"'{key_list[i]}': data_{i}, "
dict_str += "}"
print(dict_str)


# print("\n")
# model_consts = torch.load(torch_model)
# p_idx = 0
# for k, v in model_consts.items():
#     if sum(list(v.shape)) < 10:
#         print(v)
#     print(f"p{p_idx} = np.random.normal(0, 3, size={tuple(v.shape)}).astype(np.{v.dtype})")
#     print(f"p{p_idx} = torch.from_numpy(p{p_idx}).to(DEVICE)")