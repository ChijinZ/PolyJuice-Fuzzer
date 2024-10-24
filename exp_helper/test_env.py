import multiprocessing
import subprocess
import os
import re


def main():
    print("start to test environment")
    flag = True
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 16:
        print(
            f"WARNING: your cpu count is {cpu_count}, this may not satisfy the requirement. "
            f"You can use raw-data for reproducing the results. From-scratch reproduction may encounter some problems")
        flag = False
    meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1])) for i in open('/proc/meminfo').readlines())
    mem_Gib = meminfo['MemTotal'] / 1024 / 1024
    if mem_Gib < 32:
        print(f"WARNING: your total memory size is {mem_Gib} GB, this may not satisfy the requirement"
              f"You can use raw-data for reproducing the results. From-scratch reproduction may encounter some problems")
        flag = False

    if flag:
        print("Your environment is suitable for reproducing the results")


if __name__ == "__main__":
    main()
