import sys
import re


def main():
    log_path = sys.argv[1]
    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("["):
            pattern = r'\[\d{2}:\d{2}:\d{2}\]\s'
            log_without_timestamp = re.sub(pattern, '', line)
            if "Cannot emit debug location for undefined span" not in log_without_timestamp \
                    and "Warning" not in log_without_timestamp \
                    and "naive_allocator" not in log_without_timestamp:
                print(log_without_timestamp, end='')


if __name__ == '__main__':
    main()
