import sys
import difflib


# return the difference rate of two log files
def evaluate(log1_lines, log2_lines) -> float:
    # Determine which log is longer
    long_log, short_log = (log1_lines, log2_lines) if len(log1_lines) > len(log2_lines) else (log2_lines, log1_lines)

    diff = difflib.unified_diff(long_log, short_log)

    # diff_lines = len([l for l in diff if l.startswith('-')])
    diff_lines = len(list(diff))

    return diff_lines / len(long_log)


def main():
    assert len(sys.argv) == 3, "Usage: python evaluate_tvm_log.py log_path_1 log_path_2"

    log_path_1 = sys.argv[1]
    log_path_2 = sys.argv[2]

    with open(log_path_1, 'r') as f:
        lines_1 = f.readlines()

    with open(log_path_2, 'r') as f:
        lines_2 = f.readlines()

    print(evaluate(lines_1, lines_2))


if __name__ == '__main__':
    main()
