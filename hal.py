import time
import sys


def print_line(line):
    sys.stdout.write("HAL: ")
    sys.stdout.flush()
    line += '\n'

    for char in line:
        time.sleep(0.05)
        sys.stdout.write(char)
        sys.stdout.flush()


def print_hal():
    hal = ["\n****************",
           "*   HAL|9000   *",
           "*              *",
           "*              *",
           "*              *",
           "*              *",
           "*     ****     *",
           "*   *      *   *",
           "* *          * *",
           "* *          * *",
           "* *          * *",
           "*   *      *   *",
           "*     ****     *",
           "*              *",
           "****************",
           "*..............*",
           "*..............*",
           "****************\n"]

    for line in hal:
        print(line)
        time.sleep(0.1)

    print_line("I am HAL 9000. You may call me HAL.")
