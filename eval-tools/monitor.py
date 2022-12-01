import os
import sys
import psutil
import time
import subprocess
from datetime import datetime


def main():
    dataset = sys.argv[1]
    model_name = sys.argv[2]
    pid = int(sys.argv[3])

    p = psutil.Process(pid)

    mem = []

    i = 0
    while True:
        try:
            m = p.memory_info()[0] / 2.**20

            if i % 10 == 0:
                mem.append(int(m))

            print(f'[{i * 60}] mem usage: {m}')
            i += 1

        except Exception as e:
            print(e)
            break

        time.sleep(60)

    now = datetime.now().strftime("%Y-%m-%d-%H%M")

    pwd = os.getcwd()
    os.makedirs(f'{pwd}/eval-logs', exist_ok=True)

    with open(f'{pwd}/eval-logs/{dataset}-{model_name}-mem.txt', 'a') as fd:
        fd.write(f'[pid] {now}| {" ".join([str(m) for m in mem])}\n')


if __name__ == '__main__':
    main()
