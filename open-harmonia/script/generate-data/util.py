import numpy as np
from pathlib import Path
import datetime


CWD_PATH = Path('./')
MAX_INT32 = int(np.iinfo(np.int32).max)
MAX_INT64 = int(np.iinfo(np.int64).max)
ONE_KILO = 1000
ONE_MILLI = ONE_KILO * ONE_KILO
ONE_GIGA = ONE_MILLI * ONE_MILLI
DIST_UNIFORM = 'uniform'
DIST_NORMAL = 'normal'
DIST_GAMMA = 'gamma'
DIST_ZIPF = 'zipf'


def print_now():
    print(datetime.datetime.now().strftime('%Y-%m-%d, %H:%M:%S'))


def dataset_path():
    ans = CWD_PATH / 'dataset'
    ans.mkdir(parents=True, exist_ok=True)
    return ans


def max_val(bit: int):
    if bit == 32:
        return MAX_INT32
    elif bit == 64:
        return MAX_INT64
    else:
        raise KeyError('max_val: bit must be 32 or 64, but {}'.format(bit))


def output_as_file(lines: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode='w+', encoding='utf-8') as outfile:
        lines = [str(_) for _ in lines]
        lines = [str(_) if str(_).endswith('\n') else '{}\n'.format(_) for _ in lines]
        outfile.writelines(lines)


def generate_random(max_val: int, n: int, unique: bool = True, unique_set: set = set()):
    ans_list = []
    for _ in range(0, n):
        v = 0
        while True:
            v = np.random.randint(1, max_val)
            if not unique or v not in unique_set:
                break
        if unique:
            unique_set.add(v)
        ans_list.append(v)
    return ans_list


def box(v: int, min_val: int, max_val: int):
    return min(max(v, min_val), max_val)


def generate_uniform(max_val: int, n: int):
    ans_list = list(np.random.random_sample(n))
    ans_list = [box(int(_ * max_val), 1, max_val - 1) for _ in ans_list]
    return ans_list


def generate_normal(max_val: int, n: int):
    u = 0.5
    sigma = float(np.sqrt(0.125))
    ans_list = []
    while len(ans_list) < n:
        v = float(np.random.normal(u, sigma))
        if v >= 0.0 and v <= 1.0:
            v = int(v * max_val)
            ans_list.append(box(v, 1, max_val - 1))
    return ans_list


def generate_gamma(max_val: int, n: int):
    k = 3
    theta = 3
    ans_list = []
    while len(ans_list) < n:
        v = float(np.random.gamma(k, theta))
        if v >= 0.0 and v <= 20.0:
            v = int(max_val * (v / 20))
            ans_list.append(box(v, 1, max_val - 1))
    return ans_list


def generate_zipf(max_val: int, n: int):
    xxxx = 100.0
    alpha = 2
    ans_list = []
    while len(ans_list) < n:
        v = float(np.random.zipf(alpha))
        if v >= 0.0 and v < xxxx:
            v = int(max_val * (v / xxxx))
            ans_list.append(box(v, 1, max_val - 1))
    return ans_list


def dist_generator(dist: str):
    if dist == DIST_NORMAL:
        return generate_normal
    elif dist == DIST_UNIFORM:
        return generate_uniform
    elif dist == DIST_GAMMA:
        return generate_gamma
    elif dist == DIST_ZIPF:
        return generate_zipf
    else:
        raise KeyError('dist_generator: unknown dist {}'.format(dist))


def generate_uniform_range(max_val: int, n: int, max_range_size: int = 5247):
    lefts = generate_uniform(max_val, n)
    rights = list(np.random.random_sample(n))
    delta = (max_val / n) * (max_range_size - 2)
    ans_list = []
    for i in range(0, n):
        left = lefts[i]
        right = left + int(delta * rights[i])
        right = min(max_val - 1, right)
        ans_list.extend([left, right])
    return ans_list




