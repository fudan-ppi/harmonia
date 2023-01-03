import numpy as np
import util


def generate_insert(bit: int, m: int):
    """generate insert data file
    bit: must be 32 or 64
    m: the generated file will be m million lines
    """
    assert bit == 32 or bit == 64
    output_dir = util.dataset_path() / 'insert{}_{}m'.format(bit, m)
    output_path = output_dir / 'insert{}_{}m.txt'.format(bit, m)
    max_val = util.max_val(bit)
    print('generate_insert: {}'.format(output_path.name))
    lines = util.generate_random(
        max_val, m * util.ONE_MILLI, True)
    util.output_as_file(lines, output_path)


def generate_search(bit: int, dist: str, m: int):
    """generate search data file
    bit: must be 32 or 64
    dist: distribution
    m: the generated file will be m million lines
    """
    assert bit == 32 or bit == 64
    output_dir = util.dataset_path() / 'search{}_{}m'.format(bit, m)
    output_path = output_dir / 'search{}_{}_{}m.txt'.format(bit, dist, m)
    max_val = util.max_val(bit)
    print('generate_search: {}'.format(output_path.name))
    lines = util.dist_generator(dist)(max_val, m * util.ONE_MILLI)
    util.output_as_file(lines, output_path)


def generate_uniform_range(bit: int, m: int, max_range_size: int = 5247):
    """generate uniform range data file
    bit: must be 32 or 64
    m: the generated file will be 2 * m million lines, i line for from, i + 1 line for to
    max_range_size: one range query could contain at most max_range_size keys
    """
    assert bit == 32 or bit == 64
    output_dir = util.dataset_path() / 'range{}_{}m'.format(bit, m)
    output_path = output_dir / 'range{}_uniform_{}m.txt'.format(bit, m)
    max_val = util.max_val(bit)
    print('generate_uniform_range: {}'.format(output_path.name))
    util.output_as_file(util.generate_uniform_range(
        max_val, m * util.ONE_MILLI, max_range_size), output_path)


def generate_update(bit: int, uk: int, im: int, update_ratio: float = 1.0):
    """generate update data file
    bit: must be 32 or 64
    uk: update kilo queries, the generated data file will be (uk * kilo) lines
    im: insert million queries
    update_ratio: update ratio
    """
    assert bit == 32 or bit == 64
    assert update_ratio >= 0.0 and update_ratio <= 1.0
    output_dir = util.dataset_path() / 'update{}_{}k'.format(bit, uk)
    output_path = output_dir / \
        'update{}_{}_{}k'.format(bit, int(update_ratio * 10000.0), uk)

    # read insert file
    insert_path = util.dataset_path() / 'insert{}_{}m'.format(bit, im) / \
        'insert{}_{}m.txt'.format(bit, im)
    assert insert_path.exists() and insert_path.is_file()
    inserts = []
    with open(insert_path, mode='r', encoding='utf-8') as infile:
        inserts = infile.readlines()
        inserts = [int(_) for _ in inserts]

    # pickout updates
    nupdate = int(uk * util.ONE_KILO * update_ratio)
    assert nupdate <= len(inserts)
    ninsert = int(uk * util.ONE_KILO) - nupdate
    update_indexes = np.random.choice(len(inserts), nupdate, False)
    print('generate_update: {}'.format(output_path.name))
    updates = [inserts[_] for _ in update_indexes]

    # some new inserts
    max_val = util.max_val(bit)
    new_inserts = util.generate_random(max_val, ninsert, True, set(inserts))

    # join updates and inserts
    ans_list = updates + new_inserts
    np.random.shuffle(ans_list)
    util.output_as_file(ans_list, output_path)


if __name__ == '__main__':
    util.print_now()

    # print('generate: main: please wait, this usually takes about 3 minutes')
    # insert 4M
    insert_milli = 4
    generate_insert(32, insert_milli)
    generate_insert(64, insert_milli)

    # search 1M
    generate_search(32, util.DIST_NORMAL, 1)
    generate_search(32, util.DIST_UNIFORM, 1)
    generate_search(32, util.DIST_GAMMA, 1)
    generate_search(64, util.DIST_NORMAL, 1)
    generate_search(64, util.DIST_UNIFORM, 1)
    generate_search(64, util.DIST_GAMMA, 1)

    # search 100M
    # generate_search(32, util.DIST_NORMAL, 100)
    generate_search(32, util.DIST_UNIFORM, 100)
    # generate_search(32, util.DIST_GAMMA, 100)
    # generate_search(64, util.DIST_NORMAL, 100)
    generate_search(64, util.DIST_UNIFORM, 100)
    # generate_search(64, util.DIST_GAMMA, 100)

    # search 200M
    # generate_search(32, util.DIST_NORMAL, 200)
    # generate_search(32, util.DIST_UNIFORM, 200)
    # generate_search(32, util.DIST_GAMMA, 200)
    # generate_search(64, util.DIST_NORMAL, 200)
    # generate_search(64, util.DIST_UNIFORM, 200)
    # generate_search(64, util.DIST_GAMMA, 200)

    # range search 100M
    generate_uniform_range(32, 1)
    generate_uniform_range(64, 1)

    # 1024k = 1M total update data file, from 8M insert file, and update ratio is 0.7425(a magic ratio)
    generate_update(32, 1024, insert_milli, 0.7425)
    generate_update(64, 1024, insert_milli, 0.7425)
    util.print_now()
