import time
import numpy as np
import mysql.connector as sql
from sys import stdout

# useful functions kept here
def sql_get(loc, lists, limit, cond=None):
    lim_case = {True: '', False: f'limit {limit}'}
    cond_case = {True: '', False: f'{cond}'}
    db = sql.connect(host="ip-173-16.main.oberlin.edu", user="julian", password="Jul14N", database="sdss")
    c = db.cursor()
    c.execute(f'''select {lists} from {loc} {cond_case[cond is None]} {lim_case[limit == 0]};''')
    rows = c.fetchall()
    db.close()
    return np.asarray(rows)


def sql_list_format(list):
    formatted = '('
    for l in list:
        formatted += f"'{l}', "
    formatted = formatted[:-2] + ')'
    return formatted


def display_flags(galaxies):
    for galaxy, g in galaxies.items():
        total = sum(g.values())
        print(f'Flags for objID {galaxy}: {total} total')
        print(f'Non-null: {100 * (total - g[None]) / total}%')
        print(f'Star-forming: {100 * (g["S06_SF"] + g["K03_SF"]) / (total - g[None])}%')
        print(f'Non star-forming: {100 * (g["K01_SF"] + g["AGN"]) / (total - g[None])}%\n')
    return


def write_data(location, galaxies):
    f = open(location, "w")
    is_end = {True: '\n', False: ','}
    for galaxy in galaxies:
        for item in galaxy:
            f.write(f'{item}{is_end[item == galaxy[-1]]}')
    f.close()


def read_data(location, dtype=int, lim=0):
    f = open(location, 'r')
    galaxies = []
    dtype_options = {True: str, False: dtype}
    i = 1
    for l in map(lambda line: line.strip().split(','), f.readlines()):
        galaxies.append(list(map(lambda s: dtype_options[len(s) == 18 or len(s) == 24](s), l)))
        if i == lim:
            return galaxies
        i += 1
    f.close()
    return np.asarray(galaxies)


def build_array(location):
    f = open(location, 'r')
    lines = list(map(lambda line: line.strip().split(','), f.readlines()))
    lines = list(filter(lambda line: 'None' not in line, lines))
    lines = list(map(lambda line: list(map(lambda l: float(l), line)), lines))
    arr = np.array([lines[0]])
    for l in lines[1:]:
        tuple = np.array([l])
        arr = np.concatenate((arr, tuple))
    return arr


def loading_bar(i, total, t):
    # num_squares = np.ceil(10 * i / total)
    stdout.write(f'\r{i}/{total} - Done in {"%.1f" % (t * (total - i))} s ')
    stdout.flush()
    if i == total:
        print('\n')


def filter_xy_by_z(z, data_x, data_y):
    mu = np.mean(data_x)
    sd = np.std(data_x)
    clean_x, clean_y = [], []
    for i in range(len(data_x)):
        x, y = data_x[i], data_y[i]
        if np.abs((x - mu) / sd) < z:
            clean_x.append(x)
            clean_y.append(y)
    return clean_x, clean_y


def printActionTime(tInit, actionName):
    currentTime = time.perf_counter()
    print(f'{actionName} in {currentTime - tInit} s')
    return currentTime


def writeNpArray(location, array):
    f = open(location, "w")
    for item in array:
        f.write(f'{item}\n')
    f.close()


def readNpArray(location):
    f = open(location, 'r')
    outputArray = []
    for line in f.readlines():
        line = line.strip()
        outputArray.append(line)
    return np.asarray(outputArray)
