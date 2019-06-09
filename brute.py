import os
import pandas as pd
import numpy as np
import sympy as sp
from itertools import permutations
from itertools import chain
from tqdm import tqdm


def distance(coor, pen=9):
    xy, z = np.hsplit(coor, [2])
    dist = np.hypot(*(xy[:-1] - xy[1:]).T)
    dist[pen::10] *= z[:-1][pen::10].flat
    return np.sum(dist)


def submit(tour):
    np.savetxt('submission.csv', tour, fmt='%d', header='Path', comments='')


df = pd.read_csv('cities.csv')
df['Z'] = 1 + .1 * ~df['CityId'].apply(sp.isprime)
data = df[['X', 'Y', 'Z']].values
if os.path.exists('submission.csv'):
    tour = np.loadtxt('submission.csv', skiprows=1, dtype=int)
else:
    tour = np.append(df['CityId'].values, 0)
mapi = dict(zip([tuple(x) for x in data], range(len(data))))
coor = np.array([data[x] for x in tour])
print(distance(coor))

step = 3
size = 4

for i in tqdm(range(1, len(data) - size, step)):
    alter = np.insert(arr=list(permutations(coor[i:size + i])),
                      obj=[0, size],
                      values=[coor[i - 1], coor[size + i]],
                      axis=1)
    pen = 9 - (i - 1) % 10
    arg = np.argmin([distance(x, pen) for x in alter])
    if arg != 0:
        print(distance(coor))
        coor[i:size + i] = alter[arg][1:-1]
        tour = np.array([mapi[tuple(x)] for x in coor])
        submit(tour)
