import os
import argparse
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from itertools import chain
from itertools import combinations
from itertools import permutations
from itertools import product
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def initial():
    df = pd.read_csv('cities.csv')
    df['Z'] = 1 + .1 * ~df['CityId'].apply(sp.isprime)
    data = df[['X', 'Y', 'Z']].values
    if os.path.exists('submission.csv'):
        tour = np.loadtxt('submission.csv', skiprows=1, dtype=int)
    else:
        tour = np.append(df['CityId'].values, 0)
    return tour, data


def distance(tour, data, pen=9):
    xy, z = np.hsplit(data[tour], [2])
    dist = np.hypot(*(xy[:-1] - xy[1:]).T)
    dist[pen::10] *= z[:-1][pen::10].flat
    return dist


def candidates(data, opt, ext):
    nns = NearestNeighbors(n_neighbors=opt + ext).fit(data[:, :2])
    kne = nns.kneighbors(data[:, :2], return_distance=False)
    np.random.shuffle(kne)
    cand = set()
    for i in kne:
        for j in combinations(i[1:], opt - 1):
            cand.add(tuple(sorted((i[0],) + j)))
    return cand


def alternatives(tour, cuts, fil):
    edges = [tuple(x) for x in np.split(tour, cuts)[1:-1]]
    a, b = tour[cuts[0] - 1], tour[cuts[-1]]
    alter = set()
    for i in set(product(*zip(edges, [x[::-1] for x in edges]))):
        for j in permutations(i):
            if not fil or all(x != y for x, y in zip(edges, j)):
                alter.add(tuple(chain((a,), *j, (b,))))
    alter.discard(tuple(chain((a,), *edges, (b,))))
    return alter


def submit(tour):
    np.savetxt('submission.csv', tour, fmt='%d', header='Path', comments='')


def chopchop(opt, ext, fil):
    tour, data = initial()
    sequ = 1 + np.argsort(tour[1:])
    dist = distance(tour, data)
    print(f'opt:{opt} & extra:{ext} & filter:{fil} ...')
    cand = candidates(data, opt, ext)
    print(f' Score: {np.sum(dist):0.2f}')
    for c in tqdm(cand, bar_format='{percentage:6.2f}%', mininterval=1):
        cuts = sorted(sequ[i] for i in c)
        alter = alternatives(tour, cuts, fil)
        if not alter:
            continue
        atour, pen = np.array(list(alter)), 9 - (cuts[0] - 1) % 10
        adist = np.array([distance(x, data, pen) for x in atour])
        if np.any(np.sum(adist, 1) < np.sum(dist[cuts[0] - 1:cuts[-1]])):
            arg = np.argmin(np.sum(adist, 1))
            dist[cuts[0] - 1:cuts[-1]] = adist[arg]
            tour[cuts[0]:cuts[-1]] = atour[arg][1:-1]
            sequ[atour[arg][1:-1]] = range(cuts[0], cuts[-1])
            submit(tour)
            print(f'\t{np.sum(dist):0.2f}')
    print()


def tipitapa():
    tour, data = initial()
    xy = data[tour][:, :2]
    segm = np.hstack((xy[:-1], xy[1:])).reshape(-1, 2, 2)
    lc = mcoll.LineCollection(segments=segm,
                              array=np.linspace(0, 1, len(segm)),
                              cmap=plt.get_cmap('Spectral'),
                              lw=.9)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.add_collection(lc)
    ax.plot(*xy.T, lw=.3, c='black')
    fig.savefig('hohoho.png', dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--opt', type=int)
    parser.add_argument('-e', '--ext', type=int, default=0)
    parser.add_argument('-f', '--fil', action='store_true')
    parser.add_argument('-g', '--gra', action='store_true')
    args = parser.parse_args()
    if args.gra:
        tipitapa()
    else:
        if args.opt:
            chopchop(args.opt, args.ext, args.fil)
        else:
            for opt in range(2, 6):
                chopchop(opt, args.ext, args.fil)


if __name__ == '__main__':
    main()
