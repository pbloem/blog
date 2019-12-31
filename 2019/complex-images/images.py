import numpy as np
import random, sys, math, time
import heapq as hq

from math import sin, cos

from functools import total_ordering

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from skimage.io import imsave

from tqdm import trange

from PIL import Image

from collections import deque

def u(mi, mx):
    return random.uniform(mi, mx)

SEED = random.randint(-1_000_000_000, 1_000_000_000)
print('seed', SEED)
random.seed(SEED)

# total runtime in seconds
TOTALTIME = 30 * 60
# Safety margin for producing output in seconds
MARGIN = 10

NUMDEST = 1
BASECOST = 1.0

GIF = random.choice([True, False])
print('gif?', GIF)
FRAMES = 240

MAPSIZE = random.choice([[135, 240], [270, 480]]) if GIF else \
          random.choice([[540, 960], [720, 1280], [1080, 1920]])
print('map size', MAPSIZE)

NUM_AGENTS = random.choice([1, 2, 3, 4, 5, 7, 10, 100])
print('num agents', NUM_AGENTS)

PLOT_EVERY = 1_000
PLOT_THESE = [0, 10, 100, 200, 300, 500]

MOVES = random.choice(['four', 'diag', 'eight', 'circ'])
print('moves', MOVES)

MOVES_FOUR = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
MOVES_EIGHT = MOVES_FOUR + DIAG
MOVES_DIAG = [(0, 1), (0, -1)] + DIAG

CIRC_DIAM = 10.0
CIRC_MOVES = 120
MOVES_CIRC = [(int(CIRC_DIAM * cos(r)), int(CIRC_DIAM * sin(r))) for r in np.linspace(0.0, 2*math.pi, CIRC_MOVES)]

IMPROVEMENT = 10.0 ** (u(-1.5, 1.0))
IMPROVEMENT_VAR = random.choice(['perant', None, u(0.0, 5.0)])
IMPROVEMENT_REPS = random.choice([1, 2, 3, 10, 25])

INIT = random.choice(['blank', 'grad', 'bell'])
INIT_NOISE = random.choice([0.0, u(0.0, 1.0)])
INIT_INVERT = random.choice([True, False])

COLORMAP = random.choice([
        random.choice([cm.copper, cm.bone, cm.gray]),
        random.choice([cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.RdPu, cm.pink, cm.RdYlBu])
    ])

global starttime
starttime = time.time()

def moves(id):
    if id == 'four':
        return MOVES_FOUR
    if id == 'diag':
        return MOVES_DIAG
    if id == 'eight':
        return MOVES_EIGHT
    if id == 'circ':
        return MOVES_CIRC

def heuristic(id, fr, to):
    if id == 'four':
        return (abs(fr[0] - to[0]) + abs(fr[1] - to[1])) * BASECOST
    if id == 'eight' or id == 'diag':
        return BASECOST * math.sqrt((fr[0] - to[0])**2 + (fr[1] - to[1])**2) / math.sqrt(2)
    if id == 'circ':
        return BASECOST * math.sqrt((fr[0] - to[0])**2 + (fr[1] - to[1])**2) / CIRC_DIAM

def rescale(array):
    range = array.max() - array.min()
    return (array - array.min()) / range

def toimage(array, pil=False):
    # range = array.max() - array.min()
    # rescaled =  (array - array.min()) / range

    array = np.log(1.0 + array)

    norm = colors.Normalize(vmin=0.)
    rgb = COLORMAP(norm(array))
    if not pil:
        return rgb

    return Image.fromarray(np.uint8(rgb*255))


def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)

    # axes.get_xaxis().set_tick_params(which='both', top='off', bottom='off', labelbottom='off')
    # axes.get_yaxis().set_tick_params(which='both', left='off', right='off')

class Agent:

    def __init__(self, map):
        self.nx, self.ny = map.shape
        self.pos = (random.randint(0, self.nx-1), random.randint(0, self.ny-1))

        self.plan = None
        self.map = map

        self.improvement_var = random.choice([0.0, u(0.0, 5.0)]) if IMPROVEMENT_VAR == 'perant' else None

    def move(self):
        if not bool(self.plan): # empty or None

            # Choose a new destination
            cost = float('inf')
            for _ in range(NUMDEST):
                cdest = (random.randint(0, self.nx-1), random.randint(0, self.ny-1))
                cplan, ccost = pathplan(self.map, self.pos, cdest)
                if cplan is None:
                    return False

                if ccost < cost:
                    cost = ccost
                    self.plan = cplan

        try:
            dx, dy = self.plan.popleft()
        except Exception as e:
            print(self.plan)
            raise e

        # move agent
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        # improve infrastructure
        x, y = self.pos[0], self.pos[1]
        if IMPROVEMENT_VAR is not None:
            impvar = IMPROVEMENT_VAR if self.improvement_var is None else self.improvement_var

            for _ in range(IMPROVEMENT_REPS):
                xt = x + int(random.gauss(0, 1) * impvar)
                yt = y + int(random.gauss(0, 1) * impvar)
                xt = min(max(xt, 0), map.shape[0]-1)
                yt = min(max(yt, 0), map.shape[1]-1)
                map[xt, yt] += IMPROVEMENT/IMPROVEMENT_REPS

        else:
            map[x, y] += IMPROVEMENT

        return True

def pathplan(map, fr, to):
    """
    Basic A* search

    :param map:
    :param fr:
    :param to:
    :return:
    """

    maxwidth = 0

    @total_ordering
    class Node:

        def __init__(self, pos, cost, parent=None):
            self.pos = pos
            self.cost = cost
            self.heuristic = heuristic(MOVES, pos, to)
            self.parent = parent
            self.alive = True

        def children(self):
            """
            Generator for child nodes
            :return:
            """
            smoves = moves(MOVES).copy()
            random.shuffle(smoves)

            for move in smoves:

                npos = (self.pos[0] + move[0], self.pos[1] + move[1])

                if npos[0] >= map.shape[0] or npos[0] < 0 or npos[1] >= map.shape[1] or npos[1] < 0:
                    continue;

                movecost = BASECOST + (1.0 / (map[npos[0], npos[1]] + 10e-10))
                ncost = self.cost + movecost

                yield Node(npos, ncost, self)

        def path(self):
            """
            Extract the path
            :return:
            """

            path = deque()
            current = self

            while current.parent is not None:
                move = (current.pos[0] - current.parent.pos[0], current.pos[1] - current.parent.pos[1])
                path.appendleft(move)

                current = current.parent

            return path

        def key(self):
            return self.cost + self.heuristic

        def __eq__(self, other):
            return (self.key() == other.key())

        def __ne__(self, other):
            return not (self.key() == other.key())

        def __lt__(self, other):
            return self.key() < other.key()

    start = Node(pos=fr, cost=0.0, parent=None)
    pq = [start]
    hq.heapify(pq)

    dict = {start.pos:start}

    while True:
        if (time.time() - starttime) > (TOTALTIME - MARGIN):
            return (None, None)

        current = None
        while current is None or not current.alive:
            current = hq.heappop(pq)

        # print('pop ', fr, current.pos, to, f'{current.cost:.3}', current.heuristic)

        if current.pos == to:
            # print('max width ', maxwidth)
            return current.path(), current.cost

        for node in current.children():

            if node.pos not in dict:
                hq.heappush(pq, node)
                dict[node.pos] = node
            else:
                other = dict[node.pos]
                if node.key() < other.key():
                    other.alive = False

                    hq.heappush(pq, node)
                    dict[node.pos] = node

            maxwidth = max(maxwidth, len(pq))

if INIT == 'blank':
    map = np.ones(shape=MAPSIZE)

elif INIT == 'grad':

    x, y = np.arange(MAPSIZE[0], dtype=np.float), np.arange(MAPSIZE[1], dtype=np.float)
    mesh = np.meshgrid(x, y)
    map = ((mesh[0] + mesh[1]) + 0.5).transpose()
    map *= (np.random.rand(*MAPSIZE) * 0.75 + 0.25)

elif INIT == 'bell':

    map = np.zeros(shape=MAPSIZE)

    for _ in range(random.choice([1, 2, 3, 4, 5])):

        asp = MAPSIZE[0]/MAPSIZE[1]

        xmid = u(-1, 1)
        ymid = u(-asp, asp)
        var = 10 ** u(-2, 0.1)

        x, y = np.linspace(-1 + xmid, 1+xmid, MAPSIZE[1]), np.linspace(-asp+ymid, asp+ymid, MAPSIZE[0])
        mesh = np.meshgrid(x, y)
        map += np.exp((- mesh[0] ** 2 - mesh[1] **2)/var)

    map += 0.00001
else:
    raise Exception(f'INIT {INIT} not recognized.')

map += np.random.rand(*MAPSIZE) * INIT_NOISE * 0.25


if INIT_INVERT:
    map = rescale(map)
    map = 1.0 - map

numpixels = MAPSIZE[0] * MAPSIZE[1]

agents = [Agent(map) for _ in range(NUM_AGENTS)]

# plt.figure(figsize=(16, 9))

it = 0;

if GIF:
    perframe = (TOTALTIME - MARGIN)/FRAMES
    t0 = time.time()
    images = []
    f = 0

while True:

    map /= map.sum()
    map *= numpixels

    for agent in agents:
        res = agent.move()
        if not res:
            break;
    if not res:
        break;

    if it % PLOT_EVERY == 0 or it in PLOT_THESE:
        # plt.cla()
        # clean()
        # plt.imshow(map, cmap='copper')
        # plt.savefig(f'plot.{it:06}.png')
        imsave(f'plot.{it:06}.png', toimage(map), check_contrast=False)

    if GIF and time.time() - t0 > perframe:
        images.append(toimage(map, pil=True))
        t0 = time.time()

        images[-1].save(fp=f'frame{f:02}.png', format='PNG')
        f += 1


    it += 1

if GIF:
    images[0].save(fp='final.gif', format='GIF', append_images=images, save_all=True)

imsave(f'final.png', toimage(map), check_contrast=False)

print('done')



