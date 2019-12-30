import numpy as np
import random, sys, math, time
import heapq as hq

from math import sin, cos

from functools import total_ordering

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from skimage.io import imsave

from tqdm import trange

from collections import deque

def u(mi, mx):
    return random.uniform(mi, mx)

# total runtime in seconds
TOTALTIME = 30 * 60
# Safety margin for producing output
MARGIN = 10

NUMDEST = 1
BASECOST = 1.0
MAPSIZE = random.choice([[135, 240], [270, 480], [540, 960],  [720, 1280], [1080, 1920], ])

NUM_AGENTS = random.choice([1, 2, 5, 10, 100])
PLOT_EVERY = 1_000
PLOT_THESE = [0, 10, 100, 200, 300, 500]

MOVES = random.choice(['four', 'diag', 'eight', 'circ'])
MOVES_FOUR = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
MOVES_EIGHT = MOVES_FOUR + DIAG
MOVES_DIAG = [(0, 1), (0, -1)] + DIAG

CIRC_DIAM = 10.0
CIRC_MOVES = 120
MOVES_CIRC = [(int(CIRC_DIAM * cos(r)), int(CIRC_DIAM * sin(r))) for r in np.linspace(0.0, 2*math.pi, CIRC_MOVES)]

IMPROVEMENT = 10.0 ** (u(-2.0, 2.0))
IMPROVEMENT_VAR = random.choice([None, u(0.0, 5.0)])

INIT = random.choice(['blank', 'grad', 'white', 'bell'])
INIT_NOISE = random.choice([0.0, u(0.0, 1.0)])
print(INIT_NOISE)
INIT_INVERT = random.choice([True, False])

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

    def move(self):
        if not self.plan: # empty or None

            # Choose a new destination
            cost = float('inf')
            for _ in range(NUMDEST):
                cdest = (random.randint(0, self.nx-1), random.randint(0, self.ny-1))
                cplan, ccost = pathplan(self.map, self.pos, cdest)
                if cplan is None:
                    return False;

                if ccost < cost:
                    cost = ccost
                    self.plan = cplan

        dx, dy = self.plan.popleft()

        # move agent
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        # improve infrastructure
        x, y = self.pos[0], self.pos[1]
        if IMPROVEMENT_VAR is not None:
            x += int(random.gauss(0, 1) * IMPROVEMENT_VAR)
            y += int(random.gauss(0, 1) * IMPROVEMENT_VAR)
            x = min(max(x, 0), map.shape[0]-1)
            y = min(max(y, 0), map.shape[1]-1)

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

                movecost = BASECOST + (1.0 / map[npos[0], npos[1]])
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

    asp = MAPSIZE[0]/MAPSIZE[1]
    x, y = np.linspace(-1, 1, MAPSIZE[1]), np.linspace(-asp, asp, MAPSIZE[0])
    mesh = np.meshgrid(x, y)
    map = np.exp((- mesh[0] ** 2 - mesh[1] **2)/0.1)

map *= (np.random.rand(*MAPSIZE) * INIT_NOISE + (1.0 - INIT_NOISE))

if INIT_INVERT:
    map = rescale(map)
    map = 1.0 - map

numpixels = MAPSIZE[0] * MAPSIZE[1]

agents = [Agent(map) for _ in range(NUM_AGENTS)]

# plt.figure(figsize=(16, 9))

it = 0;

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
        imsave(f'plot.{it:06}.png', rescale(map), check_contrast=False)

    it += 1

imsave(f'final.png', rescale(map), check_contrast=False)

print('done')



