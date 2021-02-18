import math
import random
import functools
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

class Vertex:
    def __init__(self, parent_id, depth, heuristic):
        self.parent_id = parent_id
        self.depth = depth
        self.heuristic = heuristic

num_rows = 32
row_size = 32

def idx_to_pos(idx):
    x = idx % row_size
    y = num_rows - 1 - math.floor(idx / row_size)
    assert y >= 0 and y < num_rows
    return (x,y)

def pos_to_idx(pos):
    idx = (num_rows - 1 - pos[1]) * row_size + pos[0]
    assert idx >= 0 and idx < len(map)
    return idx

def vertex_cmp(lhs_kv, rhs_kv):
    lhs = lhs_kv[1]
    rhs = rhs_kv[1]
    return (lhs.depth + lhs.heuristic) - (rhs.depth + rhs.heuristic)

def gen_map():
    map = [0] * (num_rows * row_size)
    nblocks = int(len(map) * 0.3)

    for _ in range(nblocks):
        id = random.randint(0, len(map)-1)
        map[id] = 1
    return map

def get_heuristic(p1, p2):
    return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

def displace_point(p, disp):
    return (p[0] + disp[0], p[1] + disp[1])

# the invariant of the algorithm is that visited nodes have the shortest possible path
# already set

def find_path(map, pos_start, pos_end, star = False):
    active = {}
    traversed = {}
    endvid = None

    if star:
        active[pos_to_idx(pos_start)] = Vertex(None, 0, get_heuristic(pos_start, pos_end))
    else:
        active[pos_to_idx(pos_start)] = Vertex(None, 0, 0)

    while len(active):
        # extract an element with the highest priority
        # elements are not sorted by the priority because we access them more often by
        # the id
        vid, vertex = min(active.items(), key = functools.cmp_to_key(vertex_cmp))
        del active[vid]
        # place in the visited nodes
        assert not vid in traversed
        traversed[vid] = vertex

        pos = idx_to_pos(vid)

        if pos == pos_end:
            endvid = vid
            break

        to_queue_temp = []

        for x in range(-1,2):
            for y in range(-1,2):
                if x or y:
                    to_queue_temp.append(displace_point(pos, (x,y)))

        to_queue = []
        # remove vertices outside map bounds
        for p in to_queue_temp:
            if p[0] < 0 or p[0] >= row_size or p[1] < 0 or p[1] >= num_rows:
                continue
            to_queue.append(p)

        for new_pos in to_queue:
            new_id = pos_to_idx(new_pos)
            # don't revisit traversed nodes
            if new_id in traversed:
                continue
            # obstructed area
            if map[new_id] == 1:
                continue
            depth = vertex.depth + get_heuristic(new_pos, pos)
            heuristic = 0
            if star:
                heuristic = get_heuristic(idx_to_pos(new_id), pos_end)

            if new_id in active:
                if depth < active[new_id].depth:
                    active[new_id] = Vertex(vid, depth, heuristic)
            else:
                active[new_id] = Vertex(vid, depth, heuristic)

    if endvid is None:
        return None, None

    path = []
    vid = endvid

    while vid is not None:
        path.append(idx_to_pos(vid))
        vid = traversed[vid].parent_id
    path.reverse()
    return path, traversed

def draw_tile(ax, pos, col, border = 0.04):
    pos = pos[0] + border, pos[1] + border
    size = 1 - 2 * border
    rect = plt_patches.Rectangle(pos, size, size, color = col)
    ax.add_patch(rect)

def draw(ax, map, path, traversed):

    for id, v in enumerate(map):
        col = (0.8,0.8,0.8) if v == 0 else 'k'
        draw_tile(ax, idx_to_pos(id), col)

    for id, v in traversed.items():
        draw_tile(ax, idx_to_pos(id), 'g')

    for pos in path:
        c = 'r'
        if pos == path[-1]:
            c = 'y'
        draw_tile(ax, pos, c, 0.2)

    ax.annotate('path length: ' + str(len(path)), (0,-2))
    ax.annotate('visited nodes: ' + str(len(traversed)), (0, -4))

pos_start = (0,0)
pos_end = (row_size - 1, num_rows - 1)
map = None
path = None
traversed = None
path_star = None
traversed_star = None

# we are randomly generating the map so the path may be not always traversable
while path is None:
    map = gen_map()
    path, traversed = find_path(map, pos_start, pos_end)
    path_star, traversed_star = find_path(map, pos_start, pos_end, True)
print('done, now wating for the slow matplotlib software to render things')

fig, axs = plt.subplots(1, 2)
draw(axs[0], map, path, traversed)
draw(axs[1], map, path_star, traversed_star)
dim = max(row_size, num_rows)

for ax in axs:
    ax.axis([0, dim, 0, dim])
    ax.axis('equal')

plt.show()
