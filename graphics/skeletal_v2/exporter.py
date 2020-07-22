#filename = "/home/mat/Desktop/various/graphics/skeletal_v2/exporter.py"
#exec(compile(open(filename).read(), filename, 'exec'))

# note: convention, bp - bind pose, cp - current pose
# matrices in blender api are stored row-wise and vectors are column vectors
# bone.matrix_local: bp_mesh_from_bone
# pose.bone.matrix_basis: bone_cp_from_bp (bone space, this is not a change of basis matrix)
# todo: only default smooth normals are exported, see calc_normals_split()

import math
import bpy
import mathutils

class Vertex:
    def __init__(self):
        self.pos = None
        self.uv = None
        self.normal = None
        self.bone_ids = []
        self.weights = []

class BoneAction:
    def __init__(self):
        self.locs = []
        self.loc_time_coords = []
        self.rots = []
        self.rot_time_coords = []

def is_export_bone(bone, vertex_groups):
    if "locator" in bone.name:
        return True

    for group in vertex_groups:
        if group.name == bone.name:
            return True
    return False

def register_export_bones(bone, export_bones, vertex_groups):
    if len(export_bones) == 0 or is_export_bone(bone, vertex_groups):
        export_bones.append(bone)

    for child in bone.children:
        register_export_bones(child, export_bones, vertex_groups)

# both functions fold parents that are not exported

def get_closest_export_parent_id(parent_bone, export_bones):
    for i, bone in enumerate(export_bones):
        if bone.name == parent_bone.name:
            return i
    assert parent_bone.parent
    return get_closest_export_parent_id(parent_bone.parent, export_bones)

def get_mat_export_parent_from_bone(parent_bone, mesh_from_bone, pose_bones, export_bones):
    parent_from_bone = parent_bone.matrix_local.inverted() @ mesh_from_bone

    for bone in export_bones:
        if parent_bone.name == bone.name:
            return parent_from_bone

    mesh_from_bone = parent_bone.matrix_local @ pose_bones[parent_bone.name].matrix_basis @ parent_from_bone
    return get_mat_export_parent_from_bone(parent_bone.parent, mesh_from_bone, pose_bones, export_bones)

object = bpy.context.selected_objects[0]
assert object.type == 'ARMATURE'
armature = object.data
mesh_object = object.children[0]
assert mesh_object.type == 'MESH'
assert mesh_object.matrix_basis == mathutils.Matrix.Identity(4)
assert mesh_object.matrix_parent_inverse == mathutils.Matrix.Identity(4)
mesh = mesh_object.data
mesh.calc_loop_triangles()
vertex_groups = mesh_object.vertex_groups
assert len(vertex_groups)
action = object.animation_data.action
assert action
root_bone = None

for bone in armature.bones:
    if bone.parent == None:
        assert root_bone == None # only one root is allowed
        root_bone = bone

assert root_bone
export_bones = []
register_export_bones(root_bone, export_bones, vertex_groups)
map_groupid_boneid = {}

for group_id, group in enumerate(vertex_groups):
    for bone_id, bone in enumerate(export_bones):
        if bone.name == group.name:
            map_groupid_boneid[group_id] = bone_id
            break

format_str = "pnbw" # pos, normal, 4 bone ids, 4 weights
uv_loops = None

if mesh.uv_layers.active:
    uv_loops = mesh.uv_layers.active.data
    format_str = "punbw" # pos, uv, normal, 4 bone ids, 4 weights

vertices = []

for mv in mesh.vertices:
    vertex = Vertex()
    vertex.pos = mv.co
    vertex.normal = mv.normal
    bws = [] # (bone_id, weight) pairs

    for group in mv.groups:
        bws.append( (map_groupid_boneid[group.group], group.weight) )

    assert len(bws)
    bws = sorted(bws, key=lambda bw: bw[1], reverse=True)

    if len(bws) > 4:
        bws = bws[0:4]
    else:
        num_append = 4 - len(bws)

        for i in range(num_append):
            bws.append( (0,0) )

    mod = 0

    for bw in bws:
        mod += bw[1]

    for bw in bws:
        vertex.bone_ids.append(bw[0])
        vertex.weights.append(bw[1] / mod)

    vertices.append(vertex)

indices = []

for tri in mesh.loop_triangles:
    for loop_id in tri.loops:
        loop = mesh.loops[loop_id]
        indices.append(loop.vertex_index)
        # todo: what if loops with different uvs share the same vertex?
        if uv_loops:
            vertices[loop.vertex_index].uv = uv_loops[loop_id].uv

for channel in action.fcurves:
    if channel.mute:
        print('warning, muted animation channels') # todo: report to user
        break

# rt - restore
rt_action_blend_type = object.animation_data.action_blend_type
rt_action_influence = object.animation_data.action_influence
rt_object_mode = object.mode
rt_frame = bpy.context.scene.frame_current
rt_bone_layers = []

for layer in armature.layers:
    rt_bone_layers.append(layer)

object.animation_data.action_blend_type = 'REPLACE'
object.animation_data.action_influence = 1
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.armature.layers_show_all()
bpy.ops.pose.select_all(action='SELECT')
fps = bpy.context.scene.render.fps
bone_actions = []

for bone in export_bones:
    bone_actions.append(BoneAction())

for frame_id in range(int(action.frame_range[0]), int(action.frame_range[1] + 1)):

    # visual transform distorts the animation if not cleared (especially when switching actions), bug?
    # shouldn't frame_set() clear it?
    bpy.ops.pose.transforms_clear()
    bpy.context.scene.frame_set(frame_id)
    bpy.ops.pose.visual_transform_apply()
    time_coord = (1.0 / fps) * (frame_id - action.frame_range[0])

    for i, bone in enumerate(export_bones):

        mesh_from_bone = bone.matrix_local @ object.pose.bones[bone.name].matrix_basis
        parent_from_bone = mesh_from_bone.copy()

        if bone.parent:
            parent_from_bone = get_mat_export_parent_from_bone(bone.parent, mesh_from_bone, object.pose.bones, export_bones)

        loc, rot, scale = parent_from_bone.decompose()
        assert (scale - mathutils.Vector((1,1,1))).length < 0.001 # scaling is not supported
        bone_action = bone_actions[i]
        append_loc = False
        append_rot = False

        if len(bone_action.locs) == 0 or frame_id == action.frame_range[1]:
            append_loc = True
            append_rot = True
        else:
            prev_loc = bone_action.locs[-1]
            prev_rot = bone_action.rots[-1]

            if (loc - prev_loc).length > 0.001:
                append_loc = True

            diff_quat = rot @ prev_rot.conjugated()
            diff_angle_deg = 0

            if diff_quat.w <= 1.0: # w > 1.0 (numerical error) causes an exception
                diff_angle_deg = abs(math.degrees(2 * math.acos(diff_quat.w)))

            if diff_angle_deg > 0.1:
                append_rot = True

        if append_loc:
            bone_action.locs.append(loc)
            bone_action.loc_time_coords.append(time_coord)

        if append_rot:
            bone_action.rots.append(rot)
            bone_action.rot_time_coords.append(time_coord)

# restore state (except for selected bones)
bpy.ops.pose.transforms_clear()

for i in range(len(rt_bone_layers)):
    armature.layers[i] = rt_bone_layers[i]

bpy.ops.object.mode_set(mode=rt_object_mode)
object.animation_data.action_influence = rt_action_influence
object.animation_data.action_blend_type = rt_action_blend_type
bpy.context.scene.frame_set(rt_frame)

f = open('anim_data', 'w')
print('format', format_str, file=f)
print('vertex_count', len(vertices), file=f)

for v in vertices:
    print(v.pos.x, v.pos.y, v.pos.z, end=' ', file=f)

    if uv_loops:
        print(v.uv[0], v.uv[1], end=' ', file=f)

    print(v.normal.x, v.normal.y, v.normal.z, end=' ', file=f)

    for id in v.bone_ids:
        print(id, end=' ', file=f)

    for weight in v.weights:
        print(weight, end=' ', file=f)

    print(file=f)

print('index_count', len(indices), file=f)

for idx in indices:
    print(idx, end=' ', file=f)

print(file=f)

print('bone_count', len(export_bones), file=f)

for bone in export_bones:
    pid = -1

    if bone.parent:
        pid = get_closest_export_parent_id(bone.parent, export_bones)

    print(bone.name, pid, end=' ', file=f)
    bone_from_bp_mesh = bone.matrix_local.inverted()

    for row in bone_from_bp_mesh:
        for i in range(4):
            print(row[i], end=' ', file=f)
    print(file=f)

print('action_name', action.name, file=f)

for bone_action in bone_actions:
    print("loc_count", len(bone_action.locs), file=f)

    for i, loc in enumerate(bone_action.locs):
        print(loc.x, loc.y, loc.z, bone_action.loc_time_coords[i], file=f)

    print("rot_count", len(bone_action.rots), file=f)

    for i, rot in enumerate(bone_action.rots):
        print(rot.x, rot.y, rot.z, rot.w, bone_action.rot_time_coords[i], file=f)

f.close()
