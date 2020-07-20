#filename = "/home/mat/Desktop/various/graphics/skeletal_v2/exporter.py"
#exec(compile(open(filename).read(), filename, 'exec'))

import bpy
import mathutils

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
vertex_groups = mesh_object.vertex_groups
action = object.animation_data.action
root_bone = None

for bone in armature.bones:
    if bone.parent == None:
        assert root_bone == None # only one root is allowed
        root_bone = bone

assert root_bone
export_bones = []
register_export_bones(root_bone, export_bones, vertex_groups)

f = open('anim_data', 'w')

# positions

for vert in mesh.vertices:
    print('v ', vert.co.x, vert.co.y, vert.co.z, file=f)

# normals

for vert in mesh.vertices:
    print('n ', vert.normal.x, vert.normal.y, vert.normal.z, file=f)

# weights / bone ids

map_groupid_boneid = {}

for group_id, group in enumerate(vertex_groups):
    for bone_id, bone in enumerate(export_bones):
        if bone.name == group.name:
            map_groupid_boneid[group_id] = bone_id
            break

for vert in mesh.vertices:
    weights = []

    for group in vert.groups:
        weights.append( (map_groupid_boneid[group.group], group.weight) )

    assert len(weights)
    weights = sorted(weights, key=lambda tup: tup[1], reverse=True)

    if len(weights) > 4:
        weights = weights[0:4]
    else:
        num_append = 4 - len(weights)

        for i in range(num_append):
            weights.append( (0,0) )

    mod = 0

    for tup in weights:
        mod += tup[1]

    for i, tup in enumerate(weights):
        weights[i] = (tup[0], tup[1] / mod)

    print('w', end=' ', file=f)

    for tup in weights:
        print(tup[0], tup[1], end=' ', file=f)
    print(file=f)

# faces

for poly in mesh.polygons:
    print('f', end=' ', file=f)
    assert len(poly.vertices) == 3

    for vert_id in poly.vertices:
        print(vert_id, end=' ', file=f)
    print(file=f)

# bones
# note: convention, bp - bind pose, cp - current pose
# matrices in blender api are stored row-wise and vectors are column vectors
# bone.matrix_local: bp_mesh_from_bone
# pose_bone.matrix_basis: bone_cp_from_bp (bone space, this is not a change of basis matrix)

for bone in export_bones:
    pid = -1

    if bone.parent:
        pid = get_closest_export_parent_id(bone.parent, export_bones)

    print('b', pid, end=' ', file=f)

    for row in bone.matrix_local: # todo: export inverse
        for i in range(4):
            print(row[i], end=' ', file=f)
    print(file=f)

# action
# note: rt - restore

rt_action_blend_type = object.animation_data.action_blend_type
rt_action_influence = object.animation_data.action_influence
rt_object_mode = object.mode
rt_bone_layers = []

for layer in armature.layers:
    rt_bone_layers.append(layer)

rt_frame = bpy.context.scene.frame_current

object.animation_data.action_blend_type = 'REPLACE'
object.animation_data.action_influence = 1
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.armature.layers_show_all()
bpy.ops.pose.select_all(action='SELECT')
pose_bones = object.pose.bones

for bone in export_bones:
    for i in range(int(action.frame_range[0]), int(action.frame_range[1] + 1)):

        # visual transform distorts the animation if not cleared (especially when switching actions), bug?
        # shouldn't frame_set() clear it?
        bpy.ops.pose.transforms_clear()
        bpy.context.scene.frame_set(i)
        bpy.ops.pose.visual_transform_apply()
        mesh_from_bone = bone.matrix_local @ pose_bones[bone.name].matrix_basis
        parent_from_bone = mesh_from_bone.copy()

        if bone.parent:
            parent_from_bone = get_mat_export_parent_from_bone(bone.parent, mesh_from_bone, pose_bones, export_bones)

        loc, rot, scale = parent_from_bone.decompose()
        print('s', loc.x, loc.y, loc.z, rot.w, rot.x, rot.y, rot.z, file=f)

f.close()
# restore state (except for selected bones)
bpy.ops.pose.transforms_clear()
bpy.context.scene.frame_set(rt_frame)

for i in range(len(rt_bone_layers)):
    armature.layers[i] = rt_bone_layers[i]

bpy.ops.object.mode_set(mode=rt_object_mode)
object.animation_data.action_influence = rt_action_influence
object.animation_data.action_blend_type = rt_action_blend_type
