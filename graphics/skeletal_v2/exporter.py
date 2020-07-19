#filename = "/home/mat/Desktop/various/graphics/skeletal_v2/exporter.py"
#exec(compile(open(filename).read(), filename, 'exec'))

import bpy
import mathutils

def register_export_bones(bone, export_bones, vertex_groups):
    if len(export_bones) == 0: # root bone is exported even if is not deforming
        export_bones.append(bone)
    else:
        for group in vertex_groups:
            if group.name == bone.name:
                export_bones.append(bone)
                break

    for child in bone.children:
        register_export_bones(child, export_bones, vertex_groups)

object = bpy.context.selected_objects[0]
assert object.type == 'ARMATURE'
armature = object.data
mesh_object = object.children[0]
mesh = mesh_object.data
vertex_groups = mesh_object.vertex_groups
action = object.animation_data.action
assert len(armature.bones)
root_bone = None

for bone in armature.bones:
    if bone.parent == None:
        assert root_bone == None # only onre root allowed
        root_bone = bone

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

for vert in mesh.vertices:
    weights = []

    for group in vert.groups:
        target_id = None
        target_name = vertex_groups[group.group].name

        for id, bone in enumerate(export_bones):
            if target_name == bone.name:
                target_id = id
                break

        assert target_id
        weights.append( (target_id, group.weight) )

    assert len(weights) > 0
    weights = sorted(weights, key=lambda tup: tup[1], reverse=True)
    num_append = 4 - len(weights)

    for i in range(num_append):
        weights.append( (0,0) )

    if len(weights) > 4:
        weights = weights[0:4]

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
    # todo, triangulate mesh
    assert len(poly.vertices) == 3

    for vert_id in poly.vertices:
        print(vert_id, end=' ', file=f)

    print(file=f)

# bones
# note: matrices in blender api are stored row-wise and vectors are treated as column vectors
# matrix_local transformation: bp_mesh_from_bone (bp - bind pose)

for bone in export_bones:
    pid = -1

    if bone.parent:
        pid = export_bones.index(bone.parent) # only a deforming bone can be a parent of another deforming bone

    print('b', pid, end=' ', file=f)

    for row in bone.matrix_local:
        for i in range(4):
            print(row[i], end=' ', file=f)

    print(file=f)

# action

if bpy.context.object.mode != 'POSE':
    bpy.ops.object.posemode_toggle()

bpy.ops.armature.layers_show_all()
frame_to_restore = bpy.context.scene.frame_current
bpy.ops.pose.select_all(action='SELECT')

for bone in export_bones:
    for i in range(int(action.frame_range[0]), int(action.frame_range[1] + 1)):
        bpy.ops.pose.transforms_clear() # visual transform distorts the animation if not cleared (especially when changing actions), don't know why
        bpy.context.scene.frame_set(i)
        bpy.ops.pose.visual_transform_apply()
        tf_bone_space = None

        for pose_bone in object.pose.bones:
            if pose_bone.name == bone.name:
                tf_bone_space = pose_bone.matrix_basis
                break

        assert tf_bone_space
        parent_from_mesh = None

        if bone.parent:
            parent_from_mesh = bone.parent.matrix_local.inverted()
        else:
            parent_from_mesh = mathutils.Matrix.Identity(4)

        parent_from_bone = parent_from_mesh @ bone.matrix_local @ tf_bone_space
        loc, rot, scale = parent_from_bone.decompose()
        print('s', loc.x, loc.y, loc.z, rot.w, rot.x, rot.y, rot.z, file=f)

f.close()
bpy.context.scene.frame_set(frame_to_restore)
