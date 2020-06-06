# this script exports a selected object (mesh) and its armature and a current action

# limitations:
# exports a single mesh and a single action
# mesh is expected to have triangular faces (polygons), each bone must have loc/rot channels and
# each channel of each bone must have the same number of keyframe_points in the current action
# bone scale transformations are not exported
# probably all mesh transforms and modifiers (except armature) should be applied before running this script
# keyframe_points are expected to be evenly spaced in a time domain
# animation duration in seconds is not exported
# todo: bones which don't affect skin are exported, this is not necessary
# todo: bones with no animation are also exported

import bpy
import mathutils

def register_bone(bones, bone):
    bones.append(bone)
    for child in bone.children:
        register_bone(bones, child)

object = bpy.context.selected_objects[0]
assert object
assert object.type == 'MESH'
armature_object = object.find_armature()
assert armature_object
mesh = object.data
armature = armature_object.data
vertex_groups = object.vertex_groups
bones = []
assert armature.bones[0].parent == None # root bone
register_bone(bones, armature.bones[0])
action = object.animation_data.action
assert action
group_bone_id_map = {}

for group in vertex_groups:
    bone_id = None
    
    for id, bone in enumerate(bones):
        if(bone.name == group.name):
            bone_id = id
            break
    assert bone_id != None
    group_bone_id_map[group.index] = bone_id

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
        weights.append( (group_bone_id_map[group.group], group.weight) )
    
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

matrices_parent_from_bone = []

for bone in bones:
    print('b', end=' ', file=f)
    parent_id = 0
    parent_from_bone = bone.matrix_local # matrix_local transformation: bp_mesh_from_bone (bp - bind pose)
    
    #if not a root bone
    if(bone.parent):
        parent_id = bones.index(bone.parent)
        tmp = bone.parent.matrix_local.copy()
        tmp.invert()
        # tmp is now: parent_from_bp_model
        parent_from_bone = tmp @ bone.matrix_local
    
    matrices_parent_from_bone.append(parent_from_bone)
    print(parent_id, end=' ', file=f)
    
    for row in bone.matrix_local:
        for i in range(4):
            print(row[i], end=' ', file=f)
    
    print(file=f)

# animation (blender action)

sample_count = len(action.fcurves[0].keyframe_points)

for bone_id, bone in enumerate(bones):
    channels = []
    
    for fcurve in action.fcurves:
        if bone.name == fcurve.group.name:
            channels.append(fcurve.keyframe_points)
            
    assert len(channels) >= 7 # 3 location channels and 4 rotation channels
    assert len(channels[0]) == sample_count
    assert len(channels[3]) == sample_count
    parent_from_bone = matrices_parent_from_bone[bone_id]
    
    for i in range(sample_count):
        print('s', end=' ', file=f)
        loc = mathutils.Vector( (channels[0][i].co.y, channels[1][i].co.y, channels[2][i].co.y) )
        rot = mathutils.Quaternion( (channels[3][i].co.y, channels[4][i].co.y, channels[5][i].co.y, channels[6][i].co.y) )
        bone_transform = mathutils.Matrix.Translation(loc) @ mathutils.Matrix.to_4x4( rot.to_matrix() )
        assert bone_transform.row[3].w == 1 # test if to_4x4 returns a correct matrix
        parent_from_bone2 = parent_from_bone @ bone_transform
        loc2, rot2, sc2 = parent_from_bone2.decompose()
        print(loc2.x, loc2.y, loc2.z, rot2.w, rot2.x, rot2.y, rot2.z, file=f)
f.close()
