# place inside scripts/startup and call bpy.export() from the blender console

#filename = "/home/mat/various/graphics/skeletal_v2/exporter.py"
#exec(compile(open(filename).read(), filename, 'exec'))

# note: convention, bp - bind pose, cp - current pose
# matrices in blender api are stored row-wise and vectors are column vectors
# bone.matrix_local: bp_mesh_from_bone
# pose.bone.matrix_basis: bone_cp_from_bp (bone space, this is not a change of basis matrix)
# todo: only default smooth normals are exported, see calc_normals_split()

import math
import bpy
import mathutils
import copy

# data not related to a specific face
class TmpVertex:
    def __init__(self):
        self.pos = None
        self.bone_ids = []
        self.weights = []

# specific face data
class TmpSubVertex:
    def __init__(self, uv, normal):
        self.uv = uv
        self.normal = normal

class CompIdx:
    def __init__(self, vert_idx, sub_idx):
        self.vert_idx = vert_idx
        self.sub_idx = sub_idx

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

class Action:
    def __init__(self, name, bone_actions):
        self.name = name
        self.bone_actions = bone_actions

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

def process_mesh(mesh, map_groupid_boneid):

    mesh.calc_loop_triangles()
    mesh.calc_normals_split()
    format_str = "pnbw" # pos, normal, 4 bone ids, 4 weights
    uv_loops = None

    if mesh.uv_layers.active:
        uv_loops = mesh.uv_layers.active.data
        format_str = "punbw" # pos, uv, normal, 4 bone ids, 4 weights

    tmp_verts = []

    for mv in mesh.vertices:
        vert = TmpVertex()
        vert.pos = mv.co
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
            vert.bone_ids.append(bw[0])
            vert.weights.append(bw[1] / mod)

        tmp_verts.append(vert)

    sub_verts_list = [ [] for _ in range(len(tmp_verts)) ]
    comp_ids = []

    for tri in mesh.loop_triangles:
        for loop_id in tri.loops:

            vert_idx = mesh.loops[loop_id].vertex_index
            normal = mesh.loops[loop_id].normal
            uv = [0.0,0.0]

            if uv_loops:
                uv = uv_loops[loop_id].uv

            sub_verts = sub_verts_list[vert_idx]
            sub_idx = 0

            for sub_vert in sub_verts:
                if uv == sub_vert.uv and normal.dot(sub_vert.normal) > math.cos(math.radians(1)):
                    break
                sub_idx += 1

            if sub_idx == len(sub_verts):
                sub_verts.append( TmpSubVertex(uv, normal) )

            comp_ids.append( CompIdx(vert_idx, sub_idx) )

    # build final vertex and index buffers

    vertices = []
    offsets = []
    offset = 0

    for vid in range(len(tmp_verts)):
        offsets.append(offset)
        vert = Vertex()
        vert.pos = tmp_verts[vid].pos
        vert.bone_ids = tmp_verts[vid].bone_ids
        vert.weights = tmp_verts[vid].weights
        assert len(sub_verts_list[vid])

        for sub_vert in sub_verts_list[vid]:
            offset += 1
            vert.uv = sub_vert.uv
            vert.normal = sub_vert.normal
            vertices.append(copy.deepcopy(vert))

    indices = []

    for comp_idx in comp_ids:
        indices.append( offsets[comp_idx.vert_idx] + comp_idx.sub_idx )

    return format_str, vertices, indices

def export(filename, action_names = [], export_mesh = True):

    assert bpy.context.object
    object = bpy.context.object
    assert object.type == 'ARMATURE'
    armature = object.data
    mesh_object = None

    for child in object.children:
        if child.type == 'MESH':
            if mesh_object:
                print("armature with multiple meshes is not supported")
                assert False
            mesh_object = child

    assert mesh_object
    assert mesh_object.matrix_basis == mathutils.Matrix.Identity(4)
    assert mesh_object.matrix_parent_inverse == mathutils.Matrix.Identity(4)
    mesh = mesh_object.data
    vertex_groups = mesh_object.vertex_groups
    assert len(vertex_groups)
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

    format_str = "null"
    vertices = []
    indices = []

    if export_mesh:
        format_str, vertices, indices = process_mesh(mesh, map_groupid_boneid)

    export_actions = []

    if len(action_names):

        assert object.animation_data
        # rt - restore
        rt_object_mode = object.mode
        rt_frame = bpy.context.scene.frame_current
        rt_action = object.animation_data.action
        rt_action_blend_type = object.animation_data.action_blend_type
        rt_action_influence = object.animation_data.action_influence
        rt_bone_layers = []

        for layer in armature.layers:
            rt_bone_layers.append(layer)

        bpy.ops.object.mode_set(mode='POSE')
        object.animation_data.action_blend_type = 'REPLACE'
        object.animation_data.action_influence = 1
        bpy.ops.armature.layers_show_all()
        bpy.ops.pose.select_all(action='SELECT')

        for action_name in action_names:

            action = bpy.data.actions[action_name]
            assert action
            object.animation_data.action = action

            for channel in action.fcurves:
                if channel.mute:
                    print(f'warning [{action_name}]: muted animation channels')
                    break

            bone_actions = []

            for bone in export_bones:
                bone_actions.append(BoneAction())

            for frame_id in range(int(action.frame_range[0]), int(action.frame_range[1] + 1)):

                # visual transform distorts the animation if not cleared (especially when switching actions), bug?
                # shouldn't frame_set() clear it?
                bpy.ops.pose.transforms_clear()
                bpy.context.scene.frame_set(frame_id)
                bpy.ops.pose.visual_transform_apply()
                time_coord = (1.0 / bpy.context.scene.render.fps) * (frame_id - action.frame_range[0])

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

                        if (loc - prev_loc).length > 0.00001:
                            append_loc = True

                        diff_quat = rot @ prev_rot.conjugated()
                        diff_angle_deg = 0

                        if diff_quat.w <= 1.0: # w > 1.0 (numerical error) causes an exception
                            diff_angle_deg = abs(math.degrees(2 * math.acos(diff_quat.w)))

                        if diff_angle_deg > 0.00001:
                            append_rot = True

                    if append_loc:
                        bone_action.locs.append(loc)
                        bone_action.loc_time_coords.append(time_coord)

                    if append_rot:
                        bone_action.rots.append(rot)
                        bone_action.rot_time_coords.append(time_coord)

            # this makes it eaiser for a runtime to handle single frame actions
            if len(bone_actions[0].locs) == 1:
                for bone_action in bone_actions:
                    bone_action.locs.append(bone_action.locs[0])
                    bone_action.rots.append(bone_action.rots[0])

            export_actions.append(Action(action_name, bone_actions))

        # restore state (except for selected bones)
        bpy.ops.pose.transforms_clear()

        for i in range(len(rt_bone_layers)):
            armature.layers[i] = rt_bone_layers[i]

        object.animation_data.action_influence = rt_action_influence
        object.animation_data.action_blend_type = rt_action_blend_type
        object.animation_data.action = rt_action
        bpy.context.scene.frame_set(rt_frame)
        bpy.ops.object.mode_set(mode=rt_object_mode)

    action_bone_count = len(export_bones)

    if not export_mesh:
        export_bones = []

    f = open(filename, 'w')
    print('format', format_str, file=f)
    print('vertex_count', len(vertices), file=f)

    for v in vertices:
        print(v.pos.x, v.pos.y, v.pos.z, end=' ', file=f)

        if 'u' in format_str:
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

    if len(indices):
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

    print('bone_count', action_bone_count, file=f) # redundant but makes parsing easier
    print('action_count', len(export_actions), file=f)

    for action in export_actions:
        print('action_name', action.name, file=f)

        for bone_action in action.bone_actions:
            print("loc_count", len(bone_action.locs), file=f)

            for i, loc in enumerate(bone_action.locs):
                print(loc.x, loc.y, loc.z, bone_action.loc_time_coords[i], file=f)

            print("rot_count", len(bone_action.rots), file=f)

            for i, rot in enumerate(bone_action.rots):
                print(rot.x, rot.y, rot.z, rot.w, bone_action.rot_time_coords[i], file=f)

    f.close()
    print('done!')

def register():
    bpy.export = export

def unregister():
    del bpy.export

if __name__ == "__main__":
    register()
