#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "../main.hpp"

#define MAX_BONES 128

static const char* src_vert_skel = R"(
#version 330

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
uniform mat4 skinning_matrices[128];

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in ivec4 bone_ids;
layout(location = 3) in vec4 weights;

// in world space
out vec3 frag_pos;
out vec3 frag_normal;

void main()
{
    mat4 cp_from_bp = mat4(0);

    for(int i = 0; i < 4; ++i)
        cp_from_bp += skinning_matrices[bone_ids[i]] * weights[i];

    mat4 cp_model = model * cp_from_bp;
    vec4 pos_w = cp_model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = mat3(cp_model) * normal;
}
)";

struct Vertex2
{
    vec3 pos;
    vec3 normal;
    int bone_ids[4];
    vec4 weights;
};

struct Mesh2
{
    Vertex2* vertices;
    int vertex_count;
    int* indices;
    int index_count;
    GLuint bo;
    GLuint vao;
    char** bone_names;
    mat4* bone_f_bp_mesh;
    int* bone_parent_ids;
    int bone_count;
    int ebo_offset;
};

struct BoneAction
{
    vec3* locs;
    vec4* rots;
    float* loc_time_coords;
    float* rot_time_coords;
    int loc_count;
    int rot_count;
};

struct Action
{
    const char* name;
    BoneAction* bone_actions;
    int bone_count;
    float duration;
};

void load2(const char* filename, Mesh2& mesh, std::vector<Action>& actions)
{
    mesh = {};
    FILE* file = fopen(filename, "r");
    assert(file);
    char str_buf[256];
    int r = fscanf(file, " format %s", str_buf);
    assert(r == 1);

    bool uvs = false;

    if(strcmp(str_buf, "punbw") == 0)
        uvs = true;

    r = fscanf(file, " vertex_count %d", &mesh.vertex_count);
    assert(r == 1);
    mesh.vertices = (Vertex2*)malloc(mesh.vertex_count * sizeof(Vertex2));

    for(int i = 0; i < mesh.vertex_count; ++i)
    {
        Vertex2& v = mesh.vertices[i];
        r = fscanf(file, "%f %f %f", &v.pos.x, &v.pos.y, &v.pos.z);
        assert(r == 3);

        if(uvs)
        {
            vec2 dummy;
            r = fscanf(file, "%f %f", &dummy.x, &dummy.y);
            assert(r == 2);
        }
        r = fscanf(file, "%f %f %f", &v.normal.x, &v.normal.y, &v.normal.z);
        assert(r == 3);

        for(int i = 0; i < 4; ++i)
        {
            r = fscanf(file, "%d", v.bone_ids + i);
            assert(r == 1);
        }

        for(int i = 0; i < 4; ++i)
        {
            r = fscanf(file, "%f", &v.weights[i]);
            assert(r == 1);
        }
    }

    r = fscanf(file, " index_count %d", &mesh.index_count);
    assert(r == 1);
    mesh.indices = (int*)malloc(mesh.index_count * sizeof(int));

    for(int i = 0; i < mesh.index_count; ++i)
    {
        r = fscanf(file, "%d", mesh.indices + i);
        assert(r == 1);
    }

    r = fscanf(file, " bone_count %d", &mesh.bone_count);
    assert(r == 1);
    mesh.bone_names = (char**)malloc(mesh.bone_count * sizeof(char*));
    mesh.bone_parent_ids = (int*)malloc(mesh.bone_count * sizeof(int));
    mesh.bone_f_bp_mesh = (mat4*)malloc(mesh.bone_count * sizeof(mat4));

    for(int bone_id = 0; bone_id < mesh.bone_count; ++bone_id)
    {
        r = fscanf(file, " %s ", str_buf);
        assert(r == 1);
        mesh.bone_names[bone_id] = strdup(str_buf);
        r = fscanf(file, "%d", mesh.bone_parent_ids + bone_id);
        assert(r == 1);

        for(int i = 0; i < 16; ++i)
        {
            r = fscanf(file, "%f", mesh.bone_f_bp_mesh[bone_id].data + i);
            assert(r == 1);
        }
    }
    int bone_count;
    r = fscanf(file, " bone_count %d", &bone_count);
    assert(r == 1);
    int action_count;
    r = fscanf(file, " action_count %d", &action_count);
    assert(r == 1);

    for(int _i = 0; _i < action_count; ++_i)
    {
        Action action;
        action.bone_count = bone_count;
        r = fscanf(file, " action_name %s", str_buf);
        assert(r == 1);
        action.name = strdup(str_buf);
        action.bone_actions = (BoneAction*)malloc(action.bone_count * sizeof(BoneAction));

        for(int bone_id = 0; bone_id < action.bone_count; ++bone_id)
        {
            BoneAction& ba = action.bone_actions[bone_id];
            r = fscanf(file, " loc_count %d", &ba.loc_count);
            assert(r == 1);
            ba.locs = (vec3*)malloc(ba.loc_count * sizeof(vec3));
            ba.loc_time_coords = (float*)malloc(ba.loc_count * sizeof(float));

            for(int i = 0; i < ba.loc_count; ++i)
            {
                r = fscanf(file, "%f %f %f %f", &ba.locs[i].x, &ba.locs[i].y, &ba.locs[i].z, ba.loc_time_coords + i);
                assert(r == 4);
            }

            r = fscanf(file, " rot_count %d", &ba.rot_count);
            assert(r == 1);
            ba.rots = (vec4*)malloc(ba.rot_count * sizeof(vec4));
            ba.rot_time_coords = (float*)malloc(ba.rot_count * sizeof(float));

            for(int i = 0; i < ba.rot_count; ++i)
            {
                r = fscanf(file, "%f %f %f %f %f", &ba.rots[i].x, &ba.rots[i].y, &ba.rots[i].z, &ba.rots[i].w, ba.rot_time_coords + i);
                assert(r == 5);
            }
        }
        assert(action.bone_count);
        int i = action.bone_actions[0].loc_count - 1;
        action.duration = action.bone_actions[0].loc_time_coords[i];
        actions.push_back(action);
    }

    r = fscanf(file, " %s", str_buf);
    assert(r == EOF);
    fclose(file);

    if(mesh.vertex_count == 0)
        return;
    mesh.ebo_offset = mesh.vertex_count * sizeof(Vertex2);
    glGenBuffers(1, &mesh.bo);
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.bo);
    glBufferData(GL_ARRAY_BUFFER, mesh.index_count * sizeof(int) + mesh.vertex_count * sizeof(Vertex2), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, mesh.ebo_offset, mesh.vertices);
    glBufferSubData(GL_ARRAY_BUFFER, mesh.ebo_offset, mesh.index_count * sizeof(int), mesh.indices);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex2), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex2), (void*)offsetof(Vertex2, normal));
    glVertexAttribIPointer(2, 4, GL_INT, sizeof(Vertex2), (void*)offsetof(Vertex2, bone_ids));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex2), (void*)offsetof(Vertex2, weights));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.bo);
}

enum ActionState
{
    AS_IDLE,
    AS_RUN,
    AS_RUN_BACK,
    AS_JUMP,
    AS_FALL,
    AS_LAND,
    AS_COUNT,
};

struct AnimCtrl
{
    // output
    mat4 model_tf;
    mat4* skinning_mats;
    mat4* cp_model_f_bone;
    // state
    mat4* skinning_mats_prev;
    mat4* add_bone_rots;
    Mesh2* mesh;
    mat4 adjust_tf;
    vec3 model_dir;
    vec3 vel_xz;
    float action_time;
    float jump_angle;
    float jump_time;
    ActionState state;
    Action* actions[AS_COUNT];
    float blend_time;
};

void init(AnimCtrl& ctrl)
{
    ctrl = {};
    int size = MAX_BONES * sizeof(mat4);
    ctrl.skinning_mats = (mat4*)malloc(size);
    ctrl.skinning_mats_prev = (mat4*)malloc(size);
    ctrl.cp_model_f_bone = (mat4*)malloc(size);
    ctrl.add_bone_rots = (mat4*)malloc(size);

    for(int i = 0; i < MAX_BONES; ++i)
        ctrl.add_bone_rots[i] = identity4();
}

struct UpdateBoneInput
{
    Mesh2* mesh;
    Action* action;
    mat4* skinning_mats;
    mat4* cp_model_f_bone;
    mat4* add_bone_rots; // can be nullptr
    float* action_time;
};

void update_bone_data(UpdateBoneInput in)
{
    float time = *in.action_time;
    float dur = in.action->duration;
    assert(in.mesh->bone_count == in.action->bone_count);

    if(time > dur)
        time = min(dur, time - dur);
    else if(time < 0)
        time = max(0, time + dur);

    for(int bone_id = 0; bone_id < in.mesh->bone_count; ++bone_id)
    {
        BoneAction& action = in.action->bone_actions[bone_id];
        int loc_id = 0;
        int rot_id = 0;

        while(time > action.loc_time_coords[loc_id + 1])
            loc_id += 1;

        while(time > action.rot_time_coords[rot_id + 1])
            rot_id += 1;

        assert(loc_id < action.loc_count - 1);
        assert(rot_id < action.rot_count - 1);
        float loc_lhs_t = action.loc_time_coords[loc_id];
        float loc_rhs_t = action.loc_time_coords[loc_id + 1];
        float rot_lhs_t = action.rot_time_coords[rot_id];
        float rot_rhs_t = action.rot_time_coords[rot_id + 1];
        float loc_t = (time - loc_lhs_t) / (loc_rhs_t - loc_lhs_t);
        float rot_t = (time - rot_lhs_t) / (rot_rhs_t - rot_lhs_t);
        vec3 loc_lhs = action.locs[loc_id];
        vec3 loc_rhs = action.locs[loc_id + 1];
        vec4 rot_lhs = action.rots[rot_id];
        vec4 rot_rhs = action.rots[rot_id + 1];

        // interpolate through the shorter path
        if(dot(rot_lhs, rot_rhs) < 0)
            rot_lhs = -1 * rot_lhs;

        vec3 loc = ((1 - loc_t) * loc_lhs) + (loc_t * loc_rhs);
        vec4 rot = ((1 - rot_t) * rot_lhs) + (rot_t * rot_rhs);
        rot = normalize(rot); // linear interpolation does not preserve length (quat_to_mat4() requires a unit quaternion)
        mat4 parent_f_bone = in.add_bone_rots ? translate(loc) * quat_to_mat4(rot) * in.add_bone_rots[bone_id] :
            translate(loc) * quat_to_mat4(rot);
        mat4 cp_model_f_parent = identity4();

        if(bone_id > 0)
        {
            int pid = in.mesh->bone_parent_ids[bone_id];
            assert(pid < bone_id);
            cp_model_f_parent = in.cp_model_f_bone[pid];
        }
        in.cp_model_f_bone[bone_id] = cp_model_f_parent * parent_f_bone;
        in.skinning_mats[bone_id] = in.cp_model_f_bone[bone_id] * in.mesh->bone_f_bp_mesh[bone_id];
    }
    *in.action_time = time;
}

static const char* _vert = R"(
#version 330

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

// in world space
out vec3 frag_pos;
out vec3 frag_normal;

void main()
{
    vec4 pos_w = model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = inverse( transpose(mat3(model)) ) * normal;
}
)";

static const char* _frag = R"(
#version 330

uniform vec3 light_intensity;
uniform vec3 light_dir;
uniform float ambient_intensity;
uniform vec3 diffuse_color;
uniform vec3 specular_color;
uniform float specular_exp;
uniform vec3 eye_pos;

in vec3 frag_pos;
in vec3 frag_normal;
out vec3 out_color;

void main()
{
    vec3 L = light_dir;
    vec3 N = normalize(frag_normal);
    vec3 ambient_comp = diffuse_color * ambient_intensity;
    vec3 diff_comp = diffuse_color * light_intensity * max(dot(N, L), 0);

    vec3 V = normalize(eye_pos - frag_pos);
    vec3 H = normalize(V + L);
    vec3 spec_comp = specular_color * light_intensity * pow( max(dot(N, H), 0), specular_exp) * float(dot(N, L) > 0);

    out_color = ambient_comp + diff_comp + spec_comp;
    out_color = pow(out_color, vec3(1/2.2));
}
)";

struct Vertex
{
    vec3 pos;
    vec3 normal;
};

struct Mesh
{
    Vertex* vertices;
    int vertex_count;
    GLuint vao;
    GLuint vbo;
};

struct Object
{
    Mesh* mesh;
    vec3 pos;
    mat4 rot;
    vec3 scale;
};

Mesh load_mesh(const char* filename)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    std::vector<vec3> tmp_positions;
    std::vector<vec3> tmp_normals;

    for(;;)
    {
        int code = fgetc(file);

        if(code != 'v')
        {
            ungetc(code, file);
            break;
        }
        code = fgetc(file);
        vec3 v;
        int n = fscanf(file, " %f %f %f ", &v.x, &v.y, &v.z);
        assert(n == 3);

        if(code == ' ')
            tmp_positions.push_back(v);
        else if(code == 'n')
            tmp_normals.push_back(normalize(v));
        else
            assert(false);
    }
    assert(tmp_positions.size());
    assert(tmp_normals.size());
    std::vector<int> indices_n;
    std::vector<int> indices_p;

    while(fgetc(file) == 'f')
    {
        for(int i = 0; i < 3; ++i)
        {
            int pos_id, norm_id;
            int n = fscanf(file, " %d//%d ", &pos_id, &norm_id);
            assert(n == 2);
            indices_p.push_back(pos_id - 1);
            indices_n.push_back(norm_id - 1);
        }
    }
    assert(indices_p.size() == indices_n.size());
    Mesh mesh;
    mesh.vertex_count = indices_p.size();
    assert(mesh.vertex_count % 3 == 0);
    mesh.vertices = (Vertex*)malloc(mesh.vertex_count * sizeof(Vertex));

    for(int i = 0; i < mesh.vertex_count; ++i)
        mesh.vertices[i] = { tmp_positions[indices_p[i]], tmp_normals[indices_n[i]] };

    glGenBuffers(1, &mesh.vbo);
    glGenVertexArrays(1, &mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertex_count * sizeof(Vertex), mesh.vertices, GL_STATIC_DRAW);
    glBindVertexArray(mesh.vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(vec3));

    return mesh;
}

GLuint create_program(const char* vert, const char* frag)
{
    GLuint program = glCreateProgram();
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vert_shader, 1, &vert, nullptr);
    glCompileShader(vert_shader);
    glShaderSource(frag_shader, 1, &frag, nullptr);
    glCompileShader(frag_shader);
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);
    return program;
}

struct Ray
{
    vec3 pos;
    vec3 dir;
};

// does not handle a parallel case

float intersect_plane(Ray ray, vec3 normal, vec3 pos)
{
    float t = dot(normal, pos - ray.pos) / dot(normal, ray.dir);
    return t;
}

// sphere triangle intersection
struct STI
{
    vec3 point;
    vec3 normal;
    bool valid;
};

STI intersect_triangle(float radius, vec3 pos_start, vec3 pos_end, Vertex* verts)
{
    vec3 coords[3] = {verts[0].pos, verts[1].pos, verts[2].pos};
    float min_t = FLT_MAX;
    STI sti;
    sti.valid = false;

    // face
    do
    {
        vec3 edge01 = coords[1] - coords[0];
        vec3 edge12 = coords[2] - coords[1];
        vec3 edge20 = coords[0] - coords[2];
        vec3 normal = cross(edge01, edge12);
        float area = length(normal);
        normal = (1 / area) * normal; // normalize
        Ray ray = {pos_start, pos_end - pos_start};
        vec3 plane_pos = coords[0] + (radius * normal);
        float t = intersect_plane(ray, normal, plane_pos);

        if(t < 0 || t > min_t || isnan(t))
            break;
        vec3 p = ray.pos + t*ray.dir;
        float b0 = dot(normal, cross(edge12, p - coords[1])) / area;
        float b1 = dot(normal, cross(edge20, p - coords[2])) / area;
        float b2 = dot(normal, cross(edge01, p - coords[0])) / area;

        if(b0 < 0 || b1 < 0 || b2 < 0)
            break;
        min_t = t;
        sti.normal = normal;
    }
    while(0);

    // edges

    for(int i = 0; i < 3; ++i)
    {
        vec3 edge_v1 = coords[i];
        vec3 edge_v2 = coords[(i+1)%3];
        vec3 A = edge_v2 - edge_v1;
        vec3 V = pos_end - pos_start;
        vec3 S0 = pos_start - edge_v1;
        float a = dot(V,V) - ( (dot(V,A) * dot(V,A)) / dot(A,A) );
        float b = ( 2 * dot(S0,V) ) - ( 2 * dot(S0,A) * dot(V,A) / dot(A,A) );
        float c = dot(S0,S0) - (radius * radius) - ( dot(S0,A) * dot(S0,A) / dot(A,A) );
        float dis = b*b - 4*a*c;

        if(dis < 0)
            continue;

        float t = (-b - sqrtf(dis)) / (2*a);

        if(t < 0 || t > min_t || isnan(t))
            continue;

        float L = dot(S0 + t*V, A) / length(A);

        if(L < 0 || L > length(A))
            continue;
        min_t = t;
        vec3 tangent = L * normalize(A);
        sti.normal = normalize(S0 + t*V - tangent);
    }

    // vertices

    for(vec3 vpos: coords)
    {
        vec3 V = pos_end - pos_start;
        vec3 S0 = pos_start - vpos;
        float a = dot(V,V);
        float b = 2 * dot(S0,V);
        float c = dot(S0,S0) - radius*radius;
        float dis = b*b - 4*a*c;

        if(dis < 0)
            continue;

        float t = (-b - sqrtf(dis)) / (2*a);

        if(t < 0 || t > min_t || isnan(t))
            continue;
        min_t = t;
        sti.normal = normalize(S0 + t*V);
    }

    if(min_t <= 1)
    {
        sti.point = pos_start + min_t * (pos_end - pos_start);
        sti.valid = true;
    }
    return sti;
}

vec3 get_nearest_triangle_point(vec3 pos, Vertex* verts)
{
    vec3 coords[3] = {verts[0].pos, verts[1].pos, verts[2].pos};
    float min_dist = FLT_MAX;
    vec3 nearest;
    {
        vec3 edge01 = coords[1] - coords[0];
        vec3 edge12 = coords[2] - coords[1];
        vec3 edge20 = coords[0] - coords[2];
        vec3 normal = cross(edge01, edge12);
        float area = length(normal);
        normal = (1 / area) * normal; // normalize
        Ray ray = {pos, normal};
        float t = intersect_plane(ray, normal, coords[0]);
        vec3 p = ray.pos + t*ray.dir;
        float b0 = dot(normal, cross(edge12, p - coords[1])) / area;
        float b1 = dot(normal, cross(edge20, p - coords[2])) / area;
        float b2 = dot(normal, cross(edge01, p - coords[0])) / area;

        if(b0 >= 0 && b1 >= 0 && b2 >= 0)
        {
            min_dist = length(p - pos);
            nearest = p;
        }
    }

    for(int i = 0; i < 3; ++i)
    {
        vec3 edge_v1 = coords[i];
        vec3 edge_v2 = coords[(i+1)%3];
        vec3 A = edge_v2 - edge_v1;
        float L = dot(pos - edge_v1, A) / length(A);

        if(L < 0 || L > length(A))
            continue;

        vec3 p = edge_v1 + L * normalize(A);
        float d = length(p - pos);

        if(d > min_dist)
            continue;
        min_dist = d;
        nearest = p;
    }

    for(vec3 vpos: coords)
    {
        float d = length(vpos - pos);

        if(d > min_dist)
            continue;
        min_dist = d;
        nearest = vpos;
    }
    return nearest;
}

#define PLANE_OFFSET 0.01

vec3 get_offset_pos(float radius, vec3 pos, Mesh& level)
{
    for(;;)
    {
        bool done = true;

        for(int base = 0; base < level.vertex_count; base += 3)
        {
            vec3 p = get_nearest_triangle_point(pos, level.vertices + base);

            if(length(p - pos) + 0.001 >= radius + PLANE_OFFSET)
                continue;

            done = false;
            vec3 normal = normalize(pos - p);
            pos = p + (radius + PLANE_OFFSET) * normal;
        }

        if(done)
            return pos;
    }
}

STI intersect_level(float radius, vec3 pos_start, vec3 pos_end, Mesh& level)
{
    STI sti;
    sti.valid = false;

    for(int base = 0; base < level.vertex_count; base += 3)
    {
        STI sti2 = intersect_triangle(radius, pos_start, pos_end, level.vertices + base);

        if(!sti2.valid)
            continue;
        if(!sti.valid || (length(sti2.point - pos_start) < length(sti.point - pos_start)) )
            sti = sti2;
    }
    return sti;
}

vec3 slide(float radius, vec3 pos_start, vec3 pos_end, Mesh& level)
{
    STI sti = intersect_level(radius, pos_start, pos_end, level);

    if(!sti.valid)
        return get_offset_pos(radius, pos_end, level);

    vec3 Q = get_offset_pos(radius, sti.point, level);
    vec3 P3 = pos_end + (dot(sti.normal, Q - pos_end)) * sti.normal;
    STI sti2 = intersect_level(radius, Q, P3, level);

    if(!sti2.valid)
        return get_offset_pos(radius, P3, level);

    vec3 dir = normalize(cross(sti.normal, sti2.normal));
    P3 = Q + dot(dir, P3 - Q) * dir;
    STI sti3 = intersect_level(radius, Q, P3, level);

    if(!sti3.valid)
        return get_offset_pos(radius, P3, level);

    return get_offset_pos(radius, sti3.point, level);
}

// returns t ray paremter, t is negative on a miss

float intersect_triangle(Ray ray, Vertex* verts)
{
    vec3 coords[3] = {verts[0].pos, verts[1].pos, verts[2].pos};
    vec3 edge01 = coords[1] - coords[0];
    vec3 edge12 = coords[2] - coords[1];
    vec3 edge20 = coords[0] - coords[2];
    vec3 normal = cross(edge01, edge12);
    float area = length(normal);
    normal = (1 / area) * normal; // normalize
    float t = intersect_plane(ray, normal, coords[0]);

    if(t < 0)
        return t;
    vec3 p = ray.pos + t*ray.dir;
    float b0 = dot(normal, cross(edge12, p - coords[1])) / area;
    float b1 = dot(normal, cross(edge20, p - coords[2])) / area;
    float b2 = dot(normal, cross(edge01, p - coords[0])) / area;

    if(b0 < 0 || b1 < 0 || b2 < 0)
        return -1;
    return t;
}

float intersect_level(Ray ray, Mesh& level)
{
    float min_t = -1;

    for(int base = 0; base < level.vertex_count; base += 3)
    {
        float t = intersect_triangle(ray, level.vertices + base);

        if(t >= 0 && (t < min_t || min_t < 0))
            min_t = t;
    }
    return min_t;
}

// point on a circle

struct CirPoint
{
    vec3 b1;
    vec3 b2;
    vec3 center;
    float R;

    vec3 get(float t)
    {
        return (R*cosf(t)*b1) + (R*sinf(t)*b2) + center;
    }
};

float intersect_plane(CirPoint cirp, vec3 plane_normal, vec3 plane_point)
{
    float a = cirp.R * dot(cirp.b1, plane_normal);
    float b = cirp.R * dot(cirp.b2, plane_normal);

    if(b < 0)
        return -pi/2;
    float c = dot(plane_normal, plane_point - cirp.center);
    float inv = sqrtf(a*a + b*b);
    float arg1 = c / inv;

    if(arg1 < -1 || arg1 > 1)
        return -pi/2;
    float arg2 = a / inv;

    if(arg2 < -1 || arg2 > 1)
        return -pi/2;
    return asinf(arg1) - asinf(arg2);
}

float intersect_triangle(CirPoint cirp, Vertex* verts)
{
    vec3 coords[3] = {verts[0].pos, verts[1].pos, verts[2].pos};
    vec3 edge01 = coords[1] - coords[0];
    vec3 edge12 = coords[2] - coords[1];
    vec3 edge20 = coords[0] - coords[2];
    vec3 normal = cross(edge01, edge12);
    float area = length(normal);
    normal = (1 / area) * normal; // normalize
    float t = intersect_plane(cirp, normal, coords[0]);
    vec3 p = cirp.get(t);
    float b0 = dot(normal, cross(edge12, p - coords[1])) / area;
    float b1 = dot(normal, cross(edge20, p - coords[2])) / area;
    float b2 = dot(normal, cross(edge01, p - coords[0])) / area;

    if(b0 < 0 || b1 < 0 || b2 < 0)
        return -pi/2;
    return t;
}

// returns a maximal angle lower than 0

float intersect_level(CirPoint cirp, Mesh& level)
{
    float max_t = -pi/2;

    for(int base = 0; base < level.vertex_count; base += 3)
    {
        float t = intersect_triangle(cirp, level.vertices + base);

        if(t < 0)
            max_t = max(max_t, t);
    }
    return max_t;
}

#define DIST_INTERP_TIME 0.3

struct Controller
{
    vec3 pos;
    vec3 orbit_offset;
    vec3 forward;
    vec3 vel;
    float radius;
    float max_dist;
    float min_dist;
    float target_dist;
    float current_dist;
    float dist_vel;
    float pitch;
    float yaw;
    vec2 cursor_pos;
    vec2 win_size;
    bool lmb_down;
    bool rmb_down;
    bool mmb_down;
    bool w_down;
    bool s_down;
    bool a_down;
    bool d_down;
    bool q_down;
    bool e_down;
    bool jump_action;
    bool change_mode_action;
    vec3 eye_pos;
    mat4 view;
    mat4 proj;
};

void ctrl_init(Controller& ctrl, float win_width, float win_height, vec3 orbit_offset)
{
    ctrl.pos = vec3{50,100,0};
    ctrl.orbit_offset = orbit_offset;
    ctrl.forward = vec3{0,0,-1};
    ctrl.vel = {};
    ctrl.radius = 1.2;
    ctrl.max_dist = 150;
    ctrl.min_dist = ctrl.radius / 2;
    ctrl.target_dist = 35;
    ctrl.current_dist = ctrl.target_dist;
    ctrl.dist_vel = {};
    ctrl.pitch = to_radians(35);
    ctrl.yaw = 0;
    ctrl.win_size = {win_width, win_height};
    ctrl.lmb_down = false;
    ctrl.rmb_down = false;
    ctrl.mmb_down = false;
    ctrl.w_down = false;
    ctrl.s_down = false;
    ctrl.a_down = false;
    ctrl.d_down = false;
    ctrl.q_down = false;
    ctrl.e_down = false;
    ctrl.jump_action = false;
    ctrl.change_mode_action = false;
}

void ctrl_process_event(Controller& ctrl, SDL_Event& e, SDL_Window* window)
{
    switch(e.type)
    {
    case SDL_WINDOWEVENT:
    {
        if(e.window.event != SDL_WINDOWEVENT_SIZE_CHANGED)
            break;
        ctrl.win_size.x = e.window.data1;
        ctrl.win_size.y = e.window.data2;
        break;
    }
    case SDL_MOUSEBUTTONUP:
    case SDL_MOUSEBUTTONDOWN:
    {
        bool down = e.type == SDL_MOUSEBUTTONDOWN;

        switch(e.button.button)
        {
        case SDL_BUTTON_LEFT:
            ctrl.lmb_down = down;
            break;
        case SDL_BUTTON_RIGHT:
            ctrl.rmb_down = down;
            break;
        case SDL_BUTTON_MIDDLE:
            ctrl.mmb_down = down;
            break;
        }
        bool all_up = !ctrl.lmb_down && !ctrl.rmb_down && !ctrl.mmb_down;

        if(!all_up && SDL_GetRelativeMouseMode() == SDL_FALSE)
        {
            // fix for SDL changing cursor position in a relative mode
            ctrl.cursor_pos = vec2{(float)e.button.x, (float)e.button.y};
            SDL_SetRelativeMouseMode(SDL_TRUE);
        }
        else if(all_up && SDL_GetRelativeMouseMode() == SDL_TRUE)
        {
            SDL_SetRelativeMouseMode(SDL_FALSE);
            SDL_WarpMouseInWindow(window, ctrl.cursor_pos.x, ctrl.cursor_pos.y);
        }
        break;
    }
    case SDL_MOUSEWHEEL:
    {
        float base = 1.4;
        float scale = e.wheel.y < 0 ? powf(base, -e.wheel.y) : (1 / powf(base, e.wheel.y));
        ctrl.target_dist = max(ctrl.min_dist, min(ctrl.max_dist, ctrl.target_dist * scale));
        ctrl.dist_vel = (ctrl.target_dist - ctrl.current_dist) / DIST_INTERP_TIME;

        if(ctrl.target_dist == ctrl.min_dist)
            ctrl.change_mode_action = true;
        break;
    }
    case SDL_MOUSEMOTION:
    {
        if(!ctrl.lmb_down && !ctrl.rmb_down && !ctrl.mmb_down)
            break;
        float dx = 2*pi * (float)e.motion.xrel / ctrl.win_size.x;
        float dy = 2*pi * (float)e.motion.yrel / ctrl.win_size.y;
        ctrl.pitch += dy;
        ctrl.pitch = min(ctrl.pitch, to_radians(80));
        ctrl.pitch = max(ctrl.pitch, to_radians(-80));
        ctrl.yaw -= dx;
        break;
    }
    case SDL_KEYDOWN:
    case SDL_KEYUP:
    {
        bool down = e.type == SDL_KEYDOWN;

        switch(e.key.keysym.sym)
        {
        case SDLK_w:
            ctrl.w_down = down;
            break;
        case SDLK_s:
            ctrl.s_down = down;
            break;
        case SDLK_a:
            ctrl.a_down = down;
            break;
        case SDLK_d:
            ctrl.d_down = down;
            break;
        case SDLK_q:
            ctrl.q_down = down;
            break;
        case SDLK_e:
            ctrl.e_down = down;
            break;
        case SDLK_SPACE:
            if(down)
                ctrl.jump_action = true;
            break;
        }
        break;
    }
    }
}

#define CIRP_EPS (pi/60)
#define RAY_EPS 0.3

void ctrl_resolve_events(Controller& ctrl, float dt, Mesh& level, bool& ground)
{
    bool turn_mode = !ctrl.rmb_down && !ctrl.mmb_down;
    float turn_angle = 0;

    if(turn_mode)
    {
        float turn_dir = 0;

        if(ctrl.a_down)
            turn_dir += 1;
        if(ctrl.d_down)
            turn_dir += -1;

        turn_angle = turn_dir * (2 * pi / 2.5) * dt;
        ctrl.forward = rotate_y(turn_angle) * ctrl.forward;

        if(ctrl.lmb_down)
            ctrl.yaw -= turn_angle;
    }
    else
    {
        ctrl.forward = rotate_y(ctrl.yaw) * ctrl.forward;
        ctrl.yaw = 0;
    }

    // collider update

    vec3 move_dir = {};
    assert(ctrl.forward.y == 0);
    vec3 move_right = cross(ctrl.forward, vec3{0,1,0});

    if(ctrl.w_down || ctrl.mmb_down || (ctrl.lmb_down && ctrl.rmb_down))
        move_dir = move_dir + ctrl.forward;
    if(ctrl.s_down)
        move_dir = move_dir - ctrl.forward;
    if( ctrl.q_down || (ctrl.a_down && !turn_mode) )
        move_dir = move_dir - move_right;
    if( ctrl.e_down || (ctrl.d_down && !turn_mode) )
        move_dir = move_dir + move_right;

    if(dot(move_dir, move_dir))
        move_dir = normalize(move_dir);

    vec3 vel = 20 * move_dir;

    if(dot(move_dir, ctrl.forward) + 0.01 < 0)
        vel = 0.5 * vel;

    // slide down on steep slopes, steeper than the given angle
    float max_offset_y = PLANE_OFFSET / cosf(to_radians(60));
    STI sti = intersect_level(ctrl.radius, ctrl.pos, ctrl.pos + vec3{0,-max_offset_y,0}, level);
    ground = sti.valid;

    if(!sti.valid)
    {
        if(!ctrl.vel.x && !ctrl.vel.z && length(vel))
            ctrl.vel = ctrl.vel + 0.5 * vel;

        vec3 acc = {0, -9.8 * 20, 0};
        vec3 init_pos = ctrl.pos;
        vec3 new_pos = init_pos + (dt * ctrl.vel) + (0.5 * dt * dt * acc);
        ctrl.pos = slide(ctrl.radius, init_pos, new_pos, level);

        if(ctrl.vel.y > 0 && (ctrl.pos.y - init_pos.y < new_pos.y - init_pos.y))
            ctrl.vel.y = 0;

        ctrl.vel = ctrl.vel + dt * acc;
    }
    else
    {
        if(ctrl.jump_action)
            vel = vel + 80 * vec3{0,1,0};

        ctrl.pos = slide(ctrl.radius, ctrl.pos, ctrl.pos + dt * vel, level);

        // snap mechanic
        if(!ctrl.jump_action && length(vel))
        {
            STI sti = intersect_level(ctrl.radius, ctrl.pos, ctrl.pos + vec3{0,-0.5f*ctrl.radius,0}, level);

            if(sti.valid)
                ctrl.pos = get_offset_pos(ctrl.radius, sti.point, level);
        }
        ctrl.vel = vel;
    }

    // camera update

    if(!ctrl.lmb_down && (length(move_dir) || turn_angle))
    {
        float angle = min(2*pi/1.5 * dt, fabs(ctrl.yaw));

        if(ctrl.yaw > 0)
            angle *= -1;
        ctrl.yaw += angle;
    }

    vec3 eye_dir_xz = rotate_y(ctrl.yaw) * -ctrl.forward;
    vec3 eye_right = cross(vec3{0,1,0}, eye_dir_xz);
    vec3 eye_dir;
    float hit_dist = FLT_MAX;
    vec3 orbit_center = ctrl.pos + ctrl.orbit_offset;

    if(ctrl.pitch < 0)
    {
        CirPoint cirp;
        cirp.b1 = eye_dir_xz;
        cirp.b2 = vec3{0,1,0};
        cirp.center = orbit_center;
        cirp.R = ctrl.current_dist;
        float t_cirp = intersect_level(cirp, level);
        bool eye_hit = t_cirp + CIRP_EPS >= ctrl.pitch;
        vec3 eye_pos = eye_hit ? cirp.get(t_cirp + CIRP_EPS) : cirp.get(ctrl.pitch);

        if(eye_hit)
            hit_dist = length(eye_pos - orbit_center);
        eye_dir = normalize(eye_pos - orbit_center);
        float t_ray = intersect_level(Ray{orbit_center, eye_dir}, level);

        if(t_ray > 0 && t_ray - RAY_EPS < hit_dist)
            hit_dist = max(ctrl.min_dist, t_ray - RAY_EPS);
    }
    else
    {
        eye_dir = rotate_axis(eye_right, -ctrl.pitch) * eye_dir_xz;
        float t = intersect_level(Ray{orbit_center, eye_dir}, level);

        if(t > 0)
            hit_dist = max(ctrl.min_dist, t - RAY_EPS);
    }

    if(ctrl.current_dist > hit_dist)
    {
        ctrl.current_dist = hit_dist;
        ctrl.dist_vel = (ctrl.target_dist - ctrl.current_dist) / DIST_INTERP_TIME;
    }

    float lim_dist = min(hit_dist, ctrl.target_dist);
    float new_dist = ctrl.current_dist + ctrl.dist_vel * dt;

    if(ctrl.dist_vel > 0)
        ctrl.current_dist = min(lim_dist, new_dist);
    else
        ctrl.current_dist = max(lim_dist, new_dist);

    ctrl.eye_pos = orbit_center + ctrl.current_dist * eye_dir;
    vec3 view_dir = rotate_axis(eye_right, -ctrl.pitch) * eye_dir_xz;
    ctrl.view = lookat(ctrl.eye_pos, -view_dir);
    ctrl.proj = perspective(to_radians(90), ctrl.win_size.x / ctrl.win_size.y, 0.1, 1000);
    ctrl.jump_action = false;

    if(ctrl.change_mode_action)
    {
        ctrl.lmb_down = false;
        ctrl.rmb_down = false;
        ctrl.mmb_down = false;
        ctrl.w_down = false;
        ctrl.s_down = false;
        ctrl.a_down = false;
        ctrl.d_down = false;
        ctrl.q_down = false;
        ctrl.e_down = false;
    }
}

mat4 gl_from_blender()
{
    return rotate_x4(-pi/2);
}

void rot_diff_y(vec3 lhs, vec3 rhs, float& sign, float& abs_angle)
{
    float angle = atan2f(lhs.x, lhs.z) - atan2f(rhs.x, rhs.z);
    sign = angle < 0 ? -1 : 1;
    abs_angle = sign * angle;

    if(abs_angle > pi)
    {
        abs_angle = 2*pi - abs_angle;
        sign *= -1;
    }
}

struct AnimCtrlInput
{
    vec3 pos;
    vec3 forward;
    vec3 vel;
    bool rmb_down;
    bool ground;
    float dt;
};

#define BLEND_DUR 0.2

void update_anim_controller(AnimCtrl& actrl, AnimCtrlInput in)
{
    // update model_dir

    assert(in.forward.y == 0);
    assert(actrl.model_dir.y == 0);
    vec3 vel_xz = {in.vel.x, 0, in.vel.z};
    float dt = in.dt;
    float ang_vel = (actrl.state == AS_LAND || actrl.state == AS_IDLE) ? 2*pi/2 : 2*pi/0.5; // slower turn on landing and idle
    float max_angle_disp = ang_vel * dt;
    bool jump_action = false;
    bool run_back = false;

    if(in.ground)
    {
        actrl.jump_time = dt;
        jump_action = in.vel.y; // in.ground is true on a frame when character jumps

        if(length(vel_xz) == 0)
        {
            actrl.jump_angle = 0;
            float abs_angle, sign;
            rot_diff_y(in.forward, actrl.model_dir, sign, abs_angle);

            if(in.rmb_down)
            {
                if(abs_angle > pi/2)
                    actrl.model_dir = rotate_y(-sign * pi/2) * in.forward;
            }
            else
            {
                float angle = sign * min(max_angle_disp, abs_angle);
                actrl.model_dir = rotate_y(angle) * actrl.model_dir;
            }
        }
        else
        {
            vec3 target_dir = normalize(vel_xz);

            if(dot(target_dir, in.forward) + 0.1 < 0)
            {
                target_dir = -1 * target_dir;
                run_back = true;
            }

            {
                float sign, abs_angle;
                rot_diff_y(in.forward, target_dir, sign, abs_angle);
                float angle = sign * abs_angle;
                actrl.jump_angle = -angle;
                // this is needed for the next part to work when angles are close to pi/2
                target_dir = rotate_y(angle/100) * target_dir;
            }
            float sign, sign2, abs_angle, abs_angle2;
            rot_diff_y(target_dir, actrl.model_dir, sign, abs_angle);
            rot_diff_y(in.forward, actrl.model_dir, sign2, abs_angle2);

            if(abs_angle2 >= pi/2 && sign2 != sign)
            {
                sign *= -1;
                abs_angle = 2*pi - abs_angle;
            }
            float angle = sign * min(abs_angle, max_angle_disp);
            actrl.model_dir = rotate_y(angle) * actrl.model_dir;
        }
    }
    else
    {
        actrl.jump_time += dt;
        vec3 target_dir;

        if(length(actrl.vel_xz) == 0 && length(vel_xz)) // update jump angle
        {
            target_dir = normalize(vel_xz);

            if(dot(target_dir, in.forward) + 0.1 < 0)
                target_dir = -1 * target_dir;

            float sign, abs_angle;
            rot_diff_y(target_dir, in.forward, sign, abs_angle);
            actrl.jump_angle = sign * abs_angle;
        }
        else
            target_dir = rotate_y(actrl.jump_angle) * in.forward;

        float sign, abs_angle;
        rot_diff_y(target_dir, actrl.model_dir, sign, abs_angle);
        float angle = sign * min(abs_angle, max_angle_disp);
        actrl.model_dir = rotate_y(angle) * actrl.model_dir;
    }

    actrl.vel_xz = vel_xz;

    // update procedural bone rotations
    {
        int spine1_id = 0;
        int neck_id = 0;

        for(int i = 1; i < actrl.mesh->bone_count; ++i)
        {
            if(strcmp(actrl.mesh->bone_names[i], "spine1") == 0)
            {
                spine1_id = i;

                if(neck_id)
                    break;
            }
            else if(strcmp(actrl.mesh->bone_names[i], "neck") == 0)
            {
                neck_id = i;

                if(spine1_id)
                    break;
            }
        }
        assert(spine1_id);
        assert(neck_id);
        float sign, abs_angle;
        rot_diff_y(in.forward, actrl.model_dir, sign, abs_angle);
        float angle = sign * abs_angle;
        actrl.add_bone_rots[spine1_id] = rotate_y4(angle * 0.5);
        actrl.add_bone_rots[neck_id] = rotate_y4(angle * 0.25);
    }

    // update action state

    actrl.action_time += dt;
    actrl.blend_time += dt;
    ActionState next = actrl.state;
    bool reset_time = true;

    switch(actrl.state)
    {
    case AS_IDLE:
    case AS_RUN:
    case AS_RUN_BACK:
    case AS_LAND:
        if(jump_action)
            next = AS_JUMP;
        else if(actrl.jump_time > 0.5)
            next = AS_FALL;
        else if(length(vel_xz))
            next = run_back ? AS_RUN_BACK : AS_RUN;
        else if(actrl.state == AS_LAND)
        {
            if(actrl.action_time > actrl.actions[actrl.state]->duration)
            {
                next = AS_IDLE;
                reset_time = false;
            }
        }
        else
            next = AS_IDLE;
        break;
    case AS_JUMP:
    case AS_FALL:
        if(jump_action)
            next = AS_JUMP;
        else if(in.ground && length(vel_xz))
            next = run_back ? AS_RUN_BACK : AS_RUN;
        else if(in.ground)
            next = AS_LAND;
        else if(actrl.state == AS_JUMP && (actrl.action_time > actrl.actions[actrl.state]->duration))
        {
            next = AS_FALL;
            reset_time = false;
        }
        break;
    default:
        assert(false);
    }

    if(next != actrl.state)
    {
        if(reset_time)
        {
            actrl.action_time = 0;
            actrl.blend_time = 0;
        }
        else
        {
            actrl.action_time -= actrl.actions[actrl.state]->duration;
            assert(actrl.blend_time >= BLEND_DUR);
        }

        if(actrl.blend_time < BLEND_DUR)
            memcpy(actrl.skinning_mats_prev, actrl.skinning_mats, actrl.mesh->bone_count * sizeof(mat4));
        actrl.state = next;
    }

    // update shader matrices

    float sign, abs_angle;
    rot_diff_y(actrl.model_dir, vec3{0,0,1}, sign, abs_angle);
    actrl.model_tf = translate(in.pos) * rotate_y4(sign * abs_angle) * actrl.adjust_tf;

    UpdateBoneInput ubin;
    ubin.mesh = actrl.mesh;
    ubin.action = actrl.actions[actrl.state];
    ubin.skinning_mats = actrl.skinning_mats;
    ubin.cp_model_f_bone = actrl.cp_model_f_bone;
    ubin.add_bone_rots = actrl.add_bone_rots;
    ubin.action_time = &actrl.action_time;

    update_bone_data(ubin);

    if(actrl.blend_time < BLEND_DUR)
    {
        float t = actrl.blend_time / BLEND_DUR;

        for(int i = 0; i < actrl.mesh->bone_count; ++i)
            actrl.skinning_mats[i] = ((1-t) * actrl.skinning_mats_prev[i]) + (t * actrl.skinning_mats[i]);
    }
}

enum FpsState
{
    FPS_IDLE,
    FPS_AIM,
    FPS_WALK,
    FPS_START_AIM,
    FPS_END_AIM,
    FPS_SHOOT,
    FPS_SHOOT_AIM,
    FPS_RELOAD,
    FPS_COUNT,
};

struct FpsCtrl
{
    Action* actions[FPS_COUNT];
    mat4 proj;
    mat4 view;
    mat4 model_tf;
    mat4 adjust_tf;
    vec3 eye_pos;
    vec3 forward_xz;
    vec3 pos;
    vec3 vel;
    vec2 win_size;
    mat4* skinning_mats;
    mat4* skinning_mats_prev;
    mat4* cp_model_f_bone;
    Mesh2* mesh;
    FpsState state;
    float blend_time;
    float jump_time;
    float action_time;
    float radius;
    float pitch;
    float yaw;
    float shoot_action_pitch;
    float shoot_action_yaw;
    bool w_down;
    bool s_down;
    bool a_down;
    bool d_down;
    bool jump_action;
    bool reload_action;
    bool shoot_action;
    bool aim_action;
    bool change_mode_action;
};

// much of the relevant state is set during transition from a third person view

void fps_init(FpsCtrl& fps, float radius)
{
    fps = {};
    int size = MAX_BONES * sizeof(mat4);
    fps.skinning_mats = (mat4*)malloc(size);
    fps.skinning_mats_prev = (mat4*)malloc(size);
    fps.cp_model_f_bone = (mat4*)malloc(size);
    fps.state = FPS_IDLE;
    fps.blend_time = BLEND_DUR;
    fps.radius = radius;
    fps.w_down = false;
    fps.s_down = false;
    fps.a_down = false;
    fps.d_down = false;
    fps.jump_action = false;
    fps.reload_action = false;
    fps.shoot_action = false;
    fps.aim_action = false;
    fps.change_mode_action = false;
}

void fps_process_event(FpsCtrl& fps, SDL_Event& e)
{
    switch(e.type)
    {
    case SDL_WINDOWEVENT:
    {
        if(e.window.event != SDL_WINDOWEVENT_SIZE_CHANGED)
            break;
        fps.win_size.x = e.window.data1;
        fps.win_size.y = e.window.data2;
        break;
    }
    case SDL_MOUSEBUTTONDOWN:
    {
        if(e.button.button == SDL_BUTTON_LEFT)
        {
            fps.shoot_action = true;
            fps.shoot_action_pitch = fps.pitch;
            fps.shoot_action_yaw = fps.yaw;
        }
        else if(e.button.button == SDL_BUTTON_RIGHT)
            fps.aim_action = true;
        break;
    }
    case SDL_MOUSEWHEEL:
    {
        if(e.wheel.y < 0)
            fps.change_mode_action = true;
        break;
    }
    case SDL_MOUSEMOTION:
    {
        float dx = 1.5*pi * (float)e.motion.xrel / fps.win_size.x;
        float dy = 1.5*pi * (float)e.motion.yrel / fps.win_size.y;
        fps.yaw -= dx;
        fps.pitch -= dy;
        fps.pitch = min(fps.pitch, to_radians(85));
        fps.pitch = max(fps.pitch, to_radians(-85));
        break;
    }
    case SDL_KEYDOWN:
    case SDL_KEYUP:
    {
        bool down = e.type == SDL_KEYDOWN;

        switch(e.key.keysym.sym)
        {
        case SDLK_w:
            fps.w_down = down;
            break;
        case SDLK_s:
            fps.s_down = down;
            break;
        case SDLK_a:
            fps.a_down = down;
            break;
        case SDLK_d:
            fps.d_down = down;
            break;
        case SDLK_SPACE:
            if(down)
                fps.jump_action = true;
            break;
        case SDLK_r:
            if(down)
                fps.reload_action = true;
            break;
        }
        break;
    }
    }
}

void fps_resolve_events(FpsCtrl& fps, float dt, Mesh& level, bool& hit, vec3& hit_pos)
{
    // collider update

    vec3 move_fwd = rotate_y(fps.yaw) * vec3{0,0,1};
    vec3 move_dir = {};
    assert(move_fwd.y == 0);
    vec3 move_right = cross(move_fwd, vec3{0,1,0});

    if(fps.w_down)
        move_dir = move_dir + move_fwd;
    if(fps.s_down)
        move_dir = move_dir - move_fwd;
    if(fps.a_down)
        move_dir = move_dir - move_right;
    if(fps.d_down)
        move_dir = move_dir + move_right;

    if(dot(move_dir, move_dir))
        move_dir = normalize(move_dir);

    vec3 vel = 20 * move_dir;

    if(dot(move_dir, move_fwd) + 0.01 < 0)
        vel = 0.5 * vel;

    float max_offset_y = PLANE_OFFSET / cosf(to_radians(60));
    STI sti = intersect_level(fps.radius, fps.pos, fps.pos + vec3{0,-max_offset_y,0}, level);

    if(!sti.valid)
    {
        fps.jump_time += dt;

        if(!fps.vel.x && !fps.vel.z && length(vel))
            fps.vel = fps.vel + 0.5 * vel;

        vec3 acc = {0, -9.8 * 20, 0};
        vec3 init_pos = fps.pos;
        vec3 new_pos = init_pos + (dt * fps.vel) + (0.5 * dt * dt * acc);
        fps.pos = slide(fps.radius, init_pos, new_pos, level);

        if(fps.vel.y > 0 && (fps.pos.y - init_pos.y < new_pos.y - init_pos.y))
            fps.vel.y = 0;

        fps.vel = fps.vel + dt * acc;
    }
    else
    {
        fps.jump_time = dt;

        if(fps.jump_action)
            vel = vel + 80 * vec3{0,1,0};

        fps.pos = slide(fps.radius, fps.pos, fps.pos + dt * vel, level);

        // snap mechanic
        if(!fps.jump_action && length(vel))
        {
            STI sti = intersect_level(fps.radius, fps.pos, fps.pos + vec3{0,-0.5f*fps.radius,0}, level);

            if(sti.valid)
                fps.pos = get_offset_pos(fps.radius, sti.point, level);
        }
        fps.vel = vel;
    }

    // camera, model update
    // note: animation stuff is quite rough right now

    fps.action_time += dt;
    fps.blend_time += dt;
    FpsState next = fps.state;
    vec3 vel_xz = {fps.vel.x, 0, fps.vel.z};
    float dur = fps.actions[fps.state]->duration;
    bool idle_cond = fps.jump_time > 0.5 || !length(vel_xz);
    bool shot = false;
    // relevant only when next != state
    bool reset_time = true;
    bool enable_blend = true;

    switch(fps.state)
    {
    case FPS_IDLE:
    case FPS_WALK:
        if(fps.reload_action)
            next = FPS_RELOAD;
        else if(fps.aim_action)
            next = FPS_START_AIM;
        // todo: shooting while walking requires more advanced blending code to look good (e.g. partial blending)
        else if(fps.shoot_action && fps.state == FPS_IDLE)
            next = FPS_SHOOT;
        else
            next = idle_cond ? FPS_IDLE : FPS_WALK;
        break;

    case FPS_RELOAD:
        if(dur - fps.action_time < BLEND_DUR)
            next = idle_cond ? FPS_IDLE : FPS_WALK;
        break;

    case FPS_AIM:
        if(fps.shoot_action)
            next = FPS_SHOOT_AIM;
        else if(fps.aim_action)
            next = FPS_END_AIM;

        enable_blend = false;
        break;

    case FPS_START_AIM:
    case FPS_END_AIM:
        if(fps.aim_action)
        {
            next = fps.state == FPS_START_AIM ? FPS_END_AIM : FPS_START_AIM;
            fps.action_time = dur - (fps.action_time - dt);
            assert(fps.action_time > 0);
            reset_time = false;
            enable_blend = false;
        }
        else if(dur - fps.action_time < BLEND_DUR)
        {
            if(fps.state == FPS_START_AIM)
            {
                if(fps.shoot_action)
                    next = FPS_SHOOT_AIM;
                else
                    next = FPS_AIM;
            }
            else if(fps.reload_action)
                next = FPS_RELOAD;
            else if(fps.shoot_action)
                next = FPS_SHOOT;
            else
                next = idle_cond ? FPS_IDLE : FPS_WALK;
        }
        break;

    case FPS_SHOOT:
    case FPS_SHOOT_AIM:
        if(fps.shoot_action && fps.action_time > 0.2)
        {
            fps.action_time = 0;
            shot = true;
            fps.yaw += to_radians(-0.2);
            fps.pitch += to_radians(0.5);
        }
        else if(fps.aim_action)
        {
            next = fps.state == FPS_SHOOT ? FPS_START_AIM : FPS_END_AIM;
        }
        else if(fps.action_time > 0.2 && fps.state == FPS_SHOOT && fps.reload_action)
        {
            next = FPS_RELOAD;
        }
        else if(fps.action_time > dur)
        {
            // animations are synchonized
            fps.action_time = fps.action_time - dur;
            reset_time = false;
            enable_blend = false;

            if(fps.state == FPS_SHOOT)
                next = idle_cond ? FPS_IDLE : FPS_WALK;
            else
                next = FPS_AIM;
        }
        break;

    default:
        assert(false);
    }

    if(next != fps.state)
    {
        shot = next == FPS_SHOOT || next == FPS_SHOOT_AIM;

        if(shot)
            enable_blend = false;

        fps.state = next;

        if(reset_time)
            fps.action_time = 0;

        if(enable_blend)
        {
            fps.blend_time = 0;
            memcpy(fps.skinning_mats_prev, fps.skinning_mats, fps.mesh->bone_count * sizeof(mat4));
        }
    }

    UpdateBoneInput ubin;
    ubin.mesh = fps.mesh;
    ubin.action = fps.actions[fps.state];
    ubin.skinning_mats = fps.skinning_mats;
    ubin.cp_model_f_bone = fps.cp_model_f_bone;
    ubin.add_bone_rots = nullptr;
    ubin.action_time = &fps.action_time;

    update_bone_data(ubin);

    if(fps.blend_time < BLEND_DUR)
    {
        float t = fps.blend_time / BLEND_DUR;

        for(int i = 0; i < fps.mesh->bone_count; ++i)
            fps.skinning_mats[i] = ((1-t) * fps.skinning_mats_prev[i]) + (t * fps.skinning_mats[i]);
    }

    int locator_id = 0;

    for(int i = 1; i < fps.mesh->bone_count; ++i)
    {
        if(strcmp("camera_locator", fps.mesh->bone_names[i]) == 0)
        {
            locator_id = i;
            break;
        }
    }

    assert(locator_id);
    vec3 rotx_origin = to_vec3(fps.adjust_tf * fps.cp_model_f_bone[locator_id] * vec4{0,0,0,1});
    fps.model_tf = translate(fps.pos) * rotate_y4(fps.yaw) * translate(rotx_origin) * rotate_x4(-fps.pitch) * translate(-rotx_origin) * fps.adjust_tf;
    mat4 world_f_view = fps.model_tf * fps.cp_model_f_bone[locator_id];
    vec3 eye_dir = normalize(-world_f_view.col(2));
    fps.eye_pos = world_f_view.col(3);
    fps.view = lookat(fps.eye_pos, eye_dir);
    fps.proj = perspective(to_radians(80), fps.win_size.x / fps.win_size.y, 0.1, 1000);
    fps.forward_xz = rotate_y(fps.yaw) * vec3{0,0,1};
    fps.jump_action = false;
    fps.reload_action = false;
    fps.shoot_action = false;
    fps.aim_action = false;

    if(fps.change_mode_action)
    {
        fps.w_down = false;
        fps.s_down = false;
        fps.a_down = false;
        fps.d_down = false;
    }

    hit = false;

    if(shot)
    {
        Ray ray;
        ray.pos = fps.eye_pos;
        ray.dir = eye_dir;
        float t = intersect_level(ray, level);

        if(t >= 0)
        {
            hit = true;
            hit_pos = ray.pos + t * ray.dir;
        }
    }
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
        assert(false);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 100, 100, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
        assert(false);

    GLuint prog = create_program(_vert, _frag);
    GLuint skel_prog = create_program(src_vert_skel, _frag);

    Controller ctrl;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        ctrl_init(ctrl, width, height, vec3{0,10,0});
    }

    std::vector<Mesh> meshes;
    meshes.push_back(load_mesh("level.obj"));
    meshes.push_back(load_mesh("sphere.obj"));
    std::vector<Object> objects;
    objects.reserve(100);
    {
        // level
        Object obj;
        obj.mesh = &meshes[0];
        obj.pos = {};
        obj.rot = identity4();
        obj.scale = vec3{1,1,1};
        objects.push_back(obj);
        // character
        obj.mesh = &meshes[1];
        obj.scale = vec3{ctrl.radius,ctrl.radius,ctrl.radius};
        objects.push_back(obj);
        // forward indicator
        obj.scale = 0.5 * obj.scale;
        objects.push_back(obj);
        // camera orbit indicator
        obj.scale = 0.8 * obj.scale;
        objects.push_back(obj);
    }
    Mesh2 char_mesh;
    std::vector<Action> char_actions;
    load2("/home/mat/Downloads/blender-2.83.0-linux64/anim_third", char_mesh, char_actions);
    assert(char_mesh.vertex_count);
    assert(char_actions.size());
    AnimCtrl actrl;
    init(actrl);
    actrl.mesh = &char_mesh;
    actrl.adjust_tf = scale(vec3{8,8,8}) * gl_from_blender();
    actrl.model_dir = ctrl.forward;
    actrl.state = AS_IDLE;
    actrl.blend_time = BLEND_DUR;

    for(int i = 0; i < (int)char_actions.size(); ++i)
    {
        Action* action = char_actions.data() + i;
        ActionState state = AS_COUNT;

        if(strcmp(action->name, "idle") == 0)
            state = AS_IDLE;
        else if(strcmp(action->name, "run") == 0)
            state = AS_RUN;
        else if(strcmp(action->name, "run_back") == 0)
            state = AS_RUN_BACK;
        else if(strcmp(action->name, "jump") == 0)
            state = AS_JUMP;
        else if(strcmp(action->name, "fall") == 0)
            state = AS_FALL;
        else if(strcmp(action->name, "land") == 0)
            state = AS_LAND;

        if(state != AS_COUNT)
            actrl.actions[state] = action;
    }

    for(Action* action: actrl.actions)
        assert(action);

    FpsCtrl fps;
    fps_init(fps, ctrl.radius);
    fps.adjust_tf = actrl.adjust_tf;
    Mesh2 fps_mesh;
    std::vector<Action> fps_actions;
    load2("/home/mat/Downloads/blender-2.83.0-linux64/anim_fps", fps_mesh, fps_actions);
    assert(fps_mesh.vertex_count);
    assert(fps_actions.size());
    fps.mesh = &fps_mesh;

    for(int i = 0; i < (int)fps_actions.size(); ++i)
    {
        Action* action = fps_actions.data() + i;
        FpsState state = FPS_COUNT;

        if(strcmp(action->name, "idle") == 0)
            state = FPS_IDLE;
        else if(strcmp(action->name, "aim") == 0)
            state = FPS_AIM;
        else if(strcmp(action->name, "walk") == 0)
            state = FPS_WALK;
        else if(strcmp(action->name, "start_aim") == 0)
            state = FPS_START_AIM;
        else if(strcmp(action->name, "end_aim") == 0)
            state = FPS_END_AIM;
        else if(strcmp(action->name, "shoot") == 0)
            state = FPS_SHOOT;
        else if(strcmp(action->name, "shoot_aim") == 0)
            state = FPS_SHOOT_AIM;
        else if(strcmp(action->name, "reload") == 0)
            state = FPS_RELOAD;

        if(state != FPS_COUNT)
            fps.actions[state] = action;
    }

    for(Action* action: fps.actions)
        assert(action);

    Uint64 prev_counter = SDL_GetPerformanceCounter();
    bool quit = false;
    bool ctrl_mode = true;

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                quit = true;

            if(ctrl_mode)
                ctrl_process_event(ctrl, event, window);
            else
                fps_process_event(fps, event);
        }

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        dt = min(dt, 0.030); // debugging
        prev_counter = current_counter;

        mat4 view;
        mat4 proj;
        vec3 forward;
        vec3 eye_pos;
        vec3 char_pos;
        bool render_thirdp = true;

        if(ctrl_mode)
        {
            AnimCtrlInput anim_in;

            ctrl_resolve_events(ctrl, dt, meshes[0], anim_in.ground);

            anim_in.pos = ctrl.pos;
            anim_in.forward = ctrl.forward;
            anim_in.vel = ctrl.vel;
            anim_in.rmb_down = ctrl.rmb_down;
            anim_in.dt = dt;
            update_anim_controller(actrl, anim_in);

            view = ctrl.view;
            proj = ctrl.proj;
            forward = ctrl.forward;
            eye_pos = ctrl.eye_pos;
            char_pos = ctrl.pos;

            if(ctrl.change_mode_action)
            {
                ctrl.change_mode_action = false;
                ctrl_mode = false;
                // update fps controller
                SDL_SetRelativeMouseMode(SDL_TRUE);
                fps.pos = ctrl.pos;
                fps.vel = ctrl.vel;
                fps.win_size = ctrl.win_size;
                fps.state = FPS_IDLE;
                fps.blend_time = BLEND_DUR;
                fps.pitch = 0;
                float sign, abs_angle;
                rot_diff_y(ctrl.forward, vec3{0,0,1}, sign, abs_angle);
                fps.yaw = sign * abs_angle;
            }
        }
        else
        {
            bool hit;
            vec3 hit_pos;
            fps_resolve_events(fps, dt, meshes[0], hit, hit_pos);

            if(hit)
            {
                Object obj = objects.back();
                obj.pos = hit_pos;
                obj.scale = vec3{0.2,0.2,0.2};
                objects.push_back(obj);
            }

            view = fps.view;
            proj = fps.proj;
            forward = fps.forward_xz;
            eye_pos = fps.eye_pos;
            char_pos = fps.pos;
            render_thirdp = false;

            if(fps.change_mode_action)
            {
                fps.change_mode_action = false;
                ctrl_mode = true;
                // update controller
                SDL_SetRelativeMouseMode(SDL_FALSE);
                ctrl.pos = fps.pos;
                ctrl.vel = fps.vel;
                ctrl.win_size = fps.win_size;
                ctrl.forward = fps.forward_xz;
            }
        }

        objects[1].pos = char_pos;
        objects[2].pos = objects[1].pos + 2 * forward;
        objects[3].pos = ctrl.pos + ctrl.orbit_offset;
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        GLuint progs[] = {prog, skel_prog};

        for(GLuint prog: progs)
        {

            glUseProgram(prog);
            vec3 light_intensity = {1,1,1};
            vec3 light_dir = normalize(vec3{0.3,1,0});
            vec3 specular_color = {0.2, 0.2, 0.2};
            GLint proj_loc = glGetUniformLocation(prog, "proj");
            GLint view_loc = glGetUniformLocation(prog, "view");
            GLint light_int_loc = glGetUniformLocation(prog, "light_intensity");
            GLint light_dir_loc = glGetUniformLocation(prog, "light_dir");
            GLint ambient_int_loc = glGetUniformLocation(prog, "ambient_intensity");
            GLint eye_pos_loc = glGetUniformLocation(prog, "eye_pos");
            GLint specular_color_loc = glGetUniformLocation(prog, "specular_color");
            GLint specular_exp_loc = glGetUniformLocation(prog, "specular_exp");

            glUniformMatrix4fv(view_loc, 1, GL_TRUE, view.data);
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj.data);
            glUniform3fv(light_int_loc, 1, &light_intensity.x);
            glUniform3fv(light_dir_loc, 1, &light_dir.x);
            glUniform1f(ambient_int_loc, 0.01);
            glUniform3fv(eye_pos_loc, 1, &eye_pos.x);
            glUniform3fv(specular_color_loc, 1, &specular_color.x);
            glUniform1f(specular_exp_loc, 50);
        }

        glUseProgram(prog);

        for(int i = 0; i < (int)objects.size(); ++i)
        {
            Object& obj = objects[i];
            GLint diffuse_color_loc = glGetUniformLocation(prog, "diffuse_color");
            GLint model_loc = glGetUniformLocation(prog, "model");
            vec3 diffuse_color = vec3{0.3, 0.3, 0.3};

            if(i == 1)
                diffuse_color = vec3{1,0,0};
            else if(i == 2)
                diffuse_color = vec3{0,1,0};
            else if(i == 3)
            {
                if(!render_thirdp)
                    continue;
                diffuse_color = vec3{1,0,1};
                continue; // disabled for now
            }
            else if(i > 3)
                diffuse_color = vec3{1,0.3,1};

            mat4 model = translate(obj.pos) * obj.rot * scale(obj.scale);
            glUniform3fv(diffuse_color_loc, 1, &diffuse_color.x);
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model.data);
            glBindVertexArray(obj.mesh->vao);

            if(obj.scale.x * obj.scale.y * obj.scale.z < 0)
                glCullFace(GL_FRONT);
            else
                glCullFace(GL_BACK);

            glDrawArrays(GL_TRIANGLES, 0, obj.mesh->vertex_count);
        }

        // character model

        if(render_thirdp)
        {
            vec3 diffuse_color = {0,0.5,0};
            glUseProgram(skel_prog);
            glUniform3fv(glGetUniformLocation(skel_prog, "diffuse_color"), 1, &diffuse_color.x);
            glUniformMatrix4fv(glGetUniformLocation(skel_prog, "model"), 1, GL_TRUE, actrl.model_tf.data);
            assert(actrl.mesh->bone_count <= MAX_BONES);
            glUniformMatrix4fv(glGetUniformLocation(skel_prog, "skinning_matrices"), actrl.mesh->bone_count, GL_TRUE, actrl.skinning_mats[0].data);
            glBindVertexArray(actrl.mesh->vao);
            glDrawElements(GL_TRIANGLES, actrl.mesh->index_count, GL_UNSIGNED_INT, (void*)(uint64_t)actrl.mesh->ebo_offset); // suppress warning
        }
        else
        {
            vec3 diffuse_color = {1,1,1};
            glUseProgram(skel_prog);
            glUniform3fv(glGetUniformLocation(skel_prog, "diffuse_color"), 1, &diffuse_color.x);
            glUniformMatrix4fv(glGetUniformLocation(skel_prog, "model"), 1, GL_TRUE, fps.model_tf.data);
            assert(fps.mesh->bone_count <= MAX_BONES);
            glUniformMatrix4fv(glGetUniformLocation(skel_prog, "skinning_matrices"), fps.mesh->bone_count, GL_TRUE, fps.skinning_mats[0].data);
            glBindVertexArray(fps.mesh->vao);
            glDrawElements(GL_TRIANGLES, fps.mesh->index_count, GL_UNSIGNED_INT, (void*)(uint64_t)fps.mesh->ebo_offset); // suppress warning
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
