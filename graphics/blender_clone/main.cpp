#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "../main.hpp"

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

static const char* _vert_solid = R"(
#version 330
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
layout(location = 0) in vec3 pos;
void main()
{
    gl_Position = proj * view * model * vec4(pos, 1);
}
)";

static const char* _frag_solid = R"(
#version 330
uniform vec3 color;
out vec3 out_color;
void main()
{
    out_color = color;
    out_color = pow(out_color, vec3(1/2.2));
}
)";

struct Vertex
{
    vec3 pos;
    vec3 normal;
};

struct BBox
{
    vec3 min;
    vec3 max;
};

struct BVHPrim
{
    Vertex verts[3];
    BBox bbox;
    vec3 centroid;
};

struct BVHNode
{
    BVHNode* children[2];
    BBox bbox;
    int vert_count;
    int vert_offset;
    int split_axis;
};

struct Mesh
{
    Vertex* vertices;
    int vertex_count;
    GLuint vao;
    GLuint vbo;
    BVHNode* bvh_root;
};

struct Object
{
    Mesh* mesh;
    vec3 pos;
    mat4 rot;
    vec3 scale;
};

BBox compute_bbox(Vertex* verts, int vert_count)
{
    assert(vert_count);
    BBox bbox;
    bbox.min = verts[0].pos;
    bbox.max = verts[0].pos;

    for(int i = 1; i < vert_count; ++i)
    {
        vec3 p = verts[i].pos;

        for(int dim = 0; dim < 3; ++dim)
        {
            bbox.min[dim] = min(bbox.min[dim], p[dim]);
            bbox.max[dim] = max(bbox.max[dim], p[dim]);
        }
    }
    return bbox;
}

BBox bbox_union(BBox lhs, BBox rhs)
{
    BBox bbox;

    for(int dim = 0; dim < 3; ++dim)
    {
        bbox.min[dim] = min(lhs.min[dim], rhs.min[dim]);
        bbox.max[dim] = max(lhs.max[dim], rhs.max[dim]);
    }
    return bbox;
}

int _cmp_dim;

int cmp_prims(const void* _lhs, const void* _rhs)
{
    BVHPrim* lhs = (BVHPrim*)_lhs;
    BVHPrim* rhs = (BVHPrim*)_rhs;
    float l = lhs->centroid[_cmp_dim];
    float r = rhs->centroid[_cmp_dim];

    if(l < r)
        return -1;
    if(l == r)
        return 0;
    return 1;
}

BVHNode* build_BVH_rec(BVHPrim* prims, int prim_count, std::vector<Vertex>& ordered_verts)
{
    assert(prim_count);
    BVHNode* node = (BVHNode*)malloc(sizeof(BVHNode));
    node->vert_count = 0;
    node->bbox = prims[0].bbox;

    for(int i = 1; i < prim_count; ++i)
        node->bbox = bbox_union(node->bbox, prims[i].bbox);

    if(prim_count <= 4)
    {
        node->vert_count = prim_count * 3;
        node->vert_offset = ordered_verts.size();

        for(int i = 0; i < prim_count; ++i)
        {
            ordered_verts.push_back(prims[i].verts[0]);
            ordered_verts.push_back(prims[i].verts[1]);
            ordered_verts.push_back(prims[i].verts[2]);
        }
        return node;
    }

    BBox centroid_bbox = {prims[0].centroid, prims[0].centroid};

    for(int i = 1; i < prim_count; ++i)
        centroid_bbox = bbox_union(centroid_bbox, BBox{prims[1].centroid, prims[1].centroid});

    int dim = 0;
    float extent = 0;

    for(int i = 0; i < 3; ++i)
    {
        float d = centroid_bbox.max[i] - centroid_bbox.min[i];

        if(d > extent)
        {
            dim = i;
            extent = d;
        }
    }

    if(!extent)
    {
        node->vert_count = prim_count * 3;
        node->vert_offset = ordered_verts.size();

        for(int i = 0; i < prim_count; ++i)
        {
            ordered_verts.push_back(prims[i].verts[0]);
            ordered_verts.push_back(prims[i].verts[1]);
            ordered_verts.push_back(prims[i].verts[2]);
        }
        return node;
    }
    _cmp_dim = dim;
    qsort(prims, prim_count, sizeof(BVHPrim), cmp_prims);
    float midp = 0.5 * (centroid_bbox.max[dim] + centroid_bbox.min[dim]);
    int mid_id = 0;

    for(; mid_id < prim_count; ++mid_id)
    {
        if(prims[mid_id].centroid[dim] > midp)
            break;
    }
    // do a median splitting if a midpoint splitting fails; primitives are already sorted

    if(mid_id == 0 || mid_id == prim_count)
        mid_id = prim_count / 2;

    node->split_axis = dim;
    node->children[0] = build_BVH_rec(prims, mid_id, ordered_verts);
    node->children[1] = build_BVH_rec(prims + mid_id, prim_count - mid_id, ordered_verts);
    return node;
}

BVHNode* build_BVH(Vertex* verts, int vert_count)
{
    std::vector<BVHPrim> prims;
    prims.reserve(vert_count / 3);

    for(int base = 0; base < vert_count; base += 3)
    {
        BVHPrim prim;
        prim.verts[0] = verts[base+0];
        prim.verts[1] = verts[base+1];
        prim.verts[2] = verts[base+2];
        prim.bbox = compute_bbox(verts + base, 3);
        prim.centroid = 0.5 * prim.bbox.min + 0.5 * prim.bbox.max;
        prims.push_back(prim);
    }
    std::vector<Vertex> ordered_verts;
    ordered_verts.reserve(vert_count);
    BVHNode* root = build_BVH_rec(prims.data(), prims.size(), ordered_verts);
    memcpy(verts, ordered_verts.data(), vert_count * sizeof(Vertex));
    return root;
}

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

    mesh.bvh_root = build_BVH(mesh.vertices, mesh.vertex_count);
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

vec3 transform3(mat4 m, vec3 v)
{
    vec4 h = {v.x, v.y, v.z, 1};
    h = m * h;
    return {h.x, h.y, h.z};
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

struct Nav
{
    vec3 center;
    vec3 eye_pos;
    vec3 eye_x;
    mat4 view;
    mat4 proj;
    bool mmb_down;
    bool mmb_shift_down;
    bool shift_down;
    vec2 cursor_win;
    vec2 win_size;
    float top;
    float near;
    float far;
    bool ortho;
    bool aligned;
};

void rebuild_view_matrix(Nav& nav)
{
    vec3 x = nav.eye_x;
    vec3 z = normalize(nav.eye_pos - nav.center);
    vec3 y = cross(z, x);
    mat4 rot = identity4();
    rot.data[0] = x.x;
    rot.data[1] = x.y;
    rot.data[2] = x.z;
    rot.data[4] = y.x;
    rot.data[5] = y.y;
    rot.data[6] = y.z;
    rot.data[8] = z.x;
    rot.data[9] = z.y;
    rot.data[10] = z.z;
    nav.view = rot * translate(-nav.eye_pos);
}

void rebuild_proj_matrix(Nav& nav)
{
    float aspect = nav.win_size.x / nav.win_size.y;
    float top = !nav.ortho ? nav.top : (nav.top / nav.near) * length(nav.center - nav.eye_pos);
    float right = top * aspect;

    if(nav.ortho)
        nav.proj = orthographic(-right, right, -top, top, nav.near, nav.far);
    else
        nav.proj = frustum(-right, right, -top, top, nav.near, nav.far);
}

void nav_init(Nav& nav, vec3 eye_pos, float win_width, float win_height, float fovy, float near, float far)
{
    nav.center = {0,0,0};
    nav.eye_pos = eye_pos;
    vec3 forward = normalize(nav.center - eye_pos);
    nav.eye_x = normalize(cross(forward, vec3{0,1,0}));
    nav.mmb_down = false;
    nav.mmb_shift_down = false;
    nav.shift_down = false;
    nav.win_size.x = win_width;
    nav.win_size.y = win_height;
    nav.top = tanf(deg_to_rad(fovy) / 2.f) * near;
    nav.near = near;
    nav.far = far;
    nav.ortho = false;
    nav.aligned = false;
    rebuild_view_matrix(nav);
    rebuild_proj_matrix(nav);
}

Ray nav_get_cursor_ray(Nav& nav, vec2 cursor_win)
{
    float top = !nav.ortho ? nav.top : (nav.top / nav.near) * length(nav.center - nav.eye_pos);
    float right = top * (nav.win_size.x / nav.win_size.y);
    float x = (2*right/nav.win_size.x) * (cursor_win.x + 0.5) - right;
    float y = (-2*top/nav.win_size.y) * (cursor_win.y + 0.5) + top;
    float z = -nav.near;
    mat4 world_f_view = invert_coord_change(nav.view);
    Ray ray;

    if(nav.ortho)
    {
        ray.pos = transform3(world_f_view, vec3{x,y,z});
        ray.dir = normalize(nav.center - nav.eye_pos);
    }
    else
    {
        mat3 eye_basis = mat4_to_mat3(world_f_view);
        ray.pos = nav.eye_pos;
        ray.dir = normalize( eye_basis * vec3{x,y,z} );
    }
    return ray;
}

void nav_process_event(Nav& nav, SDL_Event& e)
{
    switch(e.type)
    {
    case SDL_MOUSEMOTION:
    {
        vec2 new_cursor_win = {(float)e.motion.x, (float)e.motion.y};

        if(nav.mmb_shift_down)
        {
            vec3 normal = normalize(nav.eye_pos - nav.center);
            vec2 cursors[2] = {nav.cursor_win, new_cursor_win};
            vec3 points[2];

            for(int i = 0; i < 2; ++i)
            {
                Ray ray = nav_get_cursor_ray(nav, cursors[i]);
                float t = intersect_plane(ray, normal, nav.center);
                assert(t > 0);
                points[i] = ray.pos + t*ray.dir;
            }
            vec3 d = points[0] - points[1];
            nav.eye_pos = nav.eye_pos + d;
            nav.center = nav.center + d;
            rebuild_view_matrix(nav);
        }
        else if(nav.mmb_down)
        {
            nav.aligned = false;
            float dx = 4*pi * -(new_cursor_win.x - nav.cursor_win.x) / nav.win_size.x;
            float dy = 4*pi * -(new_cursor_win.y - nav.cursor_win.y) / nav.win_size.y;
            mat4 rot = rotate_y(dx) * rotate_axis(nav.eye_x, dy);
            nav.eye_pos = transform3(translate(nav.center) * rot * translate(-nav.center), nav.eye_pos);
            nav.eye_x = transform3(rot, nav.eye_x);
            rebuild_view_matrix(nav);
        }
        nav.cursor_win = new_cursor_win;
        break;
    }
    case SDL_MOUSEWHEEL:
    {
        if(nav.mmb_down || nav.mmb_shift_down)
            break;
        vec3 diff = nav.eye_pos - nav.center;
        float scale = e.wheel.y < 0 ? powf(1.3, -e.wheel.y) : (1 / powf(1.3, e.wheel.y));
        nav.eye_pos = nav.center + (scale * diff);
        rebuild_view_matrix(nav);

        if(nav.ortho)
            rebuild_proj_matrix(nav);
        break;
    }
    case SDL_WINDOWEVENT:
    {
        if(e.window.event != SDL_WINDOWEVENT_SIZE_CHANGED)
            break;
        nav.win_size.x = e.window.data1;
        nav.win_size.y = e.window.data2;
        rebuild_proj_matrix(nav);
        break;
    }
    case SDL_MOUSEBUTTONDOWN:
    {
        if(e.button.button != SDL_BUTTON_MIDDLE)
            break;
        nav.mmb_down = !nav.shift_down;
        nav.mmb_shift_down = nav.shift_down;

        if(nav.mmb_down && nav.aligned)
        {
            nav.ortho = false;
            rebuild_proj_matrix(nav);
        }
        break;
    }
    case SDL_MOUSEBUTTONUP:
    {
        if(e.button.button != SDL_BUTTON_MIDDLE)
            break;
        nav.mmb_down = false;
        nav.mmb_shift_down = false;
        break;
    }
    case SDL_KEYDOWN:
    {
        vec3 new_dir = {};
        vec3 new_x;
        bool flip_x = false;

        switch(e.key.keysym.sym)
        {
        case SDLK_LSHIFT:
            nav.shift_down = true;
            break;
        case SDLK_q:
            nav.ortho = !nav.ortho;
            rebuild_proj_matrix(nav);
            break;
        case SDLK_x:
            new_dir = {-1,0,0};
            new_x = {0,0,-1};
            flip_x = true;
            break;
        case SDLK_y:
            new_dir = {0,-1,0};
            new_x = {1,0,0};
            break;
        case SDLK_z:
            new_dir = {0,0,-1};
            new_x = {1,0,0};
            flip_x = true;
            break;
        }
        if(dot(new_dir, new_dir) == 0)
            break;

        vec3 prev_dir = normalize(nav.center - nav.eye_pos);
        float prod = dot(new_dir, prev_dir);
        int flip_coeff = 1;

        if(nav.aligned && fabs(prod) > 0.99)
            flip_coeff = prod > 0 ? -1 : 1;

        float radius = length(nav.center - nav.eye_pos);
        nav.eye_pos = nav.center - (radius * flip_coeff * new_dir);
        nav.eye_x = flip_x ? flip_coeff * new_x : new_x;
        nav.ortho = true;
        nav.aligned = true;
        rebuild_view_matrix(nav);
        rebuild_proj_matrix(nav);
        break;
    }
    case SDL_KEYUP:
    {
        if(e.key.keysym.sym == SDLK_LSHIFT)
            nav.shift_down = false;
        break;
    }
    } // switch
}

bool intersect_AABB(Ray ray, BBox bbox, float& t)
{
    vec3 inv_dir = {1 / ray.dir.x, 1 / ray.dir.y, 1 / ray.dir.z};
    float tmin = FLT_MIN;
    float tmax = FLT_MAX;

    for(int i = 0; i < 3; ++i)
    {
        float t1 = (bbox.min[i] - ray.pos[i]) * inv_dir[i];
        float t2 = (bbox.max[i] - ray.pos[i]) * inv_dir[i];

        if(t1 > t2)
        {
            float tmp = t2;
            t2 = t1;
            t1 = tmp;
        }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);

        if(tmin > tmax)
            return false;
        if(tmax < 0)
            return false;
    }
    t = tmin > 0 ? tmin : tmax;
    return true;
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

// t is negative on a miss

float intersect_object(Ray ray, Object& obj, int& vert_id)
{
    vec3 inv_scale = {1/obj.scale.x, 1/obj.scale.y, 1/obj.scale.z};
    mat4 obj_f_world = scale(inv_scale) * transpose(obj.rot) * translate(-obj.pos);
    ray.pos = transform3(obj_f_world, ray.pos);
    ray.dir = mat4_to_mat3(obj_f_world) * ray.dir;
    int is_dir_neg[3] = {ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0};
    vert_id = -1;
    float t_min = -1;
    std::vector<BVHNode*> nodes_to_visit;
    nodes_to_visit.push_back(obj.mesh->bvh_root);

    while(nodes_to_visit.size())
    {
        BVHNode* node = nodes_to_visit.back();
        nodes_to_visit.pop_back();
        float t;
        bool hit = intersect_AABB(ray, node->bbox, t);

        if(!hit)
            continue;
        if(vert_id != -1 && t > t_min)
            continue;

        if(!node->vert_count)
        {
            if(is_dir_neg[node->split_axis])
            {
                nodes_to_visit.push_back(node->children[0]);
                nodes_to_visit.push_back(node->children[1]);
            }
            else
            {
                nodes_to_visit.push_back(node->children[1]);
                nodes_to_visit.push_back(node->children[0]);
            }
            continue;
        }
        int end = node->vert_offset + node->vert_count;

        for(int base = node->vert_offset; base < end; base += 3)
        {
            float t = intersect_triangle(ray, obj.mesh->vertices + base);

            if(t >= 0 && (vert_id == -1 || t <= t_min))
            {
                vert_id = base;
                t_min = t;
            }
        }
    }
    return t_min;
}

#define AXES_ALL 7

enum ControlMode
{
    CTRL_IDLE,
    CTRL_TRANSLATE,
    CTRL_SCALE,
    CTRL_ROTATE,
};

struct Control
{
    ControlMode mode;
    bool pivot_local;
    int axes;
    bool axes_local;
    bool shift_down;
    vec2 cursor_win;
    vec2 cursor_win_init;
    vec3 scale_init;
    mat4 rot_init;
    vec3 pos_init;
};

void control_init(Control& ctrl)
{
    ctrl.mode = CTRL_IDLE;
    ctrl.pivot_local = true;
    ctrl.shift_down = false;
}

void apply_transform(Control& ctrl, Nav& nav, Object& obj, vec2 cursor2_win)
{
    if(ctrl.axes == AXES_ALL)
        assert(!ctrl.axes_local);
    else if(ctrl.mode == CTRL_SCALE)
        ctrl.axes_local = true;

    vec3 axes[3];
    vec3 basis[3] = {{1,0,0},{0,1,0},{0,0,1}};
    int axis_count = 0;

    for(int i = 0; i < 3; ++i)
    {
        if(!(ctrl.axes & (1<<i)))
            continue;
        axes[axis_count] = ctrl.axes_local ? transform3(ctrl.rot_init, basis[i]) : basis[i];
        axis_count += 1;
    }

    vec2 pivot_ndc;
    {
        vec3 pivot = ctrl.pivot_local ? obj.pos : vec3{0,0,0};
        mat4 clip_f_world = nav.proj * nav.view;
        vec4 pivot_clip = clip_f_world * vec4{pivot.x, pivot.y, pivot.z, 1};
        pivot_ndc = (1 / pivot_clip.w) * vec2{pivot_clip.x, pivot_clip.y};
    }
    vec3 dir_to_eye = normalize(nav.eye_pos - nav.center);
    vec2 cursors_win[2];
    vec2 cursors_ndc[2];
    cursors_win[0] = ctrl.mode == CTRL_ROTATE ? ctrl.cursor_win : ctrl.cursor_win_init;
    cursors_win[1] = cursor2_win;

    for(int i = 0; i < 2; ++i)
    {
        cursors_ndc[i].x = ( 2 / nav.win_size.x) * cursors_win[i].x - 1;
        cursors_ndc[i].y = (-2 / nav.win_size.y) * cursors_win[i].y + 1;
    }

    switch(ctrl.mode)
    {
    case CTRL_ROTATE:
    {
        vec3 rot_axis = axes[0];

        if(axis_count == 2)
        {
            rot_axis = cross(axes[0], axes[1]);

            if( (ctrl.axes & 5) == 5) // negate cross(x,z) to follow a right-handed orientation
                rot_axis = -rot_axis;
        }
        else if(axis_count == 3)
            rot_axis = dir_to_eye;

        if(dot(rot_axis, dir_to_eye) < 0)
            rot_axis = -rot_axis;

        vec2 v1 = cursors_ndc[0] - pivot_ndc;
        vec2 v2 = cursors_ndc[1] - pivot_ndc;
        float a1 = atan2f(v1.y, v1.x);
        float a2 = atan2f(v2.y, v2.x);
        float da = a2 - a1;
        mat4 add_rot = rotate_axis(rot_axis, da);
        obj.rot = add_rot * obj.rot;

        if(!ctrl.pivot_local)
            obj.pos = transform3(add_rot, obj.pos);
        break;
    }
    case CTRL_SCALE:
    {
        vec2 v1 = cursors_ndc[0] - pivot_ndc;
        vec2 v2 = cursors_ndc[1] - pivot_ndc;
        float sign = dot(v1, v2) > 0 ? 1 : -1;
        float d0 = length(v1);
        float d1 = length(v2);
        vec3 add_scale;

        for(int i = 0; i < 3; ++i)
            add_scale[i] = (ctrl.axes & (1<<i)) ? sign * d1/d0 : 1;

        if(ctrl.pivot_local)
            obj.scale = mul_cwise(add_scale, ctrl.scale_init);
        else
        {
            mat4 world_f_model = translate(ctrl.pos_init) * ctrl.rot_init * scale(ctrl.scale_init);
            mat4 tf = ctrl.rot_init * scale(add_scale) * transpose(ctrl.rot_init) * world_f_model;
            decompose(tf, obj.pos, obj.rot, obj.scale);
        }
        break;
    }
    case CTRL_TRANSLATE:
    {
        vec3 plane_normal = dir_to_eye;

        if(axis_count == 1)
            plane_normal = cross( axes[0], normalize(cross(dir_to_eye, axes[0])) );
        else if(axis_count == 2)
            plane_normal = cross(axes[0], axes[1]);

        vec3 points[2];

        for(int i = 0; i < 2; ++i)
        {
            Ray ray = nav_get_cursor_ray(nav, cursors_win[i]);
            float t = intersect_plane(ray, plane_normal, obj.pos);
            points[i] = ray.pos + t*ray.dir;
        }

        vec3 diff = points[1] - points[0];

        if(axis_count == 1)
            diff = dot(axes[0], diff) * axes[0];

        obj.pos = ctrl.pos_init + diff;
        break;
    }
    default:
        assert(false);
    }
}

void control_process_event(Control& ctrl, Nav& nav, Object* obj, SDL_Event& e)
{
    switch(e.type)
    {
    case SDL_KEYDOWN:
    {
        switch(e.key.keysym.sym)
        {
        case SDLK_LSHIFT:
        {
            ctrl.shift_down = true;
            break;
        }
        case SDLK_v:
        {
            if(!ctrl.mode)
                ctrl.pivot_local = !ctrl.pivot_local;
            break;
        }
        case SDLK_s:
        case SDLK_r:
        case SDLK_g:
        {
            if(!obj)
                break;
            ControlMode new_mode = CTRL_SCALE;

            if(e.key.keysym.sym == SDLK_r)
                new_mode = CTRL_ROTATE;
            else if(e.key.keysym.sym == SDLK_g)
                new_mode = CTRL_TRANSLATE;

            if(ctrl.mode == new_mode)
                break;

            if(!ctrl.mode)
            {
                ctrl.mode = new_mode;
                ctrl.axes = AXES_ALL;
                ctrl.axes_local = false;
                ctrl.cursor_win_init = ctrl.cursor_win;
                ctrl.scale_init = obj->scale;
                ctrl.rot_init = obj->rot;
                ctrl.pos_init = obj->pos;
            }
            else
            {
                ctrl.mode = new_mode;
                ctrl.cursor_win_init = ctrl.cursor_win;
                obj->scale = ctrl.scale_init;
                obj->rot = ctrl.rot_init;
                obj->pos = ctrl.pos_init;
            }
            apply_transform(ctrl, nav, *obj, ctrl.cursor_win);
            break;
        }
        case SDLK_x:
        case SDLK_y:
        case SDLK_z:
        {
            if(!ctrl.mode)
                break;
            int axis_sel = 1 << 0;

            if(e.key.keysym.sym == SDLK_y)
                axis_sel = 1 << 1;
            else if(e.key.keysym.sym == SDLK_z)
                axis_sel = 1 << 2;

            int new_axes = ctrl.shift_down ? ~axis_sel : axis_sel;

            if(new_axes != ctrl.axes && ~new_axes != ctrl.axes)
            {
                ctrl.axes = new_axes;
                ctrl.axes_local = false;
            }
            else
            {
                ctrl.axes = ctrl.axes_local ? AXES_ALL : new_axes;
                ctrl.axes_local = !ctrl.axes_local;
            }
            obj->scale = ctrl.scale_init;
            obj->rot = ctrl.rot_init;
            obj->pos = ctrl.pos_init;
            apply_transform(ctrl, nav, *obj, ctrl.cursor_win);
            break;
        }
        case SDLK_ESCAPE:
        {
            if(!ctrl.mode)
                break;
            ctrl.mode = CTRL_IDLE;
            obj->scale = ctrl.scale_init;
            obj->rot = ctrl.rot_init;
            obj->pos = ctrl.pos_init;
            break;
        }
        }
        break;
    }
    case SDL_KEYUP:
    {
        if(e.key.keysym.sym == SDLK_LSHIFT)
            ctrl.shift_down = false;
        break;
    }
    case SDL_MOUSEMOTION:
    {
        vec2 cursor_win = {(float)e.motion.x, (float)e.motion.y};

        if(ctrl.mode)
            apply_transform(ctrl, nav, *obj, cursor_win);
        ctrl.cursor_win = cursor_win;
        break;
    }
    case SDL_MOUSEBUTTONDOWN:
    {
        if(!ctrl.mode)
            break;

        if(e.button.button == SDL_BUTTON_LEFT)
            ctrl.mode = CTRL_IDLE;
        else if(e.button.button == SDL_BUTTON_RIGHT)
        {
            ctrl.mode = CTRL_IDLE;
            obj->scale = ctrl.scale_init;
            obj->rot = ctrl.rot_init;
            obj->pos = ctrl.pos_init;
        }
        break;
    }
    }
}

void draw_segment(GLuint prog, vec3 v1, vec3 v2, vec3 color)
{
    vec3 vertices[] = {v1, v2};
    GLuint vao, vbo;
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    GLint color_loc = glGetUniformLocation(prog, "color");
    glUniform3f(color_loc, color.x, color.y, color.z);
    glDrawArrays(GL_LINES, 0, 2);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        assert(false);
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 100, 100, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
        assert(false);

    GLuint prog = create_program(_vert, _frag);
    GLuint prog_solid = create_program(_vert_solid, _frag_solid);

    std::vector<Mesh> meshes;
    meshes.push_back(load_mesh("cube.obj"));
    meshes.push_back(load_mesh("../model.obj"));
    std::vector<Object> objects;
    objects.push_back({}); // dummy object to reserve 0 index
    Control control;
    control_init(control);

    Nav nav;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        nav_init(nav, vec3{2,2,8}, width, height, 60, 0.1, 1000);
    }

    bool quit = false;
    std::size_t sel_id = 0;
    int sel_vert_id = 0;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            bool ctrl_active = control.mode;
            bool nav_active = nav.mmb_down || nav.mmb_shift_down;

            if(ctrl_active)
                assert(sel_id);

            if(event.type == SDL_QUIT)
                quit = true;

            if(!ctrl_active && !nav_active && event.type == SDL_KEYDOWN)
            {
                Mesh* spawn_mesh = nullptr;
                vec3 spawn_pos = {};
                mat4 spawn_rot = identity4();
                vec3 spawn_scale = {1,1,1};

                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                case SDLK_1:
                    spawn_mesh = &meshes[0];
                    break;
                case SDLK_2:
                    spawn_mesh = &meshes[1];
                    break;
                case SDLK_DELETE:
                    if(!sel_id)
                        break;
                    objects.erase(objects.begin() + sel_id);
                    sel_id = 0;
                    break;
                case SDLK_d:
                    if(!sel_id)
                        break;
                    spawn_mesh = objects[sel_id].mesh;
                    spawn_pos = objects[sel_id].pos;
                    spawn_rot = objects[sel_id].rot;
                    spawn_scale = objects[sel_id].scale;
                    break;
                }
                if(spawn_mesh)
                {
                    Object obj;
                    obj.mesh = spawn_mesh;
                    obj.pos = spawn_pos;
                    obj.rot = spawn_rot;
                    obj.scale = spawn_scale;
                    sel_id = objects.size();
                    sel_vert_id = 0;
                    objects.push_back(obj);
                }
            }

            if(!ctrl_active || event.type == SDL_MOUSEMOTION)
                nav_process_event(nav, event);

            if(!nav_active || event.type == SDL_MOUSEMOTION)
            {
                Object* obj = sel_id ? &objects[sel_id] : nullptr;
                control_process_event(control, nav, obj, event);
            }

            if(!ctrl_active && event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT)
            {
                sel_id = 0;
                float t_min = FLT_MAX;
                Ray ray = nav_get_cursor_ray(nav, nav.cursor_win);

                for(std::size_t i = 1; i < objects.size(); ++i)
                {
                    int vid;
                    float t = intersect_object(ray, objects[i], vid);

                    if(t >= 0 && t < t_min)
                    {
                        sel_id = i;
                        t_min = t;
                        sel_vert_id = vid;
                    }
                }
            }
        }
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        (void)dt;
        prev_counter = current_counter;

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);

        // objects

        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog);
        {
            vec3 light_intensity = {1,1,1};
            vec3 light_dir = normalize(nav.eye_pos - nav.center);
            vec3 specular_color = {0.2, 0.2, 0.2};
            GLint proj_loc = glGetUniformLocation(prog, "proj");
            GLint view_loc = glGetUniformLocation(prog, "view");
            GLint light_int_loc = glGetUniformLocation(prog, "light_intensity");
            GLint light_dir_loc = glGetUniformLocation(prog, "light_dir");
            GLint ambient_int_loc = glGetUniformLocation(prog, "ambient_intensity");
            GLint eye_pos_loc = glGetUniformLocation(prog, "eye_pos");
            GLint specular_color_loc = glGetUniformLocation(prog, "specular_color");
            GLint specular_exp_loc = glGetUniformLocation(prog, "specular_exp");

            glUniformMatrix4fv(view_loc, 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, nav.proj.data);
            glUniform3fv(light_int_loc, 1, &light_intensity.x);
            glUniform3fv(light_dir_loc, 1, &light_dir.x);
            glUniform1f(ambient_int_loc, 0.01);
            glUniform3fv(eye_pos_loc, 1, &nav.eye_pos.x);
            glUniform3fv(specular_color_loc, 1, &specular_color.x);
            glUniform1f(specular_exp_loc, 50);
        }

        for(std::size_t i = 1; i < objects.size(); ++i)
        {
            Object& obj = objects[i];
            GLint diffuse_color_loc = glGetUniformLocation(prog, "diffuse_color");
            GLint model_loc = glGetUniformLocation(prog, "model");
            vec3 diffuse_color = vec3{0.3, 0.3, 0.3};

            if(i == sel_id)
            {
                if(!control.mode)
                    diffuse_color = vec3{0.6, 0.3, 0};
                else
                    diffuse_color = vec3{0.6, 0.1, 0};
            }
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

        // solid color

        glUseProgram(prog_solid);
        {
            GLint proj_loc = glGetUniformLocation(prog_solid, "proj");
            GLint view_loc = glGetUniformLocation(prog_solid, "view");
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(view_loc, 1, GL_TRUE, nav.view.data);
        }

        // selected face

        if(sel_id)
        {
            Object& obj = objects[sel_id];
            Vertex* verts = obj.mesh->vertices + sel_vert_id;
            vec3 coords[] = {verts[0].pos, verts[1].pos, verts[2].pos};
            vec3 normal = normalize(cross(coords[1] - coords[0], coords[2] - coords[0]));

            for(vec3& coord: coords)
                coord = coord + 0.001 * normal;

            mat4 model = translate(obj.pos) * obj.rot * scale(obj.scale);
            GLint model_loc = glGetUniformLocation(prog_solid, "model");
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model.data);
            vec3 color = {0,1,0};
            draw_segment(prog_solid, coords[0], coords[1], color);
            draw_segment(prog_solid, coords[1], coords[2], color);
            draw_segment(prog_solid, coords[2], coords[0], color);
        }

        // axes
        {
            mat4 model = identity4();
            GLint model_loc = glGetUniformLocation(prog_solid, "model");
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model.data);
        }
        float d = 250.f;
        draw_segment(prog_solid, vec3{-d,0,0}, vec3{d,0,0}, vec3{0.3,0,0});
        draw_segment(prog_solid, vec3{0,0,-d}, vec3{0,0,d}, vec3{0.01,0.01,0.3});
        // xz plane
        vec3 color_xz = {0.05,0.05,0.05};
        draw_segment(prog_solid, vec3{d,0,d},  vec3{-d,0,d},  color_xz);
        draw_segment(prog_solid, vec3{d,0,-d}, vec3{-d,0,-d}, color_xz);
        draw_segment(prog_solid, vec3{d,0,d},  vec3{d,0,-d},  color_xz);
        draw_segment(prog_solid, vec3{-d,0,d}, vec3{-d,0,-d}, color_xz);

        glDisable(GL_DEPTH_TEST);
        vec3 pivot = control.pivot_local ? control.pos_init : vec3{0,0,0};

        if(control.mode && control.axes != AXES_ALL)
        {
            vec3 basis[3] = {{1,0,0},{0,1,0},{0,0,1}};

            for(int i = 0; i < 3; ++i)
            {
                if(!(control.axes & (1<<i)))
                    continue;
                vec3 axis = basis[i];

                if(control.axes_local)
                    axis = mat4_to_mat3(control.rot_init) * axis;

                vec3 p1 = pivot + d * axis;
                vec3 p2 = pivot - d * axis;
                vec3 color = {0.3,0.3,0.3};
                color[i] = 1;
                draw_segment(prog_solid, p1, p2, color);
            }
        }

        if(control.mode == CTRL_ROTATE || control.mode == CTRL_SCALE)
        {
            Ray ray = nav_get_cursor_ray(nav, nav.cursor_win);
            float t = intersect_plane(ray, normalize(nav.center - nav.eye_pos), nav.center);
            draw_segment(prog_solid, pivot, ray.pos + t*ray.dir, vec3{1,1,1});
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
