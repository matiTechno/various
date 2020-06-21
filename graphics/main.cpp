#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <pthread.h>
#include "glad.h"
#include "main.hpp"

// these can be tweaked
#define BVH_MAX_DEPTH 7
#define THREAD_COUNT 12
#define THREAD_PX_CHUNK_SIZE 24

struct RenderCmd
{
    mat4 view;
    mat4 proj;
    vec3 light_intensity;
    vec3 light_dir;
    float ambient_intensity;
    vec3 eye_pos;
    // model data
    vec3* positions;
    vec3* normals;
    int vertex_count;
    mat4 model_transform;
    vec3 diffuse_color;
    vec3 specular_color;
    float specular_exp;
    bool debug;
};

static const char* src_vert = R"(
#version 330

uniform mat4 view;
uniform mat4 proj;
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
    frag_normal = mat3(model) * normal;
}
)";

static const char* src_frag = R"(
#version 330

uniform vec3 light_intensity;
uniform vec3 light_dir;
uniform float ambient_intensity;
uniform vec3 diffuse_color;
uniform vec3 specular_color;
uniform float specular_exp;
uniform vec3 eye_pos;
uniform int debug;

in vec3 frag_pos;
in vec3 frag_normal;
out vec3 out_color;

void main()
{
    if(bool(debug))
    {
        out_color = vec3(1,1,1);
        return;
    }
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

static const char* _ras_src_vert = R"(
#version 330
layout(location = 0) in vec2 pos;
out vec2 tex_coord;

void main()
{
    gl_Position = vec4(pos, 0, 1);
    tex_coord = (pos + vec2(1)) / 2;
}
)";

static const char* _ras_src_frag = R"(
#version 330
out vec3 out_color;
in vec2 tex_coord;
uniform sampler2D sampler;

void main()
{
    vec4 sample = texture(sampler, tex_coord);
    out_color = sample.rgb;
}
)";

#define VFLOAT_COUNT 6

union Varying
{
    struct
    {
        vec3 pos;
        vec3 normal;
    };
    float data[VFLOAT_COUNT];
};

Varying operator+(Varying lhs, Varying rhs)
{
    for(int i = 0; i < VFLOAT_COUNT; ++i)
        lhs.data[i] += rhs.data[i];
    return lhs;
}

Varying operator-(Varying lhs, Varying rhs)
{
    for(int i = 0; i < VFLOAT_COUNT; ++i)
        lhs.data[i] -= rhs.data[i];
    return lhs;
}

Varying operator*(float scalar, Varying v)
{
    for(int i = 0; i < VFLOAT_COUNT; ++i)
        v.data[i] *= scalar;
    return v;
}

struct BV_node
{
    union
    {
        struct
        {
            vec3* positions;
            vec3* normals;
            int vert_count;
        };
        struct
        {
            BV_node* children[8];
            int child_count;
        };
    };

    vec3 bbox_min;
    vec3 bbox_max;
    bool leaf;
};

void compute_bbox(vec3* positions, int vert_count, vec3& bbox_min, vec3& bbox_max)
{
    assert(vert_count);
    bbox_min = positions[0];
    bbox_max = positions[0];

    for(int i = 1; i < vert_count; ++i)
    {
        vec3 pos = positions[i];

        for(int c = 0; c < 3; ++c)
        {
            bbox_min[c] = min(bbox_min[c], pos[c]);
            bbox_max[c] = max(bbox_max[c], pos[c]);
        }
    }
}

BV_node* get_child_for_insert(BV_node* node, vec3* positions)
{
    vec3 center = (1.f/3) * (positions[0] + positions[1] + positions[2]);
    vec3 bbox_mid = 0.5 * (node->bbox_min + node->bbox_max);
    vec3 bbox_min = node->bbox_min;
    vec3 bbox_max = bbox_mid;
    int idx = 0;
    int weight = 4;

    for(int i = 0; i < 3; ++i)
    {
        if(center[i] > bbox_mid[i])
        {
            bbox_min[i] = bbox_mid[i];
            bbox_max[i] = node->bbox_max[i];
            idx += weight;
        }
        weight /= 2;
    }

    BV_node*& child = node->children[idx];

    if(child == nullptr)
    {
        child = (BV_node*)malloc(sizeof(BV_node));
        memset(child, 0, sizeof(BV_node));
        child->bbox_min = bbox_min;
        child->bbox_max = bbox_max;
        child->leaf = true;
        node->child_count += 1;
    }
    return child;
}

void insert_triangle(BV_node* node, int depth, vec3* positions, vec3* normals)
{
    // convert to an internal node

    if(depth != BVH_MAX_DEPTH && node->leaf && node->vert_count)
    {
        vec3* tmp_pos = node->positions;
        vec3* tmp_norm = node->normals;
        node->leaf = false;
        memset(node->children, 0, sizeof node->children);
        node->child_count = 0;
        BV_node* child = get_child_for_insert(node, positions);
        insert_triangle(child, depth + 1, tmp_pos, tmp_norm);
        free(tmp_pos);
        free(tmp_norm);
    }

    if(node->leaf)
    {
        int base = node->vert_count;
        node->vert_count += 3;
        node->positions = (vec3*)realloc(node->positions, sizeof(vec3) * node->vert_count);
        node->normals = (vec3*)realloc(node->normals, sizeof(vec3) * node->vert_count);

        for(int i = 0; i < 3; ++i)
        {
            node->positions[base+i] = positions[i];
            node->normals[base+i] = normals[i];
        }
    }
    else
    {
        BV_node* child = get_child_for_insert(node, positions);
        insert_triangle(child, depth + 1, positions, normals);
    }
}

void optimize_BVH(BV_node* node)
{
    if(node->leaf)
    {
        assert(node->vert_count);
        compute_bbox(node->positions, node->vert_count, node->bbox_min, node->bbox_max);
        return;
    }

    BV_node* sorted[8] = {}; // = {} is only for an easier debugging
    int sorted_count = 0;

    for(int i = 0; i < 8; ++i)
    {
        if(node->children[i])
        {
            sorted[sorted_count] = node->children[i];
            sorted_count += 1;
        }
    }
    memcpy(node->children, sorted, sizeof sorted);

    for(int i = 0; i < node->child_count; ++i)
        optimize_BVH(node->children[i]);

    // if a node has only one child replace it with the child and return

    if(node->child_count == 1)
    {
        BV_node* child = node->children[0];
        *node = *child;
        free(child);
        return;
    }

    node->bbox_min = node->children[0]->bbox_min;
    node->bbox_max = node->children[0]->bbox_max;

    for(int i = 1; i < node->child_count; ++i)
    {
        BV_node* child = node->children[i];

        for(int c = 0; c < 3; ++c)
        {
            node->bbox_min[c] = min(node->bbox_min[c], child->bbox_min[c]);
            node->bbox_max[c] = max(node->bbox_max[c], child->bbox_max[c]);
        }
    }
}

BV_node* build_BVH(vec3* positions, vec3* normals, int vert_count)
{
    BV_node* root = (BV_node*)malloc(sizeof(BV_node));
    memset(root, 0, sizeof(BV_node));
    compute_bbox(positions, vert_count, root->bbox_min, root->bbox_max);
    root->leaf = true;

    for(int base = 0; base < vert_count; base += 3)
        insert_triangle(root, 0, positions + base, normals + base);

    optimize_BVH(root);
    return root;
}

// software rasterizer / raytracer global state
struct
{
    int width;
    int height;
    float* depth_buf;
    u8* color_buf;
    // opengl resources for ras_display()
    GLuint vbo;
    GLuint vao;
    GLuint program;
    GLuint texture;
    // raytracer data for multithreading
    uint64_t threads_busy_mask;
    int next_pixel_idx;
    pthread_cond_t* cv;
    pthread_mutex_t* mutex;
    pthread_t threads[THREAD_COUNT];
    RenderCmd* raytracer_cmd;
    BV_node* bvh_root;
} _ras;

void* raytracer_thread_start(void*);

void ras_init()
{
    _ras.depth_buf = nullptr;
    _ras.color_buf = nullptr;
    _ras.width = 0;
    _ras.height = 0;

    float verts[] = {-1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1};

    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    _ras.program = glCreateProgram();

    glGenBuffers(1, &_ras.vbo);
    glGenVertexArrays(1, &_ras.vao);

    glBindBuffer(GL_ARRAY_BUFFER, _ras.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glBindVertexArray(_ras.vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glShaderSource(vert_shader, 1, &_ras_src_vert, nullptr);
    glCompileShader(vert_shader);
    glShaderSource(frag_shader, 1, &_ras_src_frag, nullptr);
    glCompileShader(frag_shader);
    glAttachShader(_ras.program, vert_shader);
    glAttachShader(_ras.program, frag_shader);
    glLinkProgram(_ras.program);

    glGenTextures(1, &_ras.texture);
    glBindTexture(GL_TEXTURE_2D, _ras.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    _ras.cv = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
    _ras.mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));

    _ras.threads_busy_mask = 0;
    int rc = pthread_cond_init(_ras.cv, nullptr);
    assert(!rc);
    rc = pthread_mutex_init(_ras.mutex, nullptr);
    assert(!rc);
    assert(THREAD_COUNT <= sizeof(_ras.threads_busy_mask) * 8);

    for(int i = 0; i < THREAD_COUNT; ++i)
    {
        rc = pthread_create(_ras.threads + i, nullptr, raytracer_thread_start, (void*)((uint64_t)(1) << i) );
        assert(!rc);
    }
}

void ras_viewport(int width, int height)
{
    if(width == _ras.width && height == _ras.height)
        return;
    _ras.width = width;
    _ras.height = height;
    free(_ras.depth_buf);
    free(_ras.color_buf);
    _ras.depth_buf = (float*)malloc(width * height * sizeof(float));
    // note: we are using 4 bytes per pixel to avoid problems with updating OpenGL texture (alignment issues)
    _ras.color_buf = (u8*)malloc(width * height * 4);
    glBindTexture(GL_TEXTURE_2D, _ras.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void ras_display()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, _ras.width, _ras.height);
    glBindTexture(GL_TEXTURE_2D, _ras.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _ras.width, _ras.height, GL_RGBA, GL_UNSIGNED_BYTE, _ras.color_buf);
    glBindVertexArray(_ras.vao);
    glUseProgram(_ras.program);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void ras_clear_buffers()
{
    for(int i = 0; i < _ras.width * _ras.height; ++i)
        _ras.depth_buf[i] = FLT_MAX;
    memset(_ras.color_buf, 0, _ras.width * _ras.height * 4);
}

void _ras_draw1(RenderCmd& cmd, vec4* coords, Varying* varyings);
void _ras_draw2(RenderCmd& cmd, vec4* coords, Varying* varyings);

void ras_draw(RenderCmd& cmd)
{
    // something like a fixed vertex shader
    mat4 proj_view = cmd.proj * cmd.view;
    mat3 model3 = mat4_to_mat3(cmd.model_transform);

    for(int base = 0; base < cmd.vertex_count; base += 3)
    {
        vec4 coords[3];
        Varying varyings[3];

        for(int i = 0; i < 3; ++i)
        {
            vec3 vert_pos = cmd.positions[base + i];
            coords[i] = {vert_pos.x, vert_pos.y, vert_pos.z, 1};
            coords[i] = cmd.model_transform * coords[i];
            varyings[i].pos = {coords[i].x, coords[i].y, coords[i].z};
            coords[i] = proj_view * coords[i];
            varyings[i].normal = model3 * cmd.normals[base + i];
        }
        _ras_draw1(cmd, coords, varyings);
    }
}

#define W_CLIP 0.001f

void _ras_draw1(RenderCmd& cmd, vec4* coords, Varying* varyings)
{
    int out_codes[3];
    int w_code = 0;

    for(int i = 0; i < 3; ++i)
    {
        float w = coords[i].w;
        int code = 0;
        code = code | ((coords[i].x < -w) << 0);
        code = code | ((coords[i].y < -w) << 1);
        code = code | ((coords[i].z < -w) << 2);
        code = code | ((coords[i].x > w)  << 3);
        code = code | ((coords[i].y > w)  << 4);
        code = code | ((coords[i].z > w)  << 5);
        out_codes[i] = code;
        w_code += w < W_CLIP;
    }
    // all vertices are on the external side of a clipping plane
    if(out_codes[0] & out_codes[1] & out_codes[2])
        return;

    // w=0 plane clipping

    if(w_code == 1) // in this case a new triangle is generated
    {
        vec4 coords2[3];
        Varying varyings2[3];
        memcpy(coords2, coords, sizeof(coords2));
        memcpy(varyings2, varyings, sizeof(varyings2));
        int idx_next = 0;
        int idx_prev = 1;
        int idx_curr = 2;

        for(int i = 0; i < 3; ++i)
        {
            float w0 = coords[idx_curr].w;

            if(w0 < W_CLIP)
            {
                float w1 = coords[idx_prev].w;
                float w2 = coords[idx_next].w;
                float t1 = (W_CLIP - w0) / (w1 - w0);
                float t2 = (W_CLIP - w0) / (w2 - w0);
                coords[idx_curr] = coords[idx_curr] + t1*(coords[idx_prev] - coords[idx_curr]);
                varyings[idx_curr] = varyings[idx_curr] + t1*(varyings[idx_prev] - varyings[idx_curr]);
                coords2[idx_prev] = coords[idx_curr];
                varyings2[idx_prev] = varyings[idx_curr];
                coords2[idx_curr] = coords2[idx_curr] + t2*(coords2[idx_next] - coords2[idx_curr]);
                varyings2[idx_curr] = varyings2[idx_curr] + t2*(varyings2[idx_next] - varyings2[idx_curr]);
                break;
            }
            idx_prev = idx_curr;
            idx_curr = idx_next;
            idx_next += 1;
        }
        _ras_draw2(cmd, coords, varyings);
        _ras_draw2(cmd, coords2, varyings2);
    }
    else if(w_code == 2)
    {
        int idx_next = 0;
        int idx_prev = 1;
        int idx_curr = 2;

        for(int i = 0; i < 3; ++i)
        {
            float w0 = coords[idx_curr].w;

            if(w0 >= W_CLIP)
            {
                float w1 = coords[idx_prev].w;
                float w2 = coords[idx_next].w;
                float t1 = (W_CLIP - w0) / (w1 - w0);
                float t2 = (W_CLIP - w0) / (w2 - w0);
                coords[idx_prev] = coords[idx_curr] + t1*(coords[idx_prev] - coords[idx_curr]);
                varyings[idx_prev] = varyings[idx_curr] + t1*(varyings[idx_prev] - varyings[idx_curr]);
                coords[idx_next] = coords[idx_curr] + t2*(coords[idx_next] - coords[idx_curr]);
                varyings[idx_next] = varyings[idx_curr] + t2*(varyings[idx_next] - varyings[idx_curr]);
                break;
            }
            idx_prev = idx_curr;
            idx_curr = idx_next;
            idx_next += 1;
        }
        _ras_draw2(cmd, coords, varyings);
    }
    else
        _ras_draw2(cmd, coords, varyings);
}

void _ras_frag_shader(RenderCmd& cmd, int idx, Varying varying)
{
    vec3 L = cmd.light_dir;
    vec3 N = normalize(varying.normal);
    vec3 ambient_comp = cmd.ambient_intensity * cmd.diffuse_color;
    vec3 diff_comp = max(dot(N, L), 0) * mul_cwise(cmd.diffuse_color, cmd.light_intensity);
    vec3 V = normalize(cmd.eye_pos - varying.pos);
    vec3 H = normalize(V + L);
    vec3 spec_comp = powf( max(dot(N, H), 0), cmd.specular_exp) * (dot(N, L) > 0) * mul_cwise(cmd.specular_color, cmd.light_intensity);
    vec3 color = ambient_comp + diff_comp + spec_comp;
    color.x = powf(color.x, 1/2.2);
    color.y = powf(color.y, 1/2.2);
    color.z = powf(color.z, 1/2.2);
    color.x = min(color.x, 1.f);
    color.y = min(color.y, 1.f);
    color.z = min(color.z, 1.f);
    _ras.color_buf[idx*4 + 0] = 255.f * color.x + 0.5f;
    _ras.color_buf[idx*4 + 1] = 255.f * color.y + 0.5f;
    _ras.color_buf[idx*4 + 2] = 255.f * color.z + 0.5f;
}

float signed_area(float lhs_x, float lhs_y, float rhs_x, float rhs_y)
{
    return (lhs_x * rhs_y) - (lhs_y * rhs_x);
}

void _ras_draw2(RenderCmd& cmd, vec4* coords, Varying* varyings)
{
    for(int i = 0; i < 3; ++i)
    {
        // perspective division
        coords[i].x = coords[i].x / coords[i].w;
        coords[i].y = coords[i].y / coords[i].w;
        coords[i].z = coords[i].z / coords[i].w;

        // viewport transformation
        coords[i].x = (_ras.width / 2.f) * (coords[i].x + 1.f);
        coords[i].y = (_ras.height / 2.f) * (coords[i].y + 1.f);
        coords[i].z = (1.f / 2.f) * (coords[i].z + 1.f);
        coords[i].w = 1.f / coords[i].w;
    }

    // face culling
    float edge01x = coords[1].x - coords[0].x;
    float edge01y = coords[1].y - coords[0].y;
    float edge12x = coords[2].x - coords[1].x;
    float edge12y = coords[2].y - coords[1].y;
    float edge20x = coords[0].x - coords[2].x;
    float edge20y = coords[0].y - coords[2].y;
    float area = signed_area(edge01x, edge01y, edge12x, edge12y);

    if(area < 0)
        return;

    int min_x = coords[0].x;
    int max_x = min_x;
    int min_y = coords[0].y;
    int max_y = min_y;

    for(int i = 1; i < 3; ++i)
    {
        min_x = min(min_x, coords[i].x);
        max_x = max(max_x, coords[i].x);
        min_y = min(min_y, coords[i].y);
        max_y = max(max_y, coords[i].y);
    }

    // clip a triangle bounding box to a viewport area
    min_x = min_x < 0 ? 0 : min_x;
    min_y = min_y < 0 ? 0 : min_y;
    max_x = max_x >= _ras.width  ? _ras.width  - 1 : max_x;
    max_y = max_y >= _ras.height ? _ras.height - 1 : max_y;

    for(int y = min_y; y <= max_y; ++y)
    {
        for(int x = min_x; x <= max_x; ++x)
        {
            float px = x + 0.5f;
            float py = y + 0.5f;
            float b0 = signed_area(edge12x, edge12y, px - coords[1].x, py - coords[1].y) / area;
            float b1 = signed_area(edge20x, edge20y, px - coords[2].x, py - coords[2].y) / area;
            float b2 = signed_area(edge01x, edge01y, px - coords[0].x, py - coords[0].y) / area;

            if(b0 < 0 || b1 < 0 || b2 < 0)
                continue;

            float depth = (b0 * coords[0].z) + (b1 * coords[1].z) + (b2 * coords[2].z);

            // per fragment clipping against near and far planes
            if(depth < 0.f || depth > 1.f)
                continue;

            int idx = y * _ras.width + x;

            // z-test
            if(depth > _ras.depth_buf[idx])
                continue;

            _ras.depth_buf[idx] = depth;

            // perspective correct interpolation
            Varying varying = (b0 * coords[0].w * varyings[0]) + (b1 * coords[1].w * varyings[1]) + (b2 * coords[2].w * varyings[2]);
            varying = 1.f / (b0 * coords[0].w + b1 * coords[1].w + b2 * coords[2].w) * varying;
            _ras_frag_shader(cmd, idx, varying);
        }
    }
}

void extract_frustum(mat4 persp, float& l, float& r, float& b, float& t, float& n, float& f)
{
    assert(persp.data[14]);
    n = persp.data[11] / (persp.data[10] - 1);
    f = persp.data[11] / (persp.data[10] + 1);
    l = n*(persp.data[2] - 1) / persp.data[0];
    r = n*(persp.data[2] + 1) / persp.data[0];
    b = n*(persp.data[6] - 1) / persp.data[5];
    t = n*(persp.data[6] + 1) / persp.data[5];
}

// returns a negative value on a miss

float ray_AABB_test(vec3 ray_start, vec3 ray_dir, vec3 box_min, vec3 box_max)
{
    vec3 inv_dir = {1 / ray_dir.x, 1 / ray_dir.y, 1 / ray_dir.z};

    float tx1 = (box_min.x - ray_start.x) * inv_dir.x;
    float tx2 = (box_max.x - ray_start.x) * inv_dir.x;

    if(tx1 > tx2)
    {
        float tmp = tx2;
        tx2 = tx1;
        tx1 = tmp;
    }

    float ty1 = (box_min.y - ray_start.y) * inv_dir.y;
    float ty2 = (box_max.y - ray_start.y) * inv_dir.y;

    if(ty1 > ty2)
    {
        float tmp = ty2;
        ty2 = ty1;
        ty1 = tmp;
    }

    float tz1 = (box_min.z - ray_start.z) * inv_dir.z;
    float tz2 = (box_max.z - ray_start.z) * inv_dir.z;

    if(tz1 > tz2)
    {
        float tmp = tz2;
        tz2 = tz1;
        tz1 = tmp;
    }

    float tmin = max(tx1, max(ty1, tz1));
    float tmax = min(tx2, min(ty2, tz2));

    if(tmin <= tmax)
        return tmin;
    return -1;
}

struct Intersection
{
    BV_node* node;
    float t;
};

void raytracer_thread_work(RenderCmd& cmd, BV_node* bvh_root)
{
    // exit if projection is orthographic
    // todo
    if(!cmd.proj.data[14])
        return;

    mat3 model3 = mat4_to_mat3(cmd.model_transform); // this is used to transform normals
    float left, right, bot, top, near, far;
    extract_frustum(cmd.proj, left, right, bot, top, near, far);
    int width = _ras.width;
    int height = _ras.height;

    mat4 model_from_view = invert_coord_change(cmd.model_transform) * invert_coord_change(cmd.view); // model_from_world * world_from_view
    // with respect to a model space; intersection test is performed in a model space
    mat3 eye_basis = mat4_to_mat3(model_from_view);
    vec3 eye_pos = {model_from_view.data[3], model_from_view.data[7], model_from_view.data[11]};
    vec3 eye_dir = {-eye_basis.data[2], -eye_basis.data[5], -eye_basis.data[8]};

    // note: bounding volumes are allowed to overlap
    std::vector<Intersection> queue; // sorted in a descending order of t

    int image_size = width * height;

    for(;;)
    {
        int idx_start = __sync_fetch_and_add(&_ras.next_pixel_idx, THREAD_PX_CHUNK_SIZE);

        if(idx_start >= image_size)
            break;

        int idx_end = idx_start + min(THREAD_PX_CHUNK_SIZE, image_size - idx_start);

        for(int idx = idx_start; idx < idx_end; ++idx)
        {
            queue.clear();
            vec3 ray_dir;
            {
                int pix_x = idx % width;
                int pix_y = idx / width;
                float x = ((right - left)/width) * (pix_x + 0.5f) + left;
                float y = ((top - bot)/height) * (pix_y + 0.5f) + bot;
                float z = -near;
                ray_dir = normalize( eye_basis * vec3{x, y, z} );

                float t = ray_AABB_test(eye_pos, ray_dir, bvh_root->bbox_min, bvh_root->bbox_max);

                if(t >= 0)
                    queue.push_back({bvh_root, t});
            }

            // closest intersected fragment
            bool frag_valid = false; // if passed all the tests
            float frag_t = _ras.depth_buf[idx]; // t parameter of the intersection
            vec3 frag_pos;
            vec3 frag_normal;

            while(queue.size())
            {
                if(queue.back().t > frag_t)
                    break;

                BV_node* node = queue.back().node;
                queue.pop_back();

                if(!node->leaf)
                {
                    for(int child_id = 0; child_id < node->child_count; ++child_id)
                    {
                        BV_node* child = node->children[child_id];
                        float t = ray_AABB_test(eye_pos, ray_dir, child->bbox_min, child->bbox_max);

                        if(t < 0)
                            continue;

                        // node is inserted before an element at insert_idx
                        int insert_idx = queue.size();

                        for(; insert_idx > 0; --insert_idx)
                        {
                            if(t <= queue[insert_idx - 1].t)
                                break;
                        }
                        queue.insert(queue.begin() + insert_idx, {child, t});
                    }
                    continue;
                }

                // test against triangles of a leaf node

                for(int base = 0; base < node->vert_count; base += 3)
                {
                    vec3* coords = node->positions + base;

                    vec3 edge01 = coords[1] - coords[0];
                    vec3 edge12 = coords[2] - coords[1];
                    vec3 edge20 = coords[0] - coords[2];
                    vec3 normal = cross(edge01, edge12);
                    float area = length(normal);
                    normal = (1 / area) * normal; // normalize

                    // face culling
                    if(dot(normal, -ray_dir) < 0.f)
                        continue;

                    // plane-ray intersection
                    float t = (dot(normal, coords[0]) - dot(normal, eye_pos)) / dot(normal, ray_dir);

                    // depth test
                    if(t > frag_t)
                        continue;

                    float dist_z = dot(eye_dir, t*ray_dir);

                    // far / near plane cull
                    if(dist_z < near || dist_z > far)
                        continue;

                    vec3 new_frag_pos = eye_pos + t*ray_dir; // intersection point in a model space
                    float b0 = dot(normal, cross(edge12, new_frag_pos - coords[1])) / area;
                    float b1 = dot(normal, cross(edge20, new_frag_pos - coords[2])) / area;
                    float b2 = dot(normal, cross(edge01, new_frag_pos - coords[0])) / area;

                    // point on a plane is not in a triangle
                    if(b0 < 0 || b1 < 0 || b2 < 0)
                        continue;

                    frag_valid = true;
                    frag_t = t;
                    frag_pos = new_frag_pos;
                    frag_normal = (b0 * node->normals[base+0]) + (b1 * node->normals[base+1]) + (b2 * node->normals[base+2]); // model space
                }
            }

            if(!frag_valid)
                continue;

            // transform frag_pos and frag_normal from the model space to a world space
            vec4 hp = {frag_pos.x, frag_pos.y, frag_pos.z, 1};
            hp = cmd.model_transform * hp;
            Varying varying;
            varying.pos = {hp.x, hp.y, hp.z};
            varying.normal = model3 * frag_normal;

            _ras.depth_buf[idx] = frag_t;
            _ras_frag_shader(cmd, idx, varying);
        } // pixel loop
    } // chunk loop
}

void* raytracer_thread_start(void* _arg)
{
    int thread_flag = (uint64_t)_arg;

    for(;;)
    {
        pthread_mutex_lock(_ras.mutex);

        while( (_ras.threads_busy_mask & thread_flag) == 0)
            pthread_cond_wait(_ras.cv, _ras.mutex);

        pthread_mutex_unlock(_ras.mutex);

        raytracer_thread_work(*_ras.raytracer_cmd, _ras.bvh_root);

        pthread_mutex_lock(_ras.mutex);
        _ras.threads_busy_mask &= ~thread_flag;
        pthread_mutex_unlock(_ras.mutex);
        pthread_cond_broadcast(_ras.cv);
    }
    return nullptr;
}

// this is what a user calls

void raytracer_draw(RenderCmd& cmd, BV_node* bvh_root)
{
    assert(!_ras.threads_busy_mask);
    _ras.next_pixel_idx = 0;
    _ras.raytracer_cmd = &cmd;
    _ras.bvh_root = bvh_root;

    pthread_mutex_lock(_ras.mutex);

    for(int i = 0; i < THREAD_COUNT; ++i)
        _ras.threads_busy_mask |= (uint64_t)1 << i;

    pthread_cond_broadcast(_ras.cv); // kick off the threads

    while(_ras.threads_busy_mask)
        pthread_cond_wait(_ras.cv, _ras.mutex);

    pthread_mutex_unlock(_ras.mutex);
}

void gl_draw(RenderCmd& cmd, GLuint program)
{
    GLuint vbo;
    GLuint vao;
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t bytes = cmd.vertex_count * 2 * sizeof(vec3);
    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes/2, cmd.positions);

    if(!cmd.debug)
        glBufferSubData(GL_ARRAY_BUFFER, bytes/2, bytes/2, cmd.normals);

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    if(!cmd.debug)
    {
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(bytes/2));
    }

    glUseProgram(program);
    GLint view_loc = glGetUniformLocation(program, "view");
    GLint proj_loc = glGetUniformLocation(program, "proj");
    GLint light_int_loc = glGetUniformLocation(program, "light_intensity");
    GLint light_dir_loc = glGetUniformLocation(program, "light_dir");
    GLint ambient_int_loc = glGetUniformLocation(program, "ambient_intensity");
    GLint eye_pos_loc = glGetUniformLocation(program, "eye_pos");
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, cmd.view.data);
    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, cmd.proj.data);
    glUniform3fv(light_int_loc, 1, &cmd.light_intensity.x);
    glUniform3fv(light_dir_loc, 1, &cmd.light_dir.x);
    glUniform1f(ambient_int_loc, cmd.ambient_intensity);
    glUniform3fv(eye_pos_loc, 1, &cmd.eye_pos.x);
    GLint model_loc = glGetUniformLocation(program, "model");
    GLint diffuse_color_loc = glGetUniformLocation(program, "diffuse_color");
    GLint specular_color_loc = glGetUniformLocation(program, "specular_color");
    GLint specular_exp_loc = glGetUniformLocation(program, "specular_exp");
    GLint debug_loc = glGetUniformLocation(program, "debug");
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, cmd.model_transform.data);
    glUniform3fv(diffuse_color_loc, 1, &cmd.diffuse_color.x);
    glUniform3fv(specular_color_loc, 1, &cmd.specular_color.x);
    glUniform1f(specular_exp_loc, cmd.specular_exp);
    glUniform1i(debug_loc, cmd.debug);

    if(cmd.debug)
    {
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDrawArrays(GL_LINES, 0, cmd.vertex_count);
    }
    else
    {
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDrawArrays(GL_TRIANGLES, 0, cmd.vertex_count);
    }

    // note: this is super important
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

void load_model(const char* filename, vec3*& positions, vec3*& normals, int& vertex_count)
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
    vertex_count = indices_p.size();
    assert(vertex_count % 3 == 0);
    positions = (vec3*)malloc(sizeof(vec3) * vertex_count);
    normals = (vec3*)malloc(sizeof(vec3) * vertex_count);

    for(int i = 0; i < vertex_count; ++i)
    {
        positions[i] = tmp_positions[indices_p[i]];
        normals[i] = tmp_normals[indices_n[i]];
    }
}

RenderCmd test_triangle()
{
    vec3* positions = (vec3*)malloc(3 * sizeof(vec3));
    vec3* normals = (vec3*)malloc(3 * sizeof(vec3));
    positions[0] = {-1,0,0};
    positions[1] = {1,0,0};
    positions[2] = {0,5,-1};
    vec3 e01 = positions[1] - positions[0];
    vec3 e02 = positions[2] - positions[0];
    vec3 normal = normalize(cross(e01, e02));
    normals[0] = normals[1] = normals[2] = normal;
    RenderCmd cmd;
    cmd.light_intensity = {0.2, 0.2, 0.2};
    cmd.light_dir = normalize(vec3{-0.2, 1, 1});
    cmd.ambient_intensity = 0.2f;
    cmd.positions = positions;
    cmd.normals = normals;
    cmd.vertex_count = 3;
    cmd.model_transform = identity4();
    cmd.diffuse_color = {0.5, 0.5, 0.5};
    cmd.specular_color = {0, 1, 0};
    cmd.specular_exp = 20.f;
    cmd.debug = false;
    return cmd;
}

void build_BHV_viz(BV_node* node, int current_depth, int target_depth, std::vector<vec3>& segments)
{
    if(!node->leaf && current_depth != target_depth)
    {
        for(int i = 0; i < node->child_count; ++i)
            build_BHV_viz(node->children[i], current_depth + 1, target_depth, segments);
        return;
    }
    vec3 bmin = node->bbox_min;
    vec3 bmax = node->bbox_max;

    vec3 v[8];
    // lower y face vertices, counter-clockwise
    v[0] = bmin;
    v[1] = {bmax.x, bmin.y, bmin.z};
    v[2] = {bmax.x, bmin.y, bmax.z};
    v[3] = {bmin.x, bmin.y, bmax.z};
    // higher y face
    for(int i = 4; i < 8; ++i)
    {
        v[i] = v[i - 4];
        v[i].y = bmax.y;
    }
    // 12 edges
    for(int i = 0; i < 4; ++i)
    {
        int i2 = (i+1)%4;
        // lower y face edge
        segments.push_back(v[i]);
        segments.push_back(v[i2]);
        // higher y face edge
        segments.push_back(v[i+4]);
        segments.push_back(v[i2+4]);
        // side face edge
        segments.push_back(v[i]);
        segments.push_back(v[i+4]);
    }
}

void count_BVH_nodes(BV_node* node, int& internal, int& leaf)
{
    if(node->leaf)
        leaf += 1;
    else
    {
        internal += 1;

        for(int i = 0; i < node->child_count; ++i)
            count_BVH_nodes(node->children[i], internal, leaf);
    }
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
    //SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 400, 400, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
    {
        assert(false);
    }

    ras_init();

    GLuint program = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);
    }
    bool quit = false;
    int debug_bvh_level = 0;
    bool enable_move = false;
    int render_mode = 0;
    bool use_persp = true;
    bool allow_rot = false;
    vec3 camera_pos = {0.f, 0.f, 2.f};
    float pitch = 0;
    float yaw = 0;
    Uint64 prev_counter = SDL_GetPerformanceCounter(); // on linux this is clock_gettime(CLOCK_MONOTONIC)
    vec4 rotation = quat_rot({0,1,0}, 0);

    RenderCmd cmd;
    cmd.light_intensity = {0.4, 0.4, 0.4};
    cmd.light_dir = normalize(vec3{1, 1, 0.5});
    cmd.ambient_intensity = 0.01;
    cmd.model_transform = identity4();
    cmd.diffuse_color = {0.15, 0.15, 0.15};
    cmd.specular_color = {1, 0, 0};
    cmd.specular_exp = 60;
    cmd.debug = false;

    load_model("model.obj", cmd.positions, cmd.normals, cmd.vertex_count);
    //cmd = test_triangle();
    printf("vertex count: %d (%d triangles)\n", cmd.vertex_count, cmd.vertex_count/3);

    BV_node* bvh_root;
    {
        Uint64 c1 = SDL_GetPerformanceCounter();
        bvh_root = build_BVH(cmd.positions, cmd.normals, cmd.vertex_count);
        Uint64 c2 = SDL_GetPerformanceCounter();
        double dt = (c2 - c1) / (double)SDL_GetPerformanceFrequency();
        printf("BVH build time: %f\n", dt);
        int internal = 0;
        int leaf = 0;
        count_BVH_nodes(bvh_root, internal, leaf);
        printf("BVH internal nodes: %d\n", internal);
        printf("BVH leaf     nodes: %d\n", leaf);
    }
    std::vector<vec3> segments;

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
                quit = true;
            else if(event.type == SDL_KEYDOWN)
            {
                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                case SDLK_SPACE:
                    enable_move = !enable_move;
                    SDL_SetRelativeMouseMode(enable_move ? SDL_TRUE : SDL_FALSE);
                    break;
                case SDLK_1:
                    render_mode += 1;

                    if(render_mode > 2)
                        render_mode = 0;

                    printf("switched to %s\n", render_mode == 0 ? "OpenGL" : render_mode == 1 ? "software rasterizer" : "raytracer");
                    break;
                case SDLK_2:
                    use_persp = !use_persp;
                    printf("switched to %s projection\n", use_persp ? "perspective" : "orthographic");
                    break;
                case SDLK_d:
                    debug_bvh_level += 1;

                    if(debug_bvh_level > BVH_MAX_DEPTH)
                        debug_bvh_level = 0;

                    segments.clear();

                    if(debug_bvh_level == 0) // don't display anything at level 0
                        break;
                    build_BHV_viz(bvh_root, 0, debug_bvh_level, segments);
                    break;
                case SDLK_r:
                    allow_rot = !allow_rot;
                }
            }
            else if(event.type == SDL_MOUSEMOTION && enable_move)
            {
                yaw -= event.motion.xrel / 10.f;
                float new_pitch = pitch - event.motion.yrel / 10.f;

                if(fabs(new_pitch) < 80.f)
                    pitch = new_pitch;
            }
        }
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        mat4 persp = perspective(60, (float)width/height, 0.1f, 100.f);
        mat4 ortho;
        {
            float l, r, b, t, n, f;
            extract_frustum(persp, l, r, b, t, n, f);
            ortho = orthographic(l, r, b, t, n, f);
        }
        cmd.view = lookat(camera_pos, yaw, pitch);
        cmd.proj = use_persp ? persp : ortho;
        cmd.eye_pos = camera_pos;

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;

        if(allow_rot)
            rotation = quat_mul(rotation, quat_rot({0,1,0}, 2*pi/10*dt));
        cmd.model_transform = quat_to_mat4(rotation);

        switch(render_mode)
        {
        case 0:
        {
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            gl_draw(cmd, program);
            break;
        }
        case 1:
            ras_viewport(width, height);
            ras_clear_buffers();
            ras_draw(cmd);
            ras_display();
            break;
        case 2:
        {
            ras_viewport(width, height);
            ras_clear_buffers();
            //Uint64 c1 = SDL_GetPerformanceCounter();
            raytracer_draw(cmd, bvh_root);
            //Uint64 c2 = SDL_GetPerformanceCounter();
            //double dt = (c2 - c1) / (double)SDL_GetPerformanceFrequency();
            //printf("raytracer render time: %f\n", dt);
            ras_display();
            break;
        }
        default:
            assert(false);
        }

        // bvh debug view
        RenderCmd cmd_tmp = cmd;
        cmd_tmp.positions = segments.data();
        cmd_tmp.vertex_count = segments.size();
        cmd_tmp.debug = true;
        gl_draw(cmd_tmp, program);

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
