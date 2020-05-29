#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "glad.h"
#include "main.hpp"

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

struct TracerTriangle
{
    Varying vs[3];
    vec3 edge01;
    vec3 edge12;
    vec3 edge20;
    vec3 normal;
    float area;
};

// software rasterizer / raytracer global state
struct
{
    int width;
    int height;
    float* depth_buf;
    u8* color_buf;
    TracerTriangle* triangles;
    int triangles_size;
    // opengl resources for ras_display()
    GLuint vbo;
    GLuint vao;
    GLuint program;
    GLuint texture;
} _ras;

void ras_init()
{
    _ras.depth_buf = nullptr;
    _ras.color_buf = nullptr;
    _ras.width = 0;
    _ras.height = 0;
    _ras.triangles = nullptr;
    _ras.triangles_size = 0;

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
}

void ras_viewport(int width, int height)
{
    if(width == _ras.width && height == _ras.height)
        return;
    _ras.depth_buf = (float*)malloc(width * height * sizeof(float));
    _ras.color_buf = (u8*)malloc(width * height * 3);
    _ras.width = width;
    _ras.height = height;
    glBindTexture(GL_TEXTURE_2D, _ras.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
}

void ras_display()
{
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, _ras.width, _ras.height);
    glBindTexture(GL_TEXTURE_2D, _ras.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _ras.width, _ras.height, GL_RGB, GL_UNSIGNED_BYTE, _ras.color_buf);
    glBindVertexArray(_ras.vao);
    glUseProgram(_ras.program);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);
}

void ras_clear_buffers()
{
    for(int i = 0; i < _ras.width * _ras.height; ++i)
        _ras.depth_buf[i] = FLT_MAX;
    memset(_ras.color_buf, 0, _ras.width * _ras.height * 3);
}

void _ras_draw1(RenderCmd& cmd, vec4* coords, Varying* varyings);
void _ras_draw2(RenderCmd& cmd, vec4* coords, Varying* varyings);

void ras_draw(RenderCmd& cmd)
{
    // something like a fixed vertex shader
    mat4 proj_view = mul(cmd.proj, cmd.view);
    mat3 model3 = mat4_to_mat3(cmd.model_transform);

    for(int base = 0; base < cmd.vertex_count; base += 3)
    {
        vec4 coords[3];
        Varying varyings[3];

        for(int i = 0; i < 3; ++i)
        {
            vec3 vert_pos = cmd.positions[base + i];
            coords[i] = {vert_pos.x, vert_pos.y, vert_pos.z, 1};
            coords[i] = mul(cmd.model_transform, coords[i]);
            varyings[i].pos = {coords[i].x, coords[i].y, coords[i].z};
            coords[i] = mul(proj_view, coords[i]);
            varyings[i].normal = mul(model3, cmd.normals[base + i]);
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

float max(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

float min(float lhs, float rhs)
{
    return lhs < rhs ? lhs : rhs;
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
    _ras.color_buf[idx*3 + 0] = 255.f * color.x + 0.5f;
    _ras.color_buf[idx*3 + 1] = 255.f * color.y + 0.5f;
    _ras.color_buf[idx*3 + 2] = 255.f * color.z + 0.5f;
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

        // viewport transform
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
        min_x = coords[i].x < min_x ? coords[i].x : min_x;
        max_x = coords[i].x > max_x ? coords[i].x : max_x;
        min_y = coords[i].y < min_y ? coords[i].y : min_y;
        max_y = coords[i].y > max_y ? coords[i].y : max_y;
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

void raytracer_draw(RenderCmd& cmd)
{
    // exit if projection is orthographic
    // todo
    if(!cmd.proj.data[14])
        return;

    int tri_count = cmd.vertex_count / 3;

    if(tri_count > _ras.triangles_size)
    {
        free(_ras.triangles);
        _ras.triangles_size = tri_count;
        _ras.triangles = (TracerTriangle*)malloc(sizeof(TracerTriangle) * tri_count);
    }

    // preprocess vertex data
    mat3 model3 = mat4_to_mat3(cmd.model_transform);

    for(int tid = 0; tid < tri_count; ++tid)
    {
        TracerTriangle tri;

        for(int i = 0; i < 3; ++i)
        {
            vec3 p = cmd.positions[tid*3 + i];
            vec4 hp = {p.x, p.y, p.z, 1};
            hp = mul(cmd.model_transform, hp);
            tri.vs[i].pos = {hp.x, hp.y, hp.z};
            tri.vs[i].normal = mul(model3, cmd.normals[tid*3 + i]);
        }
        tri.edge01 = tri.vs[1].pos - tri.vs[0].pos;
        tri.edge12 = tri.vs[2].pos - tri.vs[1].pos;
        tri.edge20 = tri.vs[0].pos - tri.vs[2].pos;
        tri.normal = cross(tri.edge01, tri.edge12);
        tri.area = length(tri.normal);
        tri.normal = (1 / tri.area) * tri.normal; // normalize
        _ras.triangles[tid] = tri;
    }

    float left, right, bot, top, near, far;
    extract_frustum(cmd.proj, left, right, bot, top, near, far);
    int width = _ras.width;
    int height = _ras.height;
    mat3 eye_basis = transpose(mat4_to_mat3(cmd.view)); // change of basis matrix
    vec3 eye_pos = -1 * mul( eye_basis, vec3{cmd.view.data[3], cmd.view.data[7], cmd.view.data[11]} );
    vec3 eye_dir = {-eye_basis.data[2], -eye_basis.data[5], -eye_basis.data[8]};

    for(int idx = 0; idx < width * height; ++idx)
    {
        int pix_x = idx % width;
        int pix_y = idx / width;
        float x = ((right - left)/width) * (pix_x + 0.5f) + left;
        float y = ((top - bot)/height) * (pix_y + 0.5f) + bot;
        float z = -near;
        vec3 ray_dir = normalize( mul(eye_basis, vec3{x, y, z}) );

        for(int tid = 0; tid < tri_count; ++tid)
        {
            TracerTriangle tri = _ras.triangles[tid];
            vec3 normal = tri.normal;

            // face culling
            if(dot(normal, -ray_dir) < 0.f)
                continue;

            float t = (dot(tri.normal, tri.vs[0].pos) - dot(normal, eye_pos)) / dot(normal, ray_dir);

            // depth test
            if(t > _ras.depth_buf[idx])
                continue;

            float dist_z = dot(eye_dir, t*ray_dir);

            // far / near plane cull
            if(dist_z < near || dist_z > far)
                continue;

            vec3 p = eye_pos + t*ray_dir;
            float b0 = dot(normal, cross(tri.edge12, p - tri.vs[1].pos)) / tri.area;
            float b1 = dot(normal, cross(tri.edge20, p - tri.vs[2].pos)) / tri.area;
            float b2 = dot(normal, cross(tri.edge01, p - tri.vs[0].pos)) / tri.area;

            // point on a plane is not in a triangle
            if(b0 < 0 || b1 < 0 || b2 < 0)
                continue;
            _ras.depth_buf[idx] = t;
            _ras_frag_shader(cmd, idx, (b0 * tri.vs[0]) + (b1 * tri.vs[1]) + (b2 * tri.vs[2]) );
        }
    }
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
    glBufferSubData(GL_ARRAY_BUFFER, bytes/2, bytes/2, cmd.normals);

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(bytes/2));

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
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, cmd.model_transform.data);
    glUniform3fv(diffuse_color_loc, 1, &cmd.diffuse_color.x);
    glUniform3fv(specular_color_loc, 1, &cmd.specular_color.x);
    glUniform1f(specular_exp_loc, cmd.specular_exp);
    glDrawArrays(GL_TRIANGLES, 0, cmd.vertex_count);
    // note: this is super important
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

void load(vec3*& out_positions, vec3*& out_normals, int& out_vertex_count);

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
    return cmd;
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        assert(false);
    }
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 100, 100, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    //SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 800, 800, SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
    {
        assert(false);
    }

    ras_init();

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

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
    bool enable_move = false;
    int render_mode = 0;
    bool use_persp = true;
    vec3 camera_pos = {0.f, 0.f, 1.f};
    float pitch = 0;
    float yaw = 0;

    RenderCmd cmd;
    cmd.light_intensity = {0.4, 0.4, 0.4};
    cmd.light_dir = normalize(vec3{1, 1, 1});
    cmd.ambient_intensity = 0.05;
    load(cmd.positions, cmd.normals, cmd.vertex_count);
    cmd.model_transform = identity4();
    cmd.diffuse_color = {0.15, 0.15, 0.15};
    cmd.specular_color = {1, 0, 0};
    cmd.specular_exp = 60;

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

                    if(render_mode > 1) // todo: 2
                        render_mode = 0;

                    printf("switched to %s\n", render_mode == 0 ? "OpenGL" : render_mode == 1 ? "software rasterizer" : "raytracer");
                    break;
                case SDLK_2:
                    use_persp = !use_persp;
                    printf("switched to %s projection\n", use_persp ? "perspective" : "orthographic");
                    break;
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
        mat4 persp = perspective(90, (float)width/height, 0.1f, 100.f);
        mat4 ortho;
        {
            float l, r, b, t, n, f;
            extract_frustum(persp, l, r, b, t, n, f);
            ortho = orthographic(l, r, b, t, n, f);
        }
        cmd.view = lookat(camera_pos, yaw, pitch);
        cmd.proj = use_persp ? persp : ortho;
        cmd.eye_pos = camera_pos;

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
            ras_viewport(width, height);
            ras_clear_buffers();
            raytracer_draw(cmd);
            ras_display();
            break;
        default:
            assert(false);
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
