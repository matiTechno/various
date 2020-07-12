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
    vec3 eye_pos;
    mat4 view;
    mat4 proj;
};

void ctrl_init(Controller& ctrl, float win_width, float win_height)
{
    ctrl.pos = vec3{50,100,0};
    ctrl.forward = vec3{0,0,-1};
    ctrl.vel = {};
    ctrl.radius = 1.2;
    ctrl.max_dist = 150;
    ctrl.min_dist = ctrl.radius / 2;
    ctrl.target_dist = 35;
    ctrl.current_dist = ctrl.target_dist;
    ctrl.dist_vel = {};
    ctrl.pitch = deg_to_rad(35);
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
}

void ctrl_process_event(Controller& ctrl, SDL_Event& e)
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
            SDL_SetRelativeMouseMode((SDL_bool)down);
            break;
        case SDL_BUTTON_RIGHT:
            ctrl.rmb_down = down;
            SDL_SetRelativeMouseMode((SDL_bool)down);
            break;
        case SDL_BUTTON_MIDDLE:
            ctrl.mmb_down = down;
            SDL_SetRelativeMouseMode((SDL_bool)down);
            break;
        }
        break;
    }
    case SDL_MOUSEWHEEL:
    {
        float base = 1.4;
        float scale = e.wheel.y < 0 ? powf(base, -e.wheel.y) : (1 / powf(base, e.wheel.y));
        ctrl.target_dist = max(ctrl.min_dist, min(ctrl.max_dist, ctrl.target_dist * scale));
        ctrl.dist_vel = (ctrl.target_dist - ctrl.current_dist) / DIST_INTERP_TIME;
        break;
    }
    case SDL_MOUSEMOTION:
    {
        if(!ctrl.lmb_down && !ctrl.rmb_down && !ctrl.mmb_down)
            break;
        float dx = 2*pi * (float)e.motion.xrel / ctrl.win_size.x;
        float dy = 2*pi * (float)e.motion.yrel / ctrl.win_size.y;
        ctrl.pitch += dy;
        ctrl.pitch = min(ctrl.pitch, deg_to_rad(80));
        ctrl.pitch = max(ctrl.pitch, deg_to_rad(-80));
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
            ctrl.jump_action = down;
            break;
        }
        break;
    }
    }
}

#define CIRP_EPS (pi/60)
#define RAY_EPS 0.3

void ctrl_resolve_events(Controller& ctrl, float dt, Mesh& level)
{
    // character update

    bool turn_mode = !ctrl.rmb_down && !ctrl.mmb_down;
    float turn_angle = 0;

    if(turn_mode)
    {
        float turn_dir = 0;

        if(ctrl.a_down)
            turn_dir += 1;
        if(ctrl.d_down)
            turn_dir += -1;

        turn_angle = turn_dir * 2 * pi / 2.5 * dt;
        ctrl.forward = transform3(rotate_y(turn_angle), ctrl.forward);

        if(ctrl.lmb_down)
            ctrl.yaw -= turn_angle;
    }
    else
    {
        ctrl.forward = transform3(rotate_y(ctrl.yaw), ctrl.forward);
        ctrl.yaw = 0;
    }

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
    float max_offset_y = PLANE_OFFSET / cosf(deg_to_rad(60));
    STI sti = intersect_level(ctrl.radius, ctrl.pos, ctrl.pos + vec3{0,-max_offset_y,0}, level);

    if(!sti.valid)
    {
        if(!ctrl.vel.x && !ctrl.vel.z)
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

    vec3 eye_dir_xz = transform3(rotate_y(ctrl.yaw), -ctrl.forward);
    vec3 eye_right = cross(vec3{0,1,0}, eye_dir_xz);
    vec3 eye_dir;
    float hit_dist = FLT_MAX;

    if(ctrl.pitch < 0)
    {
        CirPoint cirp;
        cirp.b1 = eye_dir_xz;
        cirp.b2 = vec3{0,1,0};
        cirp.center = ctrl.pos;
        cirp.R = ctrl.current_dist;
        float t_cirp = intersect_level(cirp, level);
        bool eye_hit = t_cirp + CIRP_EPS >= ctrl.pitch;
        vec3 eye_pos = eye_hit ? cirp.get(t_cirp + CIRP_EPS) : cirp.get(ctrl.pitch);
        hit_dist = length(eye_pos - ctrl.pos);
        eye_dir = normalize(eye_pos - ctrl.pos);
        float t_ray = intersect_level(Ray{ctrl.pos, eye_dir}, level);

        if(t_ray > 0 && t_ray - RAY_EPS < hit_dist)
            hit_dist = max(ctrl.min_dist, t_ray - RAY_EPS);
    }
    else
    {
        eye_dir = transform3(rotate_axis(eye_right, -ctrl.pitch), eye_dir_xz);
        float t = intersect_level(Ray{ctrl.pos, eye_dir}, level);

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

    ctrl.eye_pos = ctrl.pos + ctrl.current_dist * eye_dir;
    vec3 view_dir = transform3(rotate_axis(eye_right, -ctrl.pitch), eye_dir_xz);
    ctrl.view = lookat(ctrl.eye_pos, -view_dir);
    ctrl.proj = perspective(60, ctrl.win_size.x / ctrl.win_size.y, 0.1, 1000);
    ctrl.jump_action = false;
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

    Controller ctrl;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        ctrl_init(ctrl, width, height);
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
    }

    Uint64 prev_counter = SDL_GetPerformanceCounter();
    bool quit = false;

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                quit = true;
            ctrl_process_event(ctrl, event);
        }

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        dt = min(dt, 0.030); // debugging
        prev_counter = current_counter;

        ctrl_resolve_events(ctrl, dt, meshes[0]);

        objects[1].pos = ctrl.pos;
        objects[2].pos = objects[1].pos + 2 * ctrl.forward;

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);

        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog);
        {
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

            glUniformMatrix4fv(view_loc, 1, GL_TRUE, ctrl.view.data);
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, ctrl.proj.data);
            glUniform3fv(light_int_loc, 1, &light_intensity.x);
            glUniform3fv(light_dir_loc, 1, &light_dir.x);
            glUniform1f(ambient_int_loc, 0.01);
            glUniform3fv(eye_pos_loc, 1, &ctrl.eye_pos.x);
            glUniform3fv(specular_color_loc, 1, &specular_color.x);
            glUniform1f(specular_exp_loc, 50);
        }

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
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
