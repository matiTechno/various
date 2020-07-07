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
    vec3 inormal;

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
        inormal = normal;
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
        inormal = normalize(S0 + t*V - tangent);
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
        inormal = normalize(S0 + t*V);
    }

    STI sti;
    sti.valid = false;

    if(min_t <= 1)
    {
        sti.point = pos_start + min_t * (pos_end - pos_start);
        sti.normal = inormal;
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

#define PLANE_OFFSET (0.01)

vec3 get_offset_pos(float radius, vec3 pos, Mesh& level)
{
    for(;;)
    {
        bool done = true;

        for(int base = 0; base < level.vertex_count; base += 3)
        {
            vec3 p = get_nearest_triangle_point(pos, level.vertices + base);

            if(length(p - pos) >= radius + PLANE_OFFSET)
                continue;

            done = false;
            vec3 normal = normalize(pos - p);
            pos = p + (radius + PLANE_OFFSET * 1.1) * normal;
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
    return get_offset_pos(radius, sti2.point, level);
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

    std::vector<Mesh> meshes;
    meshes.push_back(load_mesh("level.obj"));
    meshes.push_back(load_mesh("sphere.obj"));
    std::vector<Object> objects;
    objects.reserve(100);
    {
        Object obj;
        obj.mesh = &meshes[0];
        obj.pos = {};
        obj.rot = identity4();
        obj.scale = vec3{1,1,1};
        objects.push_back(obj);

        obj.mesh = &meshes[1];
        obj.pos.x += 50;
        obj.pos.y += 100;
        objects.push_back(obj);

        obj.scale = 0.5 * obj.scale;
        objects.push_back(obj);
    }

    Mesh& level = *objects[0].mesh;
    Object& ball = objects[1];
    Object& dir_ball = objects[2];

    Nav nav;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        nav_init(nav, vec3{200,100,0}, width, height, 60, 0.1, 1000);
    }

    Uint64 prev_counter = SDL_GetPerformanceCounter();
    bool quit = false;

    bool w_down = false;
    bool s_down = false;
    bool a_down = false;
    bool d_down = false;
    vec3 forward = {0,0,1};
    vec3 vel = {};
    float radius = ball.scale.x;

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
                quit = true;
            nav_process_event(nav, event);

            if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP)
            {
                bool down = event.type == SDL_KEYDOWN;

                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                case SDLK_w:
                    w_down = down;
                    break;
                case SDLK_s:
                    s_down = down;
                    break;
                case SDLK_a:
                    a_down = down;
                    break;
                case SDLK_d:
                    d_down = down;
                    break;
                }
            }
        }

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;
        float ang_vel= 2*pi/2;

        if(a_down)
            forward = transform3(rotate_y(ang_vel*dt), forward);
        if(d_down)
            forward = transform3(rotate_y(-ang_vel*dt), forward);

        bool apply_gravity = false;
        {
            STI sti = intersect_level(radius, ball.pos, ball.pos + vec3{0,-3*PLANE_OFFSET,0}, level);

            if(!sti.valid)
                apply_gravity = true;
            else if( dot(sti.normal, vec3{0,1,0}) < cosf(deg_to_rad(60)) ) // slide down on steep slopes, steeper than the given angle
                apply_gravity = true;
        }

        if(apply_gravity)
        {
            vec3 acc = {0, -9.8 * 10, 0};
            vec3 new_pos = ball.pos + (dt * vel) + (0.5 * dt * dt * acc);
            vel = vel + dt * acc;
            ball.pos = slide(radius, ball.pos, new_pos, level);
        }
        else
        {
            vec3 init_pos = ball.pos;
            vec3 dir = {};

            if(w_down)
                dir = dir + forward;
            if(s_down)
                dir = dir - forward;

            float forward_vel = 35;
            vec3 new_pos = ball.pos + (forward_vel * dt * dir);
            ball.pos = slide(radius, ball.pos, new_pos, level);

            /*
            // snap to ground
            float snap_dist = 0.5 * radius;

            STI sti = intersect_level(radius, ball.pos, ball.pos + vec3{0,-snap_dist,0}, level);

            if(sti.valid)
                ball.pos = slide(radius, ball.pos, ball.pos - (snap_dist * sti.normal), level);
                */

            vel = (1/dt) * (ball.pos - init_pos);
        }

        dir_ball.pos = ball.pos + 2 * forward;
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);

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

        for(int i = 0; i < (int)objects.size(); ++i)
        {
            Object& obj = objects[i];
            GLint diffuse_color_loc = glGetUniformLocation(prog, "diffuse_color");
            GLint model_loc = glGetUniformLocation(prog, "model");
            vec3 diffuse_color = vec3{0.3, 0.3, 0.3};

            if(&obj == &ball)
                diffuse_color = vec3{1,0,0};
            else if(&obj == &dir_ball)
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
