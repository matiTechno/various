#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "../main.hpp"

const char* _src_vert = R"(
#version 330
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// in world space
out vec3 frag_pos;
out vec3 frag_normal;
out vec2 frag_uv;

void main()
{
    vec4 pos_w = model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = mat3(model) * normal;
    frag_uv = uv;
}
)";

const char* _src_frag = R"(
#version 330
uniform vec3 light_intensity = vec3(1,1,1);
uniform float ambient_intensity = 0.1;
uniform vec3 specular_color = vec3(0.1);
uniform float specular_exp = 20;
uniform vec3 eye_pos;
uniform vec3 light_dir;
uniform sampler2D sampler0;

in vec3 frag_pos;
in vec3 frag_normal;
in vec2 frag_uv;
out vec3 out_color;

void main()
{
    vec3 diffuse_color = texture(sampler0, frag_uv).rgb;
    vec3 L = normalize(light_dir);
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

const char* _src_vert2 = R"(
#version 330
layout(location = 0) in vec3 pos;
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
void main()
{
    gl_Position = proj * view * model * vec4(pos,1);
}
)";

const char* _src_vert3 = R"(
#version 330
layout(location = 0) in vec3 pos;
layout(location = 2) in vec2 uv;
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
out vec4 frag_pos_clip;
out vec4 clip2;
void main()
{
    mat4 tr = mat4(1);
    tr[3][2] = -0.01;
    frag_pos_clip = proj * view * model * vec4(pos,1);
    clip2 = proj * tr * view * model * vec4(pos, 1);
    gl_Position = vec4(2*uv - 1, 0, 1);
}
)";

const char* _src_frag3 = R"(
#version 330
uniform vec2 win_start;
uniform vec2 win_size;
uniform vec2 cursor_pos_win;
uniform float radius;
uniform sampler2D sampler0;
in vec4 frag_pos_clip;
in vec4 clip2;
out vec4 out_color;

void main()
{
    vec3 pos_ndc = frag_pos_clip.xyz / frag_pos_clip.w;
    vec2 depth_uv = (pos_ndc.xy + 1) / 2;
    float dst_depth = texture(sampler0, depth_uv).r;
    float src_depth = (pos_ndc.z + 1) / 2;
    float depth2 = (clip2.z / clip2.w + 1) / 2;
    float bias = depth2 - src_depth;

    if(src_depth - bias > dst_depth)
        discard;

    vec2 pos_win = vec2(1,-1) * win_size/2 * pos_ndc.xy + win_start + win_size/2;
    float d = length(pos_win - cursor_pos_win);
    float alpha = smoothstep(0, radius, radius - d);
    out_color = vec4(1,0,0,alpha);
}
)";

struct Ray
{
    vec3 pos;
    vec3 dir;
};

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
    float right;
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
    float right = !nav.ortho ? nav.right : (nav.right / nav.near) * length(nav.center - nav.eye_pos);
    float top = right / aspect;

    if(nav.ortho)
        nav.proj = orthographic(-right, right, -top, top, nav.near, nav.far);
    else
        nav.proj = frustum(-right, right, -top, top, nav.near, nav.far);
}

void nav_init(Nav& nav, vec3 eye_pos, float win_width, float win_height, float fov_hori, float near, float far)
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
    nav.right = tanf(fov_hori / 2) * near;
    nav.near = near;
    nav.far = far;
    nav.ortho = false;
    nav.aligned = false;
    rebuild_view_matrix(nav);
    rebuild_proj_matrix(nav);
}

Ray nav_get_cursor_ray(Nav& nav, vec2 cursor_win)
{
    float right = !nav.ortho ? nav.right : (nav.right / nav.near) * length(nav.center - nav.eye_pos);
    float top = right / (nav.win_size.x / nav.win_size.y);
    float x = (2*right/nav.win_size.x) * (cursor_win.x + 0.5) - right;
    float y = (-2*top/nav.win_size.y) * (cursor_win.y + 0.5) + top;
    float z = -nav.near;
    mat4 world_f_view = invert_coord_change(nav.view);
    Ray ray;

    if(nav.ortho)
    {
        ray.pos = to_vec3(world_f_view * vec4{x,y,z,1});
        ray.dir = normalize(nav.center - nav.eye_pos);
    }
    else
    {
        ray.pos = nav.eye_pos;
        ray.dir = normalize(to_vec3( world_f_view * vec4{x,y,z,0} ));
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
            mat3 rot = rotate_y(dx) * rotate_axis(nav.eye_x, dy);
            nav.eye_pos = to_vec3(translate(nav.center) * to_mat4(rot) * translate(-nav.center) * to_point4(nav.eye_pos));
            nav.eye_x = rot * nav.eye_x;
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

struct Vertex
{
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct Mesh
{
    Vertex* verts;
    mat4 model_tf;
    GLuint vbo;
    GLuint vao;
    int vert_count;
};

enum ObjMode
{
    OBJ_OTHER,
    OBJ_POS,
    OBJ_NORM,
    OBJ_UV,
    OBJ_FACE,
};

Mesh load(const char* filename)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    char buf[256];
    std::vector<vec3> positions;
    std::vector<vec3> normals;
    std::vector<vec2> uvs;
    std::vector<Vertex> verts;

    for(;;)
    {
        int r = fscanf(file, "%s", buf);

        if(r == EOF)
            break;
        assert(r == 1);

        ObjMode mode = OBJ_OTHER;

        if(strcmp(buf, "v") == 0)
            mode = OBJ_POS;
        else if(strcmp(buf, "vn") == 0)
            mode = OBJ_NORM;
        else if(strcmp(buf, "vt") == 0)
            mode = OBJ_UV;
        else if(strcmp(buf, "f") == 0)
            mode = OBJ_FACE;

        switch(mode)
        {
        case OBJ_POS:
        case OBJ_NORM:
        {
            vec3 v;
            r = fscanf(file, "%f %f %f", &v.x, &v.y, &v.z);
            assert(r == 3);

            if(mode == OBJ_POS)
                positions.push_back(v);
            else
                normals.push_back(v);
            break;
        }
        case OBJ_UV:
        {
            vec2 uv;
            r = fscanf(file, "%f %f", &uv.x, &uv.y);
            assert(r == 2);
            uvs.push_back(uv);
            break;
        }
        case OBJ_FACE:
        {
            for(int i = 0; i < 3; ++i)
            {
                int pos_id, uv_id, norm_id;
                r = fscanf(file, "%d/%d/%d", &pos_id, &uv_id, &norm_id);
                assert(r == 3);
                Vertex vert;
                vert.pos = positions[pos_id - 1];
                vert.normal = normals[norm_id - 1];
                vert.uv = uvs[uv_id - 1];
                verts.push_back(vert);
            }
            break;
        }
        case OBJ_OTHER:
            break;
        }
    }
    Mesh mesh;
    mesh.model_tf = identity4();
    mesh.vert_count = verts.size();
    int buf_size = mesh.vert_count * sizeof(Vertex);
    mesh.verts = (Vertex*)malloc(buf_size);
    memcpy(mesh.verts, verts.data(), buf_size);
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, buf_size, mesh.verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    return mesh;
}

GLuint create_program(const char* src_vert, const char* src_frag)
{
    GLuint program = glCreateProgram();
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vert_shader, 1, &src_vert, nullptr);
    glCompileShader(vert_shader);
    glAttachShader(program, vert_shader);

    if(src_frag)
    {
        glShaderSource(frag_shader, 1, &src_frag, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(program, frag_shader);
    }
    glLinkProgram(program);
    return program;
}

struct MeshPaint
{
    GLuint vao;
    GLuint vbo;
};

struct VertexPaint
{
    vec3 pos;
    vec2 uv;
};

float signed_area(vec2 v1, vec2 v2)
{
    return {v1.x * v2.y - v1.y * v2.x};
}

#define TEX_SIZE 2048

// this is to avoid ceretain seam artifacts

MeshPaint gen_MeshPaint(Mesh src_mesh)
{
    std::vector<VertexPaint> verts;
    verts.reserve(src_mesh.vert_count);

    for(int base = 0; base < src_mesh.vert_count; base += 3)
    {
        Vertex* face = src_mesh.verts + base;
        vec2 edge01 = face[1].uv - face[0].uv;
        vec2 edge12 = face[2].uv - face[1].uv;
        vec2 edge20 = face[0].uv - face[2].uv;
        float area = signed_area(edge01, -edge20);

        for(int i = 0; i < 3; ++i)
        {
            vec2 uv0 = face[i].uv;
            vec2 uv1 = face[(i+1)%3].uv;
            vec2 uv2 = face[(i+2)%3].uv;
            // todo: better algorithm for extrusion?
            vec2 d1 = normalize(uv0 - uv1);
            vec2 d2 = normalize(uv0 - uv2);
            vec2 uv = uv0 + 2.0/TEX_SIZE * (d1 + d2);
            float b0 = signed_area(edge12, uv - face[1].uv) / area;
            float b1 = signed_area(edge20, uv - face[2].uv) / area;
            float b2 = signed_area(edge01, uv - face[0].uv) / area;
            vec3 pos = b0 * face[0].pos + b1 * face[1].pos + b2 * face[2].pos;
            verts.push_back({pos, uv});
        }
    }
    MeshPaint mesh;
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(VertexPaint), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(2); // 2 to keep compatible with basic Mesh
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPaint), nullptr);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexPaint), (void*)offsetof(VertexPaint, uv));
    return mesh;
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

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    GLuint program = create_program(_src_vert, _src_frag);
    GLuint program_depth = create_program(_src_vert2, nullptr);
    GLuint program_paint = create_program(_src_vert3, _src_frag3);

    GLuint tex_depth;
    glGenTextures(1, &tex_depth);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    GLuint fbo_depth;
    glGenFramebuffers(1, &fbo_depth);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_depth);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_depth, 0);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEX_SIZE, TEX_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // this is to support transparency in painting (MeshPaint triangles are overlapping and a test is needed to not color the same pixel twice
    GLuint tex_uv_depth;
    glGenTextures(1, &tex_uv_depth);
    glBindTexture(GL_TEXTURE_2D, tex_uv_depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, TEX_SIZE, TEX_SIZE, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glBindTexture(GL_TEXTURE_2D, tex_uv_depth);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_uv_depth, 0);
    glClearColor(0.01,0.01,0.01,0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0,0,0,0);

    Mesh mesh = load("suzanne.obj");
    MeshPaint mesh_paint = gen_MeshPaint(mesh);
    Nav nav;
    nav_init(nav, vec3{0,0,2}, width, height, to_radians(90), 0.1, 100);
    bool quit = false;
    bool lmb_down = false;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        bool cursor_moved = false;
        vec2 cursor_pos_win;
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            nav_process_event(nav, event);

            switch(event.type)
            {
            case SDL_QUIT:
            {
                quit = true;
                break;
            }
            case SDL_KEYDOWN:
            {
                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                }
                break;
            }
            case SDL_MOUSEBUTTONDOWN:
            case SDL_MOUSEBUTTONUP:
            {
                bool down = event.type == SDL_MOUSEBUTTONDOWN;

                if(event.button.button == SDL_BUTTON_LEFT)
                {
                    lmb_down = down;
                    cursor_moved = true;
                }
                break;
            }
            case SDL_MOUSEMOTION:
                cursor_moved = true;
                cursor_pos_win = vec2{(float)event.motion.x, (float)event.motion.y};
                break;
            }
        }

        int new_width, new_height;
        SDL_GetWindowSize(window, &new_width, &new_height);

        if(new_width != width || new_height != height)
        {
            width = new_width;
            height = new_height;
            glBindTexture(GL_TEXTURE_2D, tex_depth);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);
        }

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        (void)dt;
        prev_counter = current_counter;
        glEnable(GL_DEPTH_TEST);

        if(lmb_down && cursor_moved)
        {
            // depth pass

            glEnable(GL_CULL_FACE);
            glDisable(GL_BLEND);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_depth);
            glViewport(0, 0, width, height);
            glClear(GL_DEPTH_BUFFER_BIT);
            glUseProgram(program_depth);
            glUniformMatrix4fv(glGetUniformLocation(program_depth, "proj"), 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(glGetUniformLocation(program_depth, "view"), 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(glGetUniformLocation(program_depth, "model"), 1, GL_TRUE, mesh.model_tf.data);
            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh.vert_count);

            // paint pass

            glDisable(GL_CULL_FACE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, TEX_SIZE, TEX_SIZE);
            glClear(GL_DEPTH_BUFFER_BIT);
            glBindTexture(GL_TEXTURE_2D, tex_depth);
            glUseProgram(program_paint);
            glUniformMatrix4fv(glGetUniformLocation(program_paint, "proj"), 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(glGetUniformLocation(program_paint, "view"), 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(glGetUniformLocation(program_paint, "model"), 1, GL_TRUE, mesh.model_tf.data);
            glUniform2f(glGetUniformLocation(program_paint, "win_start"), 0, 0);
            glUniform2f(glGetUniformLocation(program_paint, "win_size"), width, height);
            glUniform2fv(glGetUniformLocation(program_paint, "cursor_pos_win"), 1, &nav.cursor_win.x);
            glUniform1f(glGetUniformLocation(program_paint, "radius"), 20);
            glBindVertexArray(mesh_paint.vao);
            //glBindVertexArray(mesh.vao); // uncomment to see the seam artifacts
            glDrawArrays(GL_TRIANGLES, 0, mesh.vert_count);
        }

        // display pass

        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, nav.proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, nav.view.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, mesh.model_tf.data);
        glUniform3fv(glGetUniformLocation(program, "eye_pos"), 1, &nav.eye_pos.x);
        vec3 light_dir = normalize(nav.eye_pos - nav.center);
        glUniform3fv(glGetUniformLocation(program, "light_dir"), 1, &light_dir.x);
        glBindVertexArray(mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, mesh.vert_count);

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
