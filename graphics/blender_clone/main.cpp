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
    frag_normal = mat3(model) * normal; // inverse transpose
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
    vec3 position;
    mat4 rotation;
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

vec3 transform3(mat4 m, vec3 v)
{
    vec4 h = {v.x, v.y, v.z, 1};
    h = m * h;
    return {h.x, h.y, h.z};
}

// does not handle a parallel case

float plane_ray_test(vec3 plane_normal, vec3 plane_pos, vec3 ray_start, vec3 ray_dir)
{
    float t = dot(plane_normal, plane_pos - ray_start) / dot(plane_normal, ray_dir);
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
    rebuild_view_matrix(nav);
    rebuild_proj_matrix(nav);
}

void nav_get_cursor_ray(Nav& nav, vec2 cursor_win, vec3& ray_start, vec3& ray_dir)
{
    float top = !nav.ortho ? nav.top : (nav.top / nav.near) * length(nav.center - nav.eye_pos);
    float right = top * (nav.win_size.x / nav.win_size.y);
    float x = (2*right/nav.win_size.x) * (cursor_win.x + 0.5) - right;
    float y = (-2*top/nav.win_size.y) * (cursor_win.y + 0.5) + top;
    float z = -nav.near;
    mat4 world_f_view = invert_coord_change(nav.view);

    if(nav.ortho)
    {
        ray_start = transform3(world_f_view, vec3{x,y,z});
        ray_dir = normalize(nav.center - nav.eye_pos);
    }
    else
    {
        mat3 eye_basis = mat4_to_mat3(world_f_view);
        ray_start = nav.eye_pos;
        ray_dir = normalize( eye_basis * vec3{x,y,z} );
    }
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
                vec3 ray_start, ray_dir;
                nav_get_cursor_ray(nav, cursors[i], ray_start, ray_dir);
                float t = plane_ray_test(normal, nav.center, ray_start, ray_dir);
                assert(t > 0);
                points[i] = ray_start + t*ray_dir;
            }
            vec3 d = points[0] - points[1];
            nav.eye_pos = nav.eye_pos + d;
            nav.center = nav.center + d;
            rebuild_view_matrix(nav);
        }
        else if(nav.mmb_down)
        {
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
        if(e.key.keysym.sym == SDLK_LSHIFT)
            nav.shift_down = true;
        else if(e.key.keysym.sym == SDLK_p)
        {
            nav.ortho = !nav.ortho;
            rebuild_proj_matrix(nav);
        }
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

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    GLuint prog = create_program(_vert, _frag);
    GLuint prog_solid = create_program(_vert_solid, _frag_solid);

    std::vector<Mesh> meshes;
    meshes.push_back(load_mesh("cube.obj"));
    meshes.push_back(load_mesh("../model.obj"));
    std::vector<Object> objects;
    {
        Object o;
        o.mesh = &meshes[1];
        o.position = {0,0,0};
        o.rotation = rotate_x(0);
        o.scale = {1,1,1};
        objects.push_back(o);
    }

    Nav nav;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        nav_init(nav, vec3{0.5, 0.5, 2.f}, width, height, 60, 0.1, 1000);
    }

    bool quit = false;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            nav_process_event(nav, event);

            if(event.type == SDL_QUIT)
                quit = true;
            else if(event.type == SDL_KEYDOWN)
            {
                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
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

        // objects

        glUseProgram(prog);
        {
            vec3 light_intensity = {1,1,1};
            vec3 light_dir = normalize(nav.eye_pos - nav.center);
            vec3 diffuse_color = {0.2, 0.2, 0.2};
            vec3 specular_color = {0.8, 0.8, 0.8};

            GLint proj_loc = glGetUniformLocation(prog, "proj");
            GLint view_loc = glGetUniformLocation(prog, "view");
            GLint light_int_loc = glGetUniformLocation(prog, "light_intensity");
            GLint light_dir_loc = glGetUniformLocation(prog, "light_dir");
            GLint ambient_int_loc = glGetUniformLocation(prog, "ambient_intensity");
            GLint eye_pos_loc = glGetUniformLocation(prog, "eye_pos");
            GLint diffuse_color_loc = glGetUniformLocation(prog, "diffuse_color");
            GLint specular_color_loc = glGetUniformLocation(prog, "specular_color");
            GLint specular_exp_loc = glGetUniformLocation(prog, "specular_exp");

            glUniformMatrix4fv(view_loc, 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, nav.proj.data);
            glUniform3fv(light_int_loc, 1, &light_intensity.x);
            glUniform3fv(light_dir_loc, 1, &light_dir.x);
            glUniform1f(ambient_int_loc, 0.01);
            glUniform3fv(eye_pos_loc, 1, &nav.eye_pos.x);
            glUniform3fv(diffuse_color_loc, 1, &diffuse_color.x);
            glUniform3fv(specular_color_loc, 1, &specular_color.x);
            glUniform1f(specular_exp_loc, 50);
        }

        for(Object& o: objects)
        {
            mat4 model = translate(o.position) * o.rotation * scale(o.scale);
            GLint model_loc = glGetUniformLocation(prog, "model");
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model.data);
            glBindVertexArray(o.mesh->vao);
            glDrawArrays(GL_TRIANGLES, 0, o.mesh->vertex_count);
        }

        // axes

        glDepthMask(GL_FALSE);
        glUseProgram(prog_solid);
        {
            mat4 model = identity4();
            GLint proj_loc = glGetUniformLocation(prog_solid, "proj");
            GLint view_loc = glGetUniformLocation(prog_solid, "view");
            GLint model_loc = glGetUniformLocation(prog_solid, "model");
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(view_loc, 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model.data);
        }
        float d = 10000.f;
        draw_segment(prog_solid, vec3{-d,0,0}, vec3{d,0,0}, vec3{0.3,0,0});
        draw_segment(prog_solid, vec3{0,-d,0}, vec3{0,d,0}, vec3{0,0.3,0});
        draw_segment(prog_solid, vec3{0,0,-d}, vec3{0,0,d}, vec3{0,0,0.3});
        glDepthMask(GL_TRUE);

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
