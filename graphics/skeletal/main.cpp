#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "../main.hpp"

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

static const char* src_vert_deb = R"(
#version 330
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
layout(location = 0) in vec3 pos;
void main() { gl_Position = proj * view * model * vec4(pos, 1); }
)";

static const char* src_frag_deb = R"(
#version 330
out vec3 out_color;
void main()
{ out_color = vec3(1, 0, 0); }
)";

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
    vec4* bone_ids;
    vec4* weights;
    int vertex_count;

    int* parent_ids;
    mat4* tfs_bone_to_mesh;
    int bone_count;

    mat4 model_transform;
    vec3 diffuse_color;
    vec3 specular_color;
    float specular_exp;
    bool debug;
};

void load_model(const char* filename, RenderCmd& cmd)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    std::vector<vec3> positions;
    std::vector<vec3> normals;
    std::vector<vec4> bone_ids;
    std::vector<vec4> weights;
    std::vector<int> indices;
    std::vector<int> parents_ids;
    std::vector<mat4> tfs_bone_to_mesh;

    bool done = false;

    while(!done)
    {
        int code = fgetc(file);

        switch(code)
        {
        case 'v':
        {
            vec3 v;
            int n = fscanf(file, " %f %f %f ", &v.x, &v.y, &v.z);
            assert(n == 3);
            positions.push_back(v);
            break;
        }
        case 'n':
        {
            vec3 v;
            int n = fscanf(file, " %f %f %f ", &v.x, &v.y, &v.z);
            assert(n == 3);
            normals.push_back(v);
            break;
        }
        case 'w':
        {
            vec4 bone;
            vec4 weight;
            int n = fscanf(file, " %f %f %f %f %f %f %f %f ", &bone.x, &bone.y, &bone.z, &bone.w,
                &weight.x, &weight.y, &weight.z, &weight.w);
            assert(n == 8);
            bone_ids.push_back(bone);
            weights.push_back(weight);
            break;
        }
        case 'f':
        {
            for(int i = 0; i < 3; ++i)
            {
                int id;
                int n = fscanf(file, " %d ", &id);
                assert(n == 1);
                indices.push_back(id);
            }
            break;
        }
        case 'b':
        {
            int pid;
            mat4 m;
            int n = fscanf(file, " %d ", &pid);
            assert(n == 1);

            for(int i = 0; i < 16; ++i)
            {
                n = fscanf(file, " %f ", m.data + i);
                assert(n == 1);
            }
            parents_ids.push_back(pid);
            tfs_bone_to_mesh.push_back(m);
            break;
        }
        default:
            assert(code == EOF);
            done = true;
            break;
        }
    }
    assert(positions.size());
    cmd.vertex_count = indices.size();
    assert(cmd.vertex_count % 3 == 0);
    cmd.positions = (vec3*)malloc(sizeof(vec3) * cmd.vertex_count);
    cmd.normals = (vec3*)malloc(sizeof(vec3) * cmd.vertex_count);
    cmd.bone_ids = (vec4*)malloc(sizeof(vec4) * cmd.vertex_count);
    cmd.weights = (vec4*)malloc(sizeof(vec4) * cmd.vertex_count);

    for(int i = 0; i < cmd.vertex_count; ++i)
    {
        int id = indices[i];
        cmd.positions[i] = positions[id];
        cmd.normals[i] = normals[id];
        cmd.bone_ids[i] = bone_ids[id];
        cmd.weights[i] = weights[id];
    }

    cmd.bone_count = bone_ids.size();
    cmd.parent_ids = (int*)malloc(sizeof(int) * cmd.bone_count);
    cmd.tfs_bone_to_mesh = (mat4*)malloc(sizeof(mat4) * cmd.bone_count);

    for(int i = 0; i < cmd.bone_count; ++i)
    {
        cmd.parent_ids[i] = parents_ids[i];
        cmd.tfs_bone_to_mesh[i] = tfs_bone_to_mesh[i];
    }
}

GLuint _program;
GLuint _program_debug;

void gl_draw(RenderCmd& cmd)
{
    if(cmd.debug)
    {
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
    }
    else
    {
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
    }

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

    GLuint program = cmd.debug ? _program_debug : _program;
    glUseProgram(program);

    GLint view_loc = glGetUniformLocation(program, "view");
    GLint proj_loc = glGetUniformLocation(program, "proj");
    GLint model_loc = glGetUniformLocation(program, "model");
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, cmd.view.data);
    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, cmd.proj.data);
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, cmd.model_transform.data);

    if(!cmd.debug)
    {
        GLint light_int_loc = glGetUniformLocation(program, "light_intensity");
        GLint light_dir_loc = glGetUniformLocation(program, "light_dir");
        GLint ambient_int_loc = glGetUniformLocation(program, "ambient_intensity");
        GLint eye_pos_loc = glGetUniformLocation(program, "eye_pos");
        GLint diffuse_color_loc = glGetUniformLocation(program, "diffuse_color");
        GLint specular_color_loc = glGetUniformLocation(program, "specular_color");
        GLint specular_exp_loc = glGetUniformLocation(program, "specular_exp");
        glUniform3fv(light_int_loc, 1, &cmd.light_intensity.x);
        glUniform3fv(light_dir_loc, 1, &cmd.light_dir.x);
        glUniform1f(ambient_int_loc, cmd.ambient_intensity);
        glUniform3fv(eye_pos_loc, 1, &cmd.eye_pos.x);
        glUniform3fv(diffuse_color_loc, 1, &cmd.diffuse_color.x);
        glUniform3fv(specular_color_loc, 1, &cmd.specular_color.x);
        glUniform1f(specular_exp_loc, cmd.specular_exp);
    }

    glDrawArrays(GL_TRIANGLES, 0, cmd.vertex_count);
    // note: this is super important
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

RenderCmd test_triangle()
{
    vec3* positions = (vec3*)malloc(3 * sizeof(vec3));
    vec3* normals = (vec3*)malloc(3 * sizeof(vec3));
    positions[0] = {-1,-1,0};
    positions[1] = {1,-1,0};
    positions[2] = {0,1,0};
    vec3 e01 = positions[1] - positions[0];
    vec3 e02 = positions[2] - positions[0];
    vec3 normal = normalize(cross(e01, e02));
    normals[0] = normals[1] = normals[2] = normal;
    RenderCmd cmd;
    cmd.positions = positions;
    cmd.normals = normals;
    cmd.vertex_count = 3;
    cmd.model_transform = identity4();
    cmd.debug = true;
    return cmd;
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        assert(false);
    }
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 100, 100, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
    {
        assert(false);
    }

    _program = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(_program, vert_shader);
        glAttachShader(_program, frag_shader);
        glLinkProgram(_program);
    }
    _program_debug = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert_deb, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag_deb, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(_program_debug, vert_shader);
        glAttachShader(_program_debug, frag_shader);
        glLinkProgram(_program_debug);
    }
    bool quit = false;
    bool enable_move = false;
    vec3 camera_pos = {0.f, 2.f, 5.f};
    float pitch = 0;
    float yaw = 0;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    RenderCmd cmd;
    cmd.light_intensity = {0.4, 0.4, 0.4};
    cmd.light_dir = normalize(vec3{1, 1, 0.5});
    cmd.ambient_intensity = 0.01;
    cmd.model_transform = identity4();
    cmd.diffuse_color = {0.15, 0.15, 0.15};
    cmd.specular_color = {1, 0, 0};
    cmd.specular_exp = 60;
    cmd.debug = false;

    load_model("/home/mat/test.anim", cmd);
    cmd.model_transform = rotate_y(pi/2) * scale({0.1,0.1,0.1});

    RenderCmd cmd_deb = test_triangle();
    cmd_deb.model_transform = scale({0.5,0.5,0.5});

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

        cmd.view = lookat(camera_pos, yaw, pitch);
        cmd.proj = perspective(60, (float)width/height, 0.1f, 100.f);
        cmd.eye_pos = camera_pos;

        cmd_deb.view = cmd.view;
        cmd_deb.proj = cmd.proj;
        cmd_deb.eye_pos = cmd.eye_pos;

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;
        (void)dt;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl_draw(cmd);

        for(int i = 0; i < 4; ++i)
        {
            RenderCmd cmd_bone = cmd_deb;
            mat4 tf_bone_to_mesh = cmd.tfs_bone_to_mesh[i];
            cmd_bone.model_transform = cmd.model_transform * tf_bone_to_mesh * cmd_bone.model_transform;
            gl_draw(cmd_bone);
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
