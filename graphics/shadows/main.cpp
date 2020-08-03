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
uniform mat4 light_proj;
uniform mat4 light_view;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

out vec3 frag_pos;
out vec3 frag_normal;
out vec3 frag_shadow;

void main()
{
    vec4 pos_w = model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = mat3(model) * normal;
    frag_shadow = ( (light_proj * light_view * pos_w).xyz + 1 ) / 2;
}
)";

const char* _src_frag = R"(
#version 330
uniform vec3 light_intensity = vec3(1,1,1);
uniform float ambient_intensity = 0.01;
uniform vec3 diffuse_color = vec3(0.3);
uniform vec3 specular_color = vec3(0.1);
uniform float specular_exp = 20;
uniform vec3 eye_pos;
uniform vec3 light_dir;
uniform sampler2D sampler0;

in vec3 frag_pos;
in vec3 frag_normal;
in vec3 frag_shadow;

out vec3 out_color;

void main()
{
    float dst_shadow_z = texture(sampler0, frag_shadow.xy).r;
    vec3 color_coeff = vec3(1);

    if(frag_shadow.z - 0.0005 > dst_shadow_z)
        color_coeff = vec3(0);

    vec3 L = normalize(light_dir);
    vec3 N = normalize(frag_normal);
    vec3 ambient_comp = diffuse_color * ambient_intensity;
    vec3 diff_comp = diffuse_color * light_intensity * max(dot(N, L), 0);

    vec3 V = normalize(eye_pos - frag_pos);
    vec3 H = normalize(V + L);
    vec3 spec_comp = specular_color * light_intensity * pow( max(dot(N, H), 0), specular_exp) * float(dot(N, L) > 0);

    out_color = ambient_comp + color_coeff * (diff_comp + spec_comp);
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

struct Vertex
{
    vec3 pos;
    vec3 normal;
};

struct Mesh
{
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
    OBJ_FACE,
};

Mesh load(const char* filename)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    char buf[256];
    std::vector<vec3> positions;
    std::vector<vec3> normals;
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
        case OBJ_FACE:
        {
            for(int i = 0; i < 3; ++i)
            {
                int pos_id, norm_id;
                r = fscanf(file, "%d//%d", &pos_id, &norm_id);
                assert(r == 2);
                Vertex vert;
                vert.pos = positions[pos_id - 1];
                vert.normal = normals[norm_id - 1];
                verts.push_back(vert);
            }
            break;
        }
        case OBJ_OTHER:
            break;
        }
    }
    Mesh mesh;
    mesh.vert_count = verts.size();
    mesh.model_tf = identity4();
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
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

#define TEX_SIZE 1024

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

    GLuint program = create_program(_src_vert, _src_frag);
    GLuint program_shadow = create_program(_src_vert2, nullptr);

    GLuint tex_shadow;
    glGenTextures(1, &tex_shadow);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, TEX_SIZE, TEX_SIZE, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    GLuint fbo_shadow;
    glGenFramebuffers(1, &fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_shadow, 0);

    vec3 light_pos = {50,50,0};
    vec3 center = {0,0,0};
    vec3 eye_pos = {-15,30,15};
    Mesh mesh = load("shadow_test.obj");
    bool quit = false;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                quit = true;
        }
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        light_pos = rotate_y(dt * 2*pi/30) * light_pos;
        prev_counter = current_counter;
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
        glViewport(0,0,TEX_SIZE,TEX_SIZE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_shadow);
        vec3 light_dir = normalize(light_pos - center);
        mat4 light_view = lookat(light_pos, -light_dir);
        mat4 light_proj = orthographic(-20, 20, -20, 20, 0.1, 1000);
        glUniformMatrix4fv(glGetUniformLocation(program_shadow, "proj"), 1, GL_TRUE, light_proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program_shadow, "view"), 1, GL_TRUE, light_view.data);
        glUniformMatrix4fv(glGetUniformLocation(program_shadow, "model"), 1, GL_TRUE, mesh.model_tf.data);
        glBindVertexArray(mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, mesh.vert_count);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, tex_shadow);
        glUseProgram(program);
        mat4 view = lookat(eye_pos, normalize(center - eye_pos));
        mat4 proj = perspective(to_radians(90), (float)width / height, 0.1, 1000);
        glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, view.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, mesh.model_tf.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "light_proj"), 1, GL_TRUE, light_proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "light_view"), 1, GL_TRUE, light_view.data);
        glUniform3fv(glGetUniformLocation(program, "eye_pos"), 1, &eye_pos.x);
        glUniform3fv(glGetUniformLocation(program, "light_dir"), 1, &light_dir.x);
        glBindVertexArray(mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, mesh.vert_count);

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
