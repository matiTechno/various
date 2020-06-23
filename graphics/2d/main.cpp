#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "main2d.hpp"

static const char* _rnd_src_vert = R"(
#version 330
layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 color;
uniform mat3 clip_f_world;
out vec3 frag_color;
void main()
{
    gl_Position = vec4(clip_f_world * vec3(pos, 1), 1);
    frag_color = color;
}
)";

static const char* _rnd_src_frag = R"(
#version 330
out vec3 out_color;
in vec3 frag_color;
void main()
{
    out_color = frag_color;
}
)";

struct Vertex
{
    vec2 pos;
    vec3 color;
};

// renderer global state
struct
{
    std::vector<Vertex> vertices;
    GLuint program;
    GLuint vao;
    GLuint vbo;
} _rnd;

void rnd_init()
{
    _rnd.program = glCreateProgram();
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vert_shader, 1, &_rnd_src_vert, nullptr);
    glCompileShader(vert_shader);
    glShaderSource(frag_shader, 1, &_rnd_src_frag, nullptr);
    glCompileShader(frag_shader);
    glAttachShader(_rnd.program, vert_shader);
    glAttachShader(_rnd.program, frag_shader);
    glLinkProgram(_rnd.program);

    glGenVertexArrays(1, &_rnd.vao);
    glGenBuffers(1, &_rnd.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _rnd.vbo);
    glBindVertexArray(_rnd.vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(vec2));
}

void rnd_cache(Vertex v)
{
    _rnd.vertices.push_back(v);
}

void rnd_set_matrix(mat3 clip_f_world)
{
    glUseProgram(_rnd.program);
    int loc = glGetUniformLocation(_rnd.program, "clip_f_world");
    glUniformMatrix3fv(loc, 1, GL_TRUE, clip_f_world.data);
}

void rnd_flush(GLenum mode)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glUseProgram(_rnd.program);
    glBindVertexArray(_rnd.vao);
    glBindBuffer(GL_ARRAY_BUFFER, _rnd.vbo);
    int vertex_count = _rnd.vertices.size();
    glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(Vertex), _rnd.vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(mode, 0, vertex_count);
    _rnd.vertices.clear();
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
        assert(false);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 800, 600, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
        assert(false);

    rnd_init();

    bool quit = false;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    Nav2d nav;
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        nav_init(nav, vec2{0,0}, 10, 10, width, height);
    }

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
        rnd_set_matrix(nav.clip_f_world);

        {
            Vertex v;
            v.color = {1,1,1};
            v.pos = {0,0};
            rnd_cache(v);
            v.pos = {5, 0};
            rnd_cache(v);
            v.pos = {5,5};
            rnd_cache(v);
            rnd_flush(GL_TRIANGLES);
        }

        if(nav.mmb_down)
        {
            Vertex v;
            v.color = {0,1,0};
            v.pos = nav.eye_pos;
            rnd_cache(v);
            v.pos = nav.cursor_world;
            rnd_cache(v);
            rnd_flush(GL_LINES);
        }

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
