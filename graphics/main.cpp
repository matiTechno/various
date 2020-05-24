#include <SDL2/SDL.h>
#include <assert.h>
#include "glad.h"
#include "main.hpp"

struct Vertex
{
    vec3 pos;
    vec3 color;
};

static const char* src_vert = R"(
#version 330
uniform mat4 view;
uniform mat4 proj;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;
out vec3 fcolor;

void main()
{
    gl_Position = proj * view * vec4(pos, 1);
    fcolor = color;
}
)";

static const char* src_frag = R"(
#version 330
in vec3 fcolor;
out vec3 out_color;

void main()
{
    out_color = fcolor;
}
)";

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
    bool quit = false;
    bool enable_move = false;

    Vertex triangle[3];
    triangle[0].color = {1, 0, 0};
    triangle[0].pos = {-0.5, -0.5, -2};
    triangle[1].color = {0, 1, 0};
    triangle[1].pos = {0.5, -0.5, -2};
    triangle[2].color = {0, 0, 1};
    triangle[2].pos = {0, 0.5, -3};

    GLuint vert_buffer;
    GLuint vert_array;
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();

    glGenBuffers(1, &vert_buffer);
    glGenVertexArrays(1, &vert_array);

    glBindBuffer(GL_ARRAY_BUFFER, vert_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), triangle, GL_STATIC_DRAW);

    glBindVertexArray(vert_array);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(vec3));

    glShaderSource(vert_shader, 1, &src_vert, nullptr);
    glCompileShader(vert_shader);
    glShaderSource(frag_shader, 1, &src_frag, nullptr);
    glCompileShader(frag_shader);
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);
    glUseProgram(program);

    GLint view_loc = glGetUniformLocation(program, "view");
    GLint proj_loc = glGetUniformLocation(program, "proj");

    float pitch = 0;
    float yaw = 0;

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
        glViewport(0, 0, width, height);

        vec3 camera_pos = {0.f, 1.f, 3.f};
        mat4 view = lookat(camera_pos, yaw, pitch);
        mat4 proj = perspective(90, (float)width/height, 0.1f, 100.f);

        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view.data);
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj.data);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
