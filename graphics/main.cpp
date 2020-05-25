#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
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

// software rasterizer global state
struct
{
    int width;
    int height;
    float* depth_buf;
    u8* color_buf;
    // opengl resources for ras_display()
    GLuint vert_buffer;
    GLuint vert_array;
    GLuint program;
    GLuint texture;
} _ras;

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

    glGenBuffers(1, &_ras.vert_buffer);
    glGenVertexArrays(1, &_ras.vert_array);

    glBindBuffer(GL_ARRAY_BUFFER, _ras.vert_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glBindVertexArray(_ras.vert_array);
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
    glBindVertexArray(_ras.vert_array);
    glUseProgram(_ras.program);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);
}

void ras_clear_buffers()
{
    for(int i = 0; i < _ras.width * _ras.height; ++i)
        _ras.depth_buf[i] = 1;
    memset(_ras.color_buf, 0, _ras.width * _ras.height * 3);
}

float signed_area(float lhs_x, float lhs_y, float rhs_x, float rhs_y)
{
    return (lhs_x * rhs_y) - (lhs_y * rhs_x);
}

void ras_render(Vertex* verts, mat4 transform)
{
    vec4 clip_coords[3];

    for(int i = 0; i < 3; ++i)
    {
        vec3 p = verts[i].pos;
        vec4 hp = {p.x, p.y, p.z, 1};
        clip_coords[i] = mul(transform, hp);
    }

    // todo: clipping

    vec3 ndc_coords[3];

    for(int i = 0; i < 3; ++i)
    {
        // perspective division
        ndc_coords[i].x = clip_coords[i].x / clip_coords[i].w;
        ndc_coords[i].y = clip_coords[i].y / clip_coords[i].w;
        ndc_coords[i].z = clip_coords[i].z / clip_coords[i].w;
    }

    vec3 win_coords[3];
    float w_inv[3];

    for(int i = 0; i < 3; ++i)
    {
        // viewport transform
        win_coords[i].x = (_ras.width / 2.f) * (ndc_coords[i].x + 1.f);
        win_coords[i].y = (_ras.height / 2.f) * (ndc_coords[i].y + 1.f);
        win_coords[i].z = (1.f / 2.f) * (ndc_coords[i].z + 1.f);
        w_inv[i] = 1.f / clip_coords[i].w;
    }

    // face culling
    vec3 arm01 = win_coords[1] - win_coords[0];
    vec3 arm12 = win_coords[2] - win_coords[1];
    vec3 arm20 = win_coords[0] - win_coords[2];
    float area = signed_area(arm01.x, arm01.y, arm12.x, arm12.y);

    if(area < 0)
        return;

    int min_x = win_coords[0].x;
    int max_x = min_x;
    int min_y = win_coords[0].y;
    int max_y = min_y;

    for(int i = 1; i < 3; ++i)
    {
        min_x = win_coords[i].x < min_x ? win_coords[i].x : min_x;
        max_x = win_coords[i].x > max_x ? win_coords[i].x : max_x;
        min_y = win_coords[i].y < min_y ? win_coords[i].y : min_y;
        max_y = win_coords[i].y > max_y ? win_coords[i].y : max_y;
    }

    for(int y = min_y; y <= max_y; ++y)
    {
        for(int x = min_x; x <= max_x; ++x)
        {
            float px = x + 0.5f;
            float py = y + 0.5f;
            float b0 = signed_area(arm12.x, arm12.y, px - win_coords[1].x, py - win_coords[1].y) / area;
            float b1 = signed_area(arm20.x, arm20.y, px - win_coords[2].x, py - win_coords[2].y) / area;
            float b2 = signed_area(arm01.x, arm01.y, px - win_coords[0].x, py - win_coords[0].y) / area;

            if(b0 < 0 || b1 < 0 || b2 < 0)
                continue;

            float depth = (b0 * win_coords[0].z) + (b1 * win_coords[1].z) + (b2 * win_coords[2].z);
            int idx = y * _ras.width + x;

            if(depth > _ras.depth_buf[idx])
                continue;

            _ras.depth_buf[idx] = depth;

            // perspective correct interpolation
            vec3 color = (b0 * w_inv[0] * verts[0].color) + (b1 * w_inv[1] * verts[1].color) + (b2 * w_inv[2] * verts[2].color);
            color = 1.f / (b0 * w_inv[0] + b1 * w_inv[1] + b2 * w_inv[2]) * color;

            _ras.color_buf[idx*3 + 0] = 255.f * color.x + 0.5f;
            _ras.color_buf[idx*3 + 1] = 255.f * color.y + 0.5f;
            _ras.color_buf[idx*3 + 2] = 255.f * color.z + 0.5f;
        }
    }
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
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

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

    ras_init();

    bool quit = false;
    bool enable_move = false;
    bool use_ras = false;

    vec3 camera_pos = {0.f, 1.f, 3.f};
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
                case SDLK_1:
                    use_ras = !use_ras;
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
        mat4 view = lookat(camera_pos, yaw, pitch);
        mat4 proj = perspective(90, (float)width/height, 0.1f, 100.f);

        if(use_ras)
        {
            ras_viewport(width, height);
            ras_clear_buffers();
            ras_render(triangle, mul(proj, view));
            ras_display();
        }
        else
        {
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindVertexArray(vert_array);
            glUseProgram(program);
            glUniformMatrix4fv(view_loc, 1, GL_TRUE, view.data);
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj.data);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
