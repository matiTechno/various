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

void _ras_draw1(vec4* coords, vec3* colors);
void _ras_draw2(vec4* coords, vec3* colors);

void ras_draw(Vertex* verts, int count, mat4 transform)
{
    for(int base = 0; base < count; base += 3)
    {
        vec4 coords[3];
        vec3 colors[3];

        for(int j = 0; j < 3; ++j)
        {
            coords[j].x = verts[base + j].pos.x;
            coords[j].y = verts[base + j].pos.y;
            coords[j].z = verts[base + j].pos.z;
            coords[j].w = 1;
            coords[j] = mul(transform, coords[j]);
            colors[j] = verts[base + j].color;
        }
        _ras_draw1(coords, colors);
    }
}

#define W_CLIP 0.001f

void _ras_draw1(vec4* coords, vec3* colors)
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
        vec3 colors2[3];
        memcpy(coords2, coords, sizeof(coords2));
        memcpy(colors2, colors, sizeof(colors2));
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
                colors[idx_curr] = colors[idx_curr] + t1*(colors[idx_prev] - colors[idx_curr]);
                coords2[idx_prev] = coords[idx_curr];
                colors2[idx_prev] = colors[idx_curr];
                coords2[idx_curr] = coords2[idx_curr] + t2*(coords2[idx_next] - coords2[idx_curr]);
                colors2[idx_curr] = colors2[idx_curr] + t2*(colors2[idx_next] - colors2[idx_curr]);
                break;
            }
            idx_prev = idx_curr;
            idx_curr = idx_next;
            idx_next += 1;
        }
        _ras_draw2(coords, colors);
        _ras_draw2(coords2, colors2);
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
                colors[idx_prev] = colors[idx_curr] + t1*(colors[idx_prev] - colors[idx_curr]);
                coords[idx_next] = coords[idx_curr] + t2*(coords[idx_next] - coords[idx_curr]);
                colors[idx_next] = colors[idx_curr] + t2*(colors[idx_next] - colors[idx_curr]);
                break;
            }
            idx_prev = idx_curr;
            idx_curr = idx_next;
            idx_next += 1;
        }
        _ras_draw2(coords, colors);
    }
    else
        _ras_draw2(coords, colors);
}

float signed_area(float lhs_x, float lhs_y, float rhs_x, float rhs_y)
{
    return (lhs_x * rhs_y) - (lhs_y * rhs_x);
}

void _ras_draw2(vec4* coords, vec3* colors)
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
    float arm01x = coords[1].x - coords[0].x;
    float arm01y = coords[1].y - coords[0].y;
    float arm12x = coords[2].x - coords[1].x;
    float arm12y = coords[2].y - coords[1].y;
    float arm20x = coords[0].x - coords[2].x;
    float arm20y = coords[0].y - coords[2].y;
    float area = signed_area(arm01x, arm01y, arm12x, arm12y);

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
            float b0 = signed_area(arm12x, arm12y, px - coords[1].x, py - coords[1].y) / area;
            float b1 = signed_area(arm20x, arm20y, px - coords[2].x, py - coords[2].y) / area;
            float b2 = signed_area(arm01x, arm01y, px - coords[0].x, py - coords[0].y) / area;

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
            vec3 color = (b0 * coords[0].w * colors[0]) + (b1 * coords[1].w * colors[1]) + (b2 * coords[2].w * colors[2]);
            color = 1.f / (b0 * coords[0].w + b1 * coords[1].w + b2 * coords[2].w) * color;

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
    //SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 100, 100, SDL_WINDOW_OPENGL);
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
    triangle[0].pos = {-0.5, 0, 4};
    triangle[1].pos = {0.5, 0, 4};
    triangle[2].pos = {0, 1, -5};
    triangle[0].color = {1, 0, 0};
    triangle[1].color = {0, 1, 0};
    triangle[2].color = {0, 0, 1};

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
            ras_draw(triangle, 3, mul(proj, view));
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
