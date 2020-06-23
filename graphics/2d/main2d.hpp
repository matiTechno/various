#include "../main.hpp"
#include <SDL2/SDL.h>

mat3 translate(vec2 v)
{
    mat3 m = identity3();
    m.data[2] = v.x;
    m.data[5] = v.y;
    return m;
}

// eye_x and eye_y must be normalized; view_from_world
inline
mat3 view(vec2 eye_pos, vec2 eye_x, vec2 eye_y)
{
    mat3 rot = {};
    rot.data[0] = eye_x.x;
    rot.data[1] = eye_x.y;
    rot.data[3] = eye_y.x;
    rot.data[4] = eye_y.y;
    rot.data[8] = 1;
    return rot * translate(-eye_pos);
}

// clip_from_view
inline
mat3 orthographic(float l, float r, float b, float t)
{
    mat3 m = {};
    m.data[0] = 2/(r-l);
    m.data[2] = -(r+l)/(r-l);
    m.data[4] = 2/(t-b);
    m.data[5] = -(t+b)/(t-b);
    m.data[8] = 1;
    return m;
}

inline
vec2 transform2(mat3 m, vec2 v)
{
    vec3 tmp = m * vec3{v.x, v.y, 1};
    return {tmp.x, tmp.y};
}

// to get a cursor coordinates in a world space use inverse( window_from_clip * cip_from_world )
inline
mat3 window_from_clip(float width, float height)
{
    mat3 m = {};
    m.data[0] = width/2;
    m.data[2] = width/2;
    m.data[4] = -height/2;
    m.data[5] = height/2;
    m.data[8] = 1;
    return m;
}

struct Nav2d
{
    vec2 eye_pos;
    vec2 eye_x;
    vec2 eye_y;
    float right;
    float top;
    vec2 cursor_win;
    vec2 cursor_world;
    mat3 clip_f_world;
    float win_width;
    float win_height;
    bool rmb_down;
    bool mmb_down;
};

// internal
inline
void rebuild_matrix(Nav2d& nav)
{
    float right = nav.right;
    float top = nav.top;
    float rt_aspect = right / top;
    float aspect = nav.win_width / nav.win_height;

    if(aspect > rt_aspect)
        right = aspect * top;
    else
        top = right / aspect;

    nav.clip_f_world = orthographic(-right, right, -top, top) * view(nav.eye_pos, nav.eye_x, nav.eye_y);
}

inline
void nav_init(Nav2d& nav, vec2 eye_pos, float right, float top, float win_width, float win_height)
{
    nav.eye_pos = eye_pos;
    nav.eye_x = {1,0};
    nav.eye_y = {0,1};
    nav.right = right;
    nav.top = top;
    nav.rmb_down = false;
    nav.mmb_down = false;
    nav.win_width = win_width;
    nav.win_height = win_height;
    rebuild_matrix(nav);
}

inline
void nav_process_event(Nav2d& nav, SDL_Event& e)
{
    if(e.type == SDL_MOUSEMOTION)
    {
        mat3 world_f_window = inverse( window_from_clip(nav.win_width, nav.win_height) * nav.clip_f_world );
        vec2 new_cursor_win = {(float)e.motion.x, (float)e.motion.y};
        vec2 new_cursor_world = transform2(world_f_window, new_cursor_win);

        if(nav.rmb_down)
        {
            // translate
            nav.eye_pos = nav.eye_pos - (new_cursor_world - nav.cursor_world);
            rebuild_matrix(nav);
        }
        else if(nav.mmb_down)
        {
            // rotate around eye_pos
            vec2 v1 = nav.cursor_world - nav.eye_pos;
            vec2 v2 = new_cursor_world - nav.eye_pos;
            float a1 = atan2f(v1.y, v1.x);
            float a2 = atan2f(v2.y, v2.x);
            mat3 rot = mat4_to_mat3( rotate_z(a1 - a2) );
            nav.eye_x = transform2(rot, nav.eye_x);
            nav.eye_y = transform2(rot, nav.eye_y);
            rebuild_matrix(nav);
            // recalculate cursor world cooridnates
            world_f_window = inverse( window_from_clip(nav.win_width, nav.win_height) * nav.clip_f_world);
            nav.cursor_world = transform2(world_f_window, new_cursor_win);
        }
        else
            nav.cursor_world = new_cursor_world;

        nav.cursor_win = new_cursor_win;
    }
    else if(e.type == SDL_MOUSEWHEEL)
    {
        // zoom to cursor
        float scale = e.wheel.y < 0 ? (1.f / powf(1.5, -e.wheel.y)) : powf(1.5, e.wheel.y);
        nav.right /= scale;
        nav.top /= scale;
        rebuild_matrix(nav);
        mat3 world_f_window = inverse( window_from_clip(nav.win_width, nav.win_height) * nav.clip_f_world );
        vec2 new_cursor_world = transform2(world_f_window, nav.cursor_win);
        nav.eye_pos = nav.eye_pos - (new_cursor_world - nav.cursor_world);
        rebuild_matrix(nav);
    }
    else if(e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
    {
        nav.win_width = e.window.data1;
        nav.win_height = e.window.data2;
        rebuild_matrix(nav);
    }
    else if(e.type == SDL_MOUSEBUTTONDOWN)
    {
        if(e.button.button == SDL_BUTTON_RIGHT)
            nav.rmb_down = true;
        else if(e.button.button == SDL_BUTTON_MIDDLE)
            nav.mmb_down = true;
    }
    else if(e.type == SDL_MOUSEBUTTONUP)
    {
        if(e.button.button == SDL_BUTTON_RIGHT)
            nav.rmb_down = false;
        else if(e.button.button == SDL_BUTTON_MIDDLE)
            nav.mmb_down = false;
    }
}
