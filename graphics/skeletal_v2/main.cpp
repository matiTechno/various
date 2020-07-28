#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <stddef.h>
#include "../glad.h"
#include "../main.hpp"

#define MAX_BONES 128

static const char* src_vert = R"(
#version 330

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
uniform mat4 skinning_matrices[128];

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in ivec4 bone_ids;
layout(location = 3) in vec4 weights;

// in world space
out vec3 frag_pos;
out vec3 frag_normal;

void main()
{
    mat4 cp_from_bp = mat4(0);

    for(int i = 0; i < 4; ++i)
        cp_from_bp += skinning_matrices[bone_ids[i]] * weights[i];

    mat4 cp_model = model * cp_from_bp;
    vec4 pos_w = cp_model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = mat3(cp_model) * normal;
}
)";

static const char* src_frag = R"(
#version 330

uniform vec3 light_intensity = vec3(1,1,1);
uniform vec3 light_dir = vec3(0,1,0.5);
uniform float ambient_intensity = 0.1;
uniform vec3 diffuse_color = vec3(0.5,0.5,0.5);
uniform vec3 specular_color = vec3(1,0.5,0);
uniform float specular_exp = 20;
uniform vec3 eye_pos;

in vec3 frag_pos;
in vec3 frag_normal;
out vec3 out_color;

void main()
{
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
{ out_color = vec3(0.5, 0, 0); }
)";

struct Vertex
{
    vec3 pos;
    vec3 normal;
    int bone_ids[4];
    vec4 weights;
};

struct Mesh
{
    Vertex* vertices;
    int vertex_count;
    int* indices;
    int index_count;
    GLuint bo;
    GLuint vao;
    char** bone_names;
    mat4* bone_f_bp_mesh;
    int* bone_parent_ids;
    int bone_count;
    int ebo_offset;
};

struct BoneAction
{
    vec3* locs;
    vec4* rots;
    float* loc_time_coords;
    float* rot_time_coords;
    int loc_count;
    int rot_count;
};

struct Action
{
    const char* name;
    BoneAction* bone_actions;
    int bone_count;
    float duration;
};

struct Object
{
    Mesh* mesh;
    Action* action;
    mat4 model_tf;
    mat4* skinning_matrices;
    mat4* cp_model_f_bone;
    float action_time;
};

void load(const char* filename, Mesh& mesh, std::vector<Action>& actions)
{
    mesh = {};
    FILE* file = fopen(filename, "r");
    assert(file);
    char str_buf[256];
    int r = fscanf(file, " format %s", str_buf);
    assert(r == 1);

    bool uvs = false;

    if(strcmp(str_buf, "punbw") == 0)
        uvs = true;

    r = fscanf(file, " vertex_count %d", &mesh.vertex_count);
    assert(r == 1);
    mesh.vertices = (Vertex*)malloc(mesh.vertex_count * sizeof(Vertex));

    for(int i = 0; i < mesh.vertex_count; ++i)
    {
        Vertex& v = mesh.vertices[i];
        r = fscanf(file, "%f %f %f", &v.pos.x, &v.pos.y, &v.pos.z);
        assert(r == 3);

        if(uvs)
        {
            vec2 dummy;
            r = fscanf(file, "%f %f", &dummy.x, &dummy.y);
            assert(r == 2);
        }
        r = fscanf(file, "%f %f %f", &v.normal.x, &v.normal.y, &v.normal.z);
        assert(r == 3);

        for(int i = 0; i < 4; ++i)
        {
            r = fscanf(file, "%d", v.bone_ids + i);
            assert(r == 1);
        }

        for(int i = 0; i < 4; ++i)
        {
            r = fscanf(file, "%f", &v.weights[i]);
            assert(r == 1);
        }
    }

    r = fscanf(file, " index_count %d", &mesh.index_count);
    assert(r == 1);
    mesh.indices = (int*)malloc(mesh.index_count * sizeof(int));

    for(int i = 0; i < mesh.index_count; ++i)
    {
        r = fscanf(file, "%d", mesh.indices + i);
        assert(r == 1);
    }

    r = fscanf(file, " bone_count %d", &mesh.bone_count);
    assert(r == 1);
    mesh.bone_names = (char**)malloc(mesh.bone_count * sizeof(char*));
    mesh.bone_parent_ids = (int*)malloc(mesh.bone_count * sizeof(int));
    mesh.bone_f_bp_mesh = (mat4*)malloc(mesh.bone_count * sizeof(mat4));

    for(int bone_id = 0; bone_id < mesh.bone_count; ++bone_id)
    {
        r = fscanf(file, " %s ", str_buf);
        assert(r == 1);
        mesh.bone_names[bone_id] = strdup(str_buf);
        r = fscanf(file, "%d", mesh.bone_parent_ids + bone_id);
        assert(r == 1);

        for(int i = 0; i < 16; ++i)
        {
            r = fscanf(file, "%f", mesh.bone_f_bp_mesh[bone_id].data + i);
            assert(r == 1);
        }
    }
    int bone_count;
    r = fscanf(file, " bone_count %d", &bone_count);
    assert(r == 1);
    int action_count;
    r = fscanf(file, " action_count %d", &action_count);
    assert(r == 1);

    for(int _i = 0; _i < action_count; ++_i)
    {
        Action action;
        action.bone_count = bone_count;
        r = fscanf(file, " action_name %s", str_buf);
        assert(r == 1);
        action.name = strdup(str_buf);
        action.bone_actions = (BoneAction*)malloc(action.bone_count * sizeof(BoneAction));

        for(int bone_id = 0; bone_id < action.bone_count; ++bone_id)
        {
            BoneAction& ba = action.bone_actions[bone_id];
            r = fscanf(file, " loc_count %d", &ba.loc_count);
            assert(r == 1);
            ba.locs = (vec3*)malloc(ba.loc_count * sizeof(vec3));
            ba.loc_time_coords = (float*)malloc(ba.loc_count * sizeof(float));

            for(int i = 0; i < ba.loc_count; ++i)
            {
                r = fscanf(file, "%f %f %f %f", &ba.locs[i].x, &ba.locs[i].y, &ba.locs[i].z, ba.loc_time_coords + i);
                assert(r == 4);
            }

            r = fscanf(file, " rot_count %d", &ba.rot_count);
            assert(r == 1);
            ba.rots = (vec4*)malloc(ba.rot_count * sizeof(vec4));
            ba.rot_time_coords = (float*)malloc(ba.rot_count * sizeof(float));

            for(int i = 0; i < ba.rot_count; ++i)
            {
                r = fscanf(file, "%f %f %f %f %f", &ba.rots[i].x, &ba.rots[i].y, &ba.rots[i].z, &ba.rots[i].w, ba.rot_time_coords + i);
                assert(r == 5);
            }
        }
        assert(action.bone_count);
        int i = action.bone_actions[0].loc_count - 1;
        action.duration = action.bone_actions[0].loc_time_coords[i];
        actions.push_back(action);
    }

    r = fscanf(file, " %s", str_buf);
    assert(r == EOF);
    fclose(file);

    if(mesh.vertex_count == 0)
        return;
    mesh.ebo_offset = mesh.vertex_count * sizeof(Vertex);
    glGenBuffers(1, &mesh.bo);
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.bo);
    glBufferData(GL_ARRAY_BUFFER, mesh.index_count * sizeof(int) + mesh.vertex_count * sizeof(Vertex), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, mesh.ebo_offset, mesh.vertices);
    glBufferSubData(GL_ARRAY_BUFFER, mesh.ebo_offset, mesh.index_count * sizeof(int), mesh.indices);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glVertexAttribIPointer(2, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, bone_ids));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, weights));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.bo);
}

void alloc_object_anim_data(Object& obj, Mesh& mesh)
{
    if(!mesh.bone_count)
        return;
    obj.cp_model_f_bone = (mat4*)malloc(mesh.bone_count * sizeof(mat4));
    obj.skinning_matrices = (mat4*)malloc(mesh.bone_count * sizeof(mat4));
}

void update_anim_data(Object& obj, float dt)
{
    if(!obj.action)
    {
        for(int i = 0; i < obj.mesh->bone_count; ++i)
        {
            obj.skinning_matrices[i] = identity4();
            obj.cp_model_f_bone[i] = invert_coord_change(obj.mesh->bone_f_bp_mesh[i]);
        }
        return;
    }
    assert(obj.mesh->bone_count == obj.action->bone_count);
    obj.action_time += dt;
    float dur = obj.action->duration;

    if(obj.action_time > dur)
        obj.action_time = min(dur, obj.action_time - dur);
    else if(obj.action_time < 0)
        obj.action_time = max(0, obj.action_time + dur);

    for(int bone_id = 0; bone_id < obj.mesh->bone_count; ++bone_id)
    {
        BoneAction& action = obj.action->bone_actions[bone_id];
        int loc_id = 0;
        int rot_id = 0;

        while(obj.action_time > action.loc_time_coords[loc_id + 1])
            loc_id += 1;

        while(obj.action_time > action.rot_time_coords[rot_id + 1])
            rot_id += 1;

        assert(loc_id < action.loc_count - 1);
        assert(rot_id < action.rot_count - 1);
        float loc_lhs_t = action.loc_time_coords[loc_id];
        float loc_rhs_t = action.loc_time_coords[loc_id + 1];
        float rot_lhs_t = action.rot_time_coords[rot_id];
        float rot_rhs_t = action.rot_time_coords[rot_id + 1];
        float loc_t = (obj.action_time - loc_lhs_t) / (loc_rhs_t - loc_lhs_t);
        float rot_t = (obj.action_time - rot_lhs_t) / (rot_rhs_t - rot_lhs_t);
        vec3 loc_lhs = action.locs[loc_id];
        vec3 loc_rhs = action.locs[loc_id + 1];
        vec4 rot_lhs = action.rots[rot_id];
        vec4 rot_rhs = action.rots[rot_id + 1];

        // interpolate through the shorter path
        if(dot(rot_lhs, rot_rhs) < 0)
            rot_lhs = -1 * rot_lhs;

        vec3 loc = ((1 - loc_t) * loc_lhs) + (loc_t * loc_rhs);
        vec4 rot = ((1 - rot_t) * rot_lhs) + (rot_t * rot_rhs);
        rot = normalize(rot); // linear interpolation does not preserve length (quat_to_mat4() requires a unit quaternion)
        mat4 parent_f_bone = translate(loc) * quat_to_mat4(rot);
        mat4 cp_model_f_parent = identity4();

        if(bone_id > 0)
        {
            int pid = obj.mesh->bone_parent_ids[bone_id];
            assert(pid < bone_id);
            cp_model_f_parent = obj.cp_model_f_bone[pid];
        }
        obj.cp_model_f_bone[bone_id] = cp_model_f_parent * parent_f_bone;
        obj.skinning_matrices[bone_id] = obj.cp_model_f_bone[bone_id] * obj.mesh->bone_f_bp_mesh[bone_id];
    }
}

mat4 gl_from_blender()
{
    return rotate_x4(-pi/2);
}

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

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
        assert(false);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 1000, 1000, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
        assert(false);

    GLuint program = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);
    }
    GLuint program_debug = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert_deb, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag_deb, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(program_debug, vert_shader);
        glAttachShader(program_debug, frag_shader);
        glLinkProgram(program_debug);
    }
    Mesh mesh;
    std::vector<Action> actions;
    load("/home/mat/Downloads/blender-2.83.0-linux64/anim_data", mesh, actions);

    for(int i = 0; i < mesh.bone_count; ++i)
        printf("%s\n", mesh.bone_names[i]);

    assert(mesh.vertex_count);
    Object obj;
    obj.mesh = &mesh;
    obj.action = actions.empty() ? nullptr : actions.data();
    obj.model_tf = gl_from_blender();
    alloc_object_anim_data(obj, mesh);
    obj.action_time = 0;
    Mesh bone_mesh;
    std::vector<Action> dummy_actions;
    load("bone_data", bone_mesh, dummy_actions);
    assert(bone_mesh.vertex_count);
    Nav nav;
    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    nav_init(nav, vec3{0.5,2,3}, width, height, to_radians(90), 0.1, 100);
    bool quit = false;
    bool draw_bones = false;
    bool en_camera_locator = false;
    float en_update = 1;
    float time_dir = 1;
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
                quit = true;

            if(event.type == SDL_KEYDOWN)
            {
                switch(event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                case SDLK_1:
                    if(!obj.action)
                        break;
                    ++obj.action;
                    obj.action_time = 0;

                    if(obj.action == actions.data() + actions.size())
                        obj.action = actions.data();
                    break;
                case SDLK_2:
                    draw_bones = !draw_bones;
                    break;
                case SDLK_3:
                    en_camera_locator = !en_camera_locator;
                    break;
                case SDLK_MINUS:
                    time_dir *= -1;
                    break;
                case SDLK_SPACE:
                    en_update = en_update ? 0 : 1;
                    break;
                }
            }
            nav_process_event(nav, event);
        }
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;
        vec3 eye_pos = nav.eye_pos;
        mat4 view = nav.view;
        mat4 proj = nav.proj;
        update_anim_data(obj, dt * time_dir * en_update);

        if(en_camera_locator)
        {
            for(int bone_id = 0; bone_id < obj.mesh->bone_count; ++bone_id)
            {
                const char* name = obj.mesh->bone_names[bone_id];

                if(strcmp(name, "camera_locator") != 0)
                    continue;
                mat4 world_f_view = obj.model_tf * obj.cp_model_f_bone[bone_id];
                eye_pos = vec3{world_f_view.data[3], world_f_view.data[7], world_f_view.data[11]};
                view = invert_coord_change(world_f_view);
                break;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, view.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, obj.model_tf.data);
        assert(obj.mesh->bone_count <= MAX_BONES);
        glUniformMatrix4fv(glGetUniformLocation(program, "skinning_matrices"), obj.mesh->bone_count, GL_TRUE, obj.skinning_matrices[0].data);
        glUniform3fv(glGetUniformLocation(program, "eye_pos"), 1, &eye_pos.x);
        glBindVertexArray(obj.mesh->vao);
        glDrawElements(GL_TRIANGLES, obj.mesh->index_count, GL_UNSIGNED_INT, (void*)(uint64_t)obj.mesh->ebo_offset); // suppress warning

        if(draw_bones)
        {
            glDisable(GL_DEPTH_TEST);
            glUseProgram(program_debug);
            glUniformMatrix4fv(glGetUniformLocation(program_debug, "proj"), 1, GL_TRUE, proj.data);
            glUniformMatrix4fv(glGetUniformLocation(program_debug, "view"), 1, GL_TRUE, view.data);

            for(int i = 0; i < obj.mesh->bone_count; ++i)
            {
                mat4 model_tf = obj.model_tf * obj.cp_model_f_bone[i];
                glUniformMatrix4fv(glGetUniformLocation(program_debug, "model"), 1, GL_TRUE, model_tf.data);
                glBindVertexArray(bone_mesh.vao);
                glDrawElements(GL_TRIANGLES, bone_mesh.index_count, GL_UNSIGNED_INT, (void*)(uint64_t)bone_mesh.ebo_offset);
            }
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
