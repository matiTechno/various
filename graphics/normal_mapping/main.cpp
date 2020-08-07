#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include "../glad.h"
#include "../main.hpp"
#include "../extern/stb_image.h"

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

const char* _src_vert = R"(
#version 330
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;

out vec3 frag_pos;
out vec2 frag_uv;
out mat3 frag_TBN;

void main()
{
    vec4 pos_w = model * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_uv = uv;
    vec3 B = tangent.w * cross(normal, tangent.xyz);
    frag_TBN = mat3(model) * mat3(tangent.xyz, B, normal);
}
)";

const char* _src_frag = R"(
#version 330
uniform vec3 light_intensity = vec3(1,1,1);
uniform float ambient_intensity = 0.01;
uniform float specular_exp = 30;
uniform vec3 eye_pos;
uniform vec3 light_dir;
uniform sampler2D sampler0;
uniform sampler2D sampler1;
uniform sampler2D sampler2;
uniform int use_normal_map;

in vec3 frag_pos;
in vec2 frag_uv;
in mat3 frag_TBN;

out vec3 out_color;

void main()
{
    vec3 diffuse_color = texture(sampler0, frag_uv).rgb;
    vec3 specular_color = texture(sampler1, frag_uv).rgb;
    vec3 N = normalize( frag_TBN * (2 * texture(sampler2, frag_uv).rgb - vec3(1)) );

    if(use_normal_map == 0)
        N = normalize(frag_TBN[2]);

    vec3 L = normalize(light_dir);
    vec3 ambient_comp = diffuse_color * ambient_intensity;
    vec3 diff_comp = diffuse_color * light_intensity * max(dot(N, L), 0);

    vec3 V = normalize(eye_pos - frag_pos);
    vec3 H = normalize(V + L);
    vec3 spec_comp = specular_color * light_intensity * pow( max(dot(N, H), 0), specular_exp) * float(dot(N, L) > 0);

    out_color = ambient_comp + diff_comp + spec_comp;
    out_color = pow(out_color, vec3(1/2.2));
}
)";

const char* _src_frag_bump = R"(
#version 330
uniform vec3 light_intensity = vec3(1,1,1);
uniform float ambient_intensity = 0.01;
uniform float specular_exp = 20;
uniform vec3 specular_color = vec3(0.2);
uniform vec3 eye_pos;
uniform vec3 light_dir;
uniform sampler2D sampler0;
uniform sampler2D sampler1;
uniform float bump_strength;
uniform int use_normal_map;

in vec3 frag_pos;
in vec2 frag_uv;
in mat3 frag_TBN;

out vec3 out_color;

void main()
{
    vec2 texel_uv_size = vec2(1) / vec2(textureSize(sampler1, 0));
    vec2 uv_dx = vec2(texel_uv_size.x, 0);
    vec2 uv_dy = vec2(0, texel_uv_size.y);

    float s0 = texture(sampler1, frag_uv + uv_dx).r;
    float s1 = texture(sampler1, frag_uv - uv_dx).r;
    float s2 = texture(sampler1, frag_uv + uv_dy).r;
    float s3 = texture(sampler1, frag_uv - uv_dy).r;

    float bump_dx = (s0 - s1) * bump_strength;
    float bump_dy = (s2 - s3) * bump_strength;
    vec3 S = vec3(1, 0, bump_dx);
    vec3 T = vec3(0, 1, bump_dy);
    vec3 N = normalize(frag_TBN * cross(S, T));

    if(use_normal_map == 0)
        N = normalize(frag_TBN[2]);

    vec3 diffuse_color = texture(sampler0, frag_uv).rgb;

    vec3 L = normalize(light_dir);
    vec3 ambient_comp = diffuse_color * ambient_intensity;
    vec3 diff_comp = diffuse_color * light_intensity * max(dot(N, L), 0);

    vec3 V = normalize(eye_pos - frag_pos);
    vec3 H = normalize(V + L);
    vec3 spec_comp = specular_color * light_intensity * pow( max(dot(N, H), 0), specular_exp) * float(dot(N, L) > 0);

    out_color = ambient_comp + diff_comp + spec_comp;
    out_color = pow(out_color, vec3(1/2.2));
}
)";

struct Vertex
{
    vec3 pos;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
};

struct Mesh
{
    GLuint bo;
    GLuint vao;
    int ebo_offset;
    int index_count;
};

enum ObjMode
{
    OBJ_OTHER,
    OBJ_POS,
    OBJ_NORM,
    OBJ_UV,
    OBJ_FACE,
};

struct VertexNode
{
    VertexNode* next;
    int norm_id;
    int uv_id;
};

// compound
struct CompId
{
    int pos_id;
    int sub_id;
};

Mesh load(const char* filename)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    char buf[256];
    std::vector<vec3> positions;
    std::vector<vec3> normals;
    std::vector<vec2> uvs;
    std::vector<VertexNode*> nodes;
    std::vector<CompId> ids;
    bool nodes_init = false;

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
                normals.push_back(normalize(v));
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
            if(!nodes_init)
            {
                nodes_init = true;
                nodes.resize(positions.size(), nullptr);
            }
            for(int i = 0; i < 3; ++i)
            {
                int pos_id, uv_id, norm_id;
                r = fscanf(file, "%d/%d/%d", &pos_id, &uv_id, &norm_id);
                assert(r == 3);
                pos_id -= 1;
                uv_id -= 1;
                norm_id -= 1;

                VertexNode** ptr = &nodes[pos_id];
                int sub_id = 0;

                while(*ptr)
                {
                    if((**ptr).norm_id == norm_id && (**ptr).uv_id == uv_id)
                        break;
                    sub_id += 1;
                    ptr = &(**ptr).next;
                }

                if(!*ptr)
                {
                    VertexNode* node = (VertexNode*)malloc(sizeof(VertexNode));
                    node->next = nullptr;
                    node->norm_id = norm_id;
                    node->uv_id = uv_id;
                    *ptr = node;
                }
                CompId id;
                id.pos_id = pos_id;
                id.sub_id = sub_id;
                ids.push_back(id);
            }
            break;
        }
        case OBJ_OTHER:
            break;
        }
    }

    std::vector<int> id_offsets;
    std::vector<Vertex> verts;
    verts.reserve(nodes.size()); // coarse reserve
    int offset = 0;
    id_offsets.reserve(nodes.size());

    for(int i = 0; i < (int)nodes.size(); ++i)
    {
        VertexNode* node = nodes[i];
        assert(node);
        id_offsets.push_back(offset);

        while(node)
        {
            Vertex vert;
            vert.pos = positions[i];
            vert.normal = normals[node->norm_id];
            vert.uv = uvs[node->uv_id];
            verts.push_back(vert);
            offset += 1;
            node = node->next;
        }
    }

    std::vector<int> indices;
    indices.reserve(ids.size());

    for(CompId id: ids)
        indices.push_back(id_offsets[id.pos_id] + id.sub_id);

    std::vector<vec3> tangents;
    std::vector<vec3> bitangents;
    tangents.resize(verts.size(), vec3{});
    bitangents.resize(verts.size(), vec3{});

    for(std::size_t base = 0; base < indices.size(); base += 3)
    {
        int face_ids[3] = {indices[base+0], indices[base+1], indices[base+2]};
        Vertex face[3];

        for(int i = 0; i < 3; ++i)
            face[i] = verts[face_ids[i]];

        vec3 dpos1 = face[1].pos - face[0].pos;
        vec3 dpos2 = face[2].pos - face[0].pos;
        vec2 duv1 = face[1].uv - face[0].uv;
        vec2 duv2 = face[2].uv - face[0].uv;
        vec2 col1 = {dpos1.x, dpos2.x};
        vec2 col2 = {dpos1.y, dpos2.y};
        vec2 col3 = {dpos1.z, dpos2.z};
        vec2 row1 = {duv2.y, -duv1.y};
        vec2 row2 = {-duv2.x, duv1.x};
        float c = 1.0 / (duv1.x * duv2.y - duv2.x * duv1.y);
        vec3 tangent =   c * vec3{dot(row1, col1), dot(row1, col2), dot(row1, col3)};
        vec3 bitangent = c * vec3{dot(row2, col1), dot(row2, col2), dot(row2, col3)};

        for(int id: face_ids)
        {
            tangents[id] = tangents[id] + tangent;
            bitangents[id] = bitangents[id] + bitangent;
        }
    }

    for(std::size_t i = 0; i < verts.size(); ++i)
    {
        vec3 T = tangents[i];
        vec3 N = verts[i].normal;
        // use this when performing lighting in a tangent space, tangent basis must be orthonormal if inverse is obtained with transpose()
        //T = normalize(T - dot(N, T) * N);
        T = normalize(T);
        float handedness = dot(cross(N, T), bitangents[i]) > 0 ? 1 : -1;
        verts[i].tangent = vec4{T.x, T.y, T.z, handedness};
    }

    Mesh mesh;
    mesh.index_count = indices.size();
    mesh.ebo_offset = verts.size() * sizeof(Vertex);
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.bo);
    glBindVertexArray(mesh.bo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.bo);
    glBufferData(GL_ARRAY_BUFFER, mesh.ebo_offset + indices.size() * sizeof(int), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, mesh.ebo_offset, verts.data());
    glBufferSubData(GL_ARRAY_BUFFER, mesh.ebo_offset, indices.size() * sizeof(int), indices.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.bo);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
    printf("vertices: %d\n", (int)verts.size());
    printf("indices:  %d\n", (int)indices.size());
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

#define LOAD_SRGB 1
#define LOAD_RGB 0

GLuint load_texture(const char* name, bool srgb)
{
    GLint internalFmt = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    stbi_set_flip_vertically_on_load(1);
    int x, y, dum;
    u8* data = stbi_load(name, &x, &y, &dum, 4);
    assert(data);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    stbi_image_free(data);
    return tex;
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

    GLuint program = create_program(_src_vert, _src_frag);
    GLuint program_bump = create_program(_src_vert, _src_frag_bump);

    Mesh mesh = load("res/model.obj");
    GLuint tex_diff = load_texture("res/diffuse.jpg", LOAD_SRGB);
    GLuint tex_spec = load_texture("res/specular.jpg", LOAD_SRGB);
    GLuint tex_norm = load_texture("res/normal.jpg", LOAD_RGB);

    Mesh mesh_bump = load("res_bump/model.obj");
    GLuint tex_diff_bump = load_texture("res_bump/diffuse.jpg", LOAD_SRGB);
    GLuint tex_bump = load_texture("res_bump/bump.png", LOAD_SRGB);

    bool quit = false;
    bool rotate = false;
    int use_normal_map = 1;
    bool use_bump_model = false;
    mat4 model_tf = identity4();
    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    Nav nav;
    nav_init(nav, vec3{0,0,2.2}, width, height, to_radians(90), 0.01, 1000);
    Uint64 prev_counter = SDL_GetPerformanceCounter();

    while(!quit)
    {
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                quit = true;
            else if(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r)
                rotate = !rotate;
            else if(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_n)
                use_normal_map = !use_normal_map;
            else if(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_1)
                use_bump_model = !use_bump_model;

            nav_process_event(nav, event);
        }
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;

        if(rotate)
            model_tf = rotate_y4(dt * 2*pi/10) * model_tf;

        vec3 light_dir = normalize(nav.eye_pos - nav.center);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(!use_bump_model)
        {
            glActiveTexture(GL_TEXTURE0 + 0);
            glBindTexture(GL_TEXTURE_2D, tex_diff);
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTexture(GL_TEXTURE_2D, tex_spec);
            glActiveTexture(GL_TEXTURE0 + 2);
            glBindTexture(GL_TEXTURE_2D, tex_norm);

            glUseProgram(program);
            glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, model_tf.data);
            glUniform3fv(glGetUniformLocation(program, "eye_pos"), 1, &nav.eye_pos.x);
            glUniform3fv(glGetUniformLocation(program, "light_dir"), 1, &light_dir.x);
            glUniform1i(glGetUniformLocation(program, "sampler0"), 0);
            glUniform1i(glGetUniformLocation(program, "sampler1"), 1);
            glUniform1i(glGetUniformLocation(program, "sampler2"), 2);
            glUniform1i(glGetUniformLocation(program, "use_normal_map"), use_normal_map);

            glBindVertexArray(mesh.vao);
            glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, (const void*)(uint64_t)mesh.ebo_offset);
        }
        else
        {
            glActiveTexture(GL_TEXTURE0 + 0);
            glBindTexture(GL_TEXTURE_2D, tex_diff_bump);
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTexture(GL_TEXTURE_2D, tex_bump);

            glUseProgram(program_bump);
            glUniformMatrix4fv(glGetUniformLocation(program_bump, "proj"), 1, GL_TRUE, nav.proj.data);
            glUniformMatrix4fv(glGetUniformLocation(program_bump, "view"), 1, GL_TRUE, nav.view.data);
            glUniformMatrix4fv(glGetUniformLocation(program_bump, "model"), 1, GL_TRUE, model_tf.data);
            glUniform3fv(glGetUniformLocation(program_bump, "eye_pos"), 1, &nav.eye_pos.x);
            glUniform3fv(glGetUniformLocation(program_bump, "light_dir"), 1, &light_dir.x);
            glUniform1i(glGetUniformLocation(program_bump, "sampler0"), 0);
            glUniform1i(glGetUniformLocation(program_bump, "sampler1"), 1);
            glUniform1i(glGetUniformLocation(program_bump, "use_normal_map"), use_normal_map);
            glUniform1f(glGetUniformLocation(program_bump, "bump_strength"), 10);

            glBindVertexArray(mesh_bump.vao);
            glDrawElements(GL_TRIANGLES, mesh_bump.index_count, GL_UNSIGNED_INT, (const void*)(uint64_t)mesh_bump.ebo_offset);
        }

        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
