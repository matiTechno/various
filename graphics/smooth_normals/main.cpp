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
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
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

const char* _src_frag = R"(
#version 330
uniform vec3 light_intensity = vec3(1,1,1);
uniform float ambient_intensity = 0.01;
uniform float specular_exp = 30;
uniform vec3 diffuse_color = vec3(0.5);
uniform vec3 specular_color = vec3(0.2,0.2,0);
uniform vec3 eye_pos;
uniform vec3 light_dir;

in vec3 frag_pos;
in vec3 frag_normal;
out vec3 out_color;

void main()
{
    vec3 N = normalize(frag_normal);
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
    OBJ_FACE,
};

struct VertexNode
{
    VertexNode* next;
    vec3 normal;
};

// compound
struct CompId
{
    int pos_id;
    int sub_id;
};

void load_smooth(FILE* file, std::vector<Vertex>& verts, std::vector<int>& indices)
{
    char buf[256];

    for(;;)
    {
        int r = fscanf(file, "%s", buf);

        if(r == EOF)
            break;
        assert(r == 1);
        ObjMode mode = OBJ_OTHER;

        if(strcmp(buf, "v") == 0)
            mode = OBJ_POS;
        else if(strcmp(buf, "f") == 0)
            mode = OBJ_FACE;

        switch(mode)
        {
        case OBJ_POS:
        {
            Vertex vert;
            vert.normal = {};
            r = fscanf(file, "%f %f %f", &vert.pos.x, &vert.pos.y, &vert.pos.z);
            assert(r == 3);
            verts.push_back(vert);
            break;
        }
        case OBJ_FACE:
        {
            for(int i = 0; i < 3; ++i)
            {
                int vert_id, dum1, dum2;
                r = fscanf(file, "%d/%d/%d", &vert_id, &dum1, &dum2);
                assert(r == 3);
                indices.push_back(vert_id - 1);
            }
            break;
        }
        case OBJ_OTHER:
            break;
        }
    }

    for(std::size_t base = 0; base < indices.size(); base += 3)
    {
        Vertex& v0 = verts[indices[base+0]];
        Vertex& v1 = verts[indices[base+1]];
        Vertex& v2 = verts[indices[base+2]];
        vec3 area_vector = cross(v1.pos - v0.pos, v2.pos - v0.pos);
        // triangles with a bigger area have more influence on a vertex normal (area_vector is not normalized)
        v0.normal = v0.normal + area_vector;
        v1.normal = v1.normal + area_vector;
        v2.normal = v2.normal + area_vector;
    }

    for(std::size_t i = 0; i < verts.size(); ++i)
        verts[i].normal = normalize(verts[i].normal);
}

void load_flat(FILE* file, std::vector<Vertex>& verts, std::vector<int>& indices)
{
    std::vector<vec3> positions;
    std::vector<VertexNode*> nodes;
    std::vector<CompId> comp_ids;
    char buf[256];
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
        else if(strcmp(buf, "f") == 0)
            mode = OBJ_FACE;

        switch(mode)
        {
        case OBJ_POS:
        {
            vec3 v;
            r = fscanf(file, "%f %f %f", &v.x, &v.y, &v.z);
            assert(r == 3);
            positions.push_back(v);
            break;
        }
        case OBJ_FACE:
        {
            if(!nodes_init)
            {
                nodes_init = true;
                nodes.resize(positions.size(), nullptr);
            }

            int pos_ids[3];

            for(int i = 0; i < 3; ++i)
            {
                int pos_id, dum1, dum2;
                r = fscanf(file, "%d/%d/%d", &pos_id, &dum1, &dum2);
                assert(r == 3);
                pos_ids[i] = pos_id - 1;
            }
            vec3 pos0 = positions[pos_ids[0]];
            vec3 pos1 = positions[pos_ids[1]];
            vec3 pos2 = positions[pos_ids[2]];
            vec3 normal = normalize(cross(pos1 - pos0, pos2 - pos0));

            for(int pos_id: pos_ids)
            {
                VertexNode** ptr = &nodes[pos_id];
                int sub_id = 0;

                while(*ptr)
                {
                    if(dot(normal, (**ptr).normal) > cosf(to_radians(1))) // merge vertices with close normals
                        break;
                    sub_id += 1;
                    ptr = &(**ptr).next;
                }

                if(!*ptr)
                {
                    VertexNode* node = (VertexNode*)malloc(sizeof(VertexNode));
                    node->next = nullptr;
                    node->normal = normal;
                    *ptr = node;
                }
                CompId id;
                id.pos_id = pos_id;
                id.sub_id = sub_id;
                comp_ids.push_back(id);
            }
            break;
        }
        case OBJ_OTHER:
            break;
        }
    }

    std::vector<int> offsets;
    offsets.reserve(nodes.size());
    verts.reserve(nodes.size()); // coarse reserve
    int offset = 0;

    for(std::size_t i = 0; i < nodes.size(); ++i)
    {
        VertexNode* node = nodes[i];
        assert(node);
        offsets.push_back(offset);

        while(node)
        {
            Vertex vert;
            vert.pos = positions[i];
            vert.normal = node->normal;
            verts.push_back(vert);
            offset += 1;
            node = node->next;
        }
    }

    indices.reserve(comp_ids.size());

    for(CompId id: comp_ids)
        indices.push_back(offsets[id.pos_id] + id.sub_id);
}

#define LOAD_SMOOTH 1
#define LOAD_FLAT 0

Mesh load(const char* filename, bool smooth)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    std::vector<Vertex> verts;
    std::vector<int> indices;

    if(smooth)
        load_smooth(file, verts, indices);
    else
        load_flat(file, verts, indices);

    fclose(file);

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
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
    Mesh mesh_smooth = load("../texture_paint/suzanne.obj", LOAD_SMOOTH);
    Mesh mesh_flat = load("../texture_paint/suzanne.obj", LOAD_FLAT);
    Mesh* mesh = &mesh_smooth;
    vec3 eye_pos = {0,0,3};
    mat4 view = lookat(eye_pos, vec3{0,0,-1});
    vec3 light_dir = {0,0,1};
    bool quit = false;
    bool rotate = false;
    mat4 model_tf = identity4();
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
                mesh = mesh == &mesh_smooth ? &mesh_flat : &mesh_smooth;
        }
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;
        mat4 proj = perspective(to_radians(90), (float)width/height, 0.1, 100);

        if(rotate)
            model_tf = rotate_y4(dt * 2*pi/10) * model_tf;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, view.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, model_tf.data);
        glUniform3fv(glGetUniformLocation(program, "eye_pos"), 1, &eye_pos.x);
        glUniform3fv(glGetUniformLocation(program, "light_dir"), 1, &light_dir.x);
        glBindVertexArray(mesh->vao);
        glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, (const void*)(uint64_t)mesh->ebo_offset);
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
