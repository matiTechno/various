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
    {
        int tid = obj.action->bone_actions[0].loc_count - 1;
        float duration = obj.action->bone_actions[0].loc_time_coords[tid];

        if(obj.action_time > duration)
            obj.action_time = min(duration, obj.action_time - duration);
    }

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
    return rotate_x(-pi/2);
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
    assert(mesh.vertex_count);
    Object obj;
    obj.mesh = &mesh;
    obj.action = actions.empty() ? nullptr : actions.data();
    obj.model_tf = rotate_y(-pi/4) * gl_from_blender();
    alloc_object_anim_data(obj, mesh);
    obj.action_time = 0;
    Mesh bone_mesh;
    std::vector<Action> dummy_actions;
    load("bone_data", bone_mesh, dummy_actions);
    assert(bone_mesh.vertex_count);
    bool quit = false;
    bool draw_bones = false;
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
                    draw_bones = !draw_bones;
                    break;
                case SDLK_2:
                    if(!obj.action)
                        break;
                    ++obj.action;
                    obj.action_time = 0;

                    if(obj.action == actions.data() + actions.size())
                        obj.action = actions.data();
                    break;
                }
            }
        }
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;
        update_anim_data(obj, dt);
        vec3 eye_pos = {0,1,2};
        mat4 proj = perspective(60, (float)width/height, 0.1, 100);
        mat4 view = lookat(eye_pos, 0, -10);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_TRUE, proj.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_TRUE, view.data);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_TRUE, obj.model_tf.data);
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
                mat4 model_tf = obj.model_tf * obj.cp_model_f_bone[i] * rotate_x(-pi/2);
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
