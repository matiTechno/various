#include <SDL2/SDL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <vector>
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
layout(location = 2) in vec4 bone_id;
layout(location = 3) in vec4 bone_weight;

// in world space
out vec3 frag_pos;
out vec3 frag_normal;

void main()
{
    mat4 cp_from_bp = mat4(0);
    cp_from_bp += skinning_matrices[int(bone_id.x)] * bone_weight.x;
    cp_from_bp += skinning_matrices[int(bone_id.y)] * bone_weight.y;
    cp_from_bp += skinning_matrices[int(bone_id.z)] * bone_weight.z;
    cp_from_bp += skinning_matrices[int(bone_id.w)] * bone_weight.w;

    mat4 model2 = model * cp_from_bp;

    vec4 pos_w = model2 * vec4(pos, 1);
    gl_Position = proj * view * pos_w;
    frag_pos = vec3(pos_w);
    frag_normal = mat3(model2) * normal;
}
)";

static const char* src_frag = R"(
#version 330

uniform vec3 light_intensity;
uniform vec3 light_dir;
uniform float ambient_intensity;
uniform vec3 diffuse_color;
uniform vec3 specular_color;
uniform float specular_exp;
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
{ out_color = vec3(1, 0, 0); }
)";

struct RenderCmd
{
    mat4 view;
    mat4 proj;
    vec3 light_intensity;
    vec3 light_dir;
    float ambient_intensity;
    vec3 eye_pos;

    // model data
    mat4 model_transform;
    vec3 diffuse_color;
    vec3 specular_color;
    float specular_exp;
    bool debug; // no depth test, no face culling, no skinning, solid color
    vec3* positions;
    vec3* normals;
    vec4* bone_ids;
    vec4* weights;
    int vertex_count;

    // bp - bind pose
    // cp - current pose

    // bone data
    int* parent_ids;
    mat4* bp_model_from_bone;
    int bone_count;

    // animation data
    // [bone_id * sample_count][sample_id]
    // these are parent-relative transformations (bone to parent space)
    vec3* translations;
    vec4* rotations;
    int sample_count;

    // animation runtime data
    mat4* skinning_matrices; // tranforms coordinates in a model space from a bind pose to a current pose
    mat4* cp_model_from_bone;
    float anim_time;
    float anim_duration;
};

void load_model(const char* filename, RenderCmd& cmd)
{
    FILE* file = fopen(filename, "r");
    assert(file);
    std::vector<vec3> positions;
    std::vector<vec3> normals;
    std::vector<vec4> bone_ids;
    std::vector<vec4> weights;
    std::vector<int> indices;
    std::vector<int> parent_ids;
    std::vector<mat4> bp_model_from_bone;
    std::vector<vec3> translations;
    std::vector<vec4> rotations;
    bool done = false;

    while(!done)
    {
        int code = fgetc(file);

        switch(code)
        {
        case 'v':
        {
            vec3 v;
            int n = fscanf(file, " %f %f %f ", &v.x, &v.y, &v.z);
            assert(n == 3);
            positions.push_back(v);
            break;
        }
        case 'n':
        {
            vec3 v;
            int n = fscanf(file, " %f %f %f ", &v.x, &v.y, &v.z);
            assert(n == 3);
            normals.push_back(v);
            break;
        }
        case 'w':
        {
            vec4 bone;
            vec4 weight;
            int n = fscanf(file, " %f %f %f %f %f %f %f %f ", &bone.x, &weight.x, &bone.y, &weight.y,
                &bone.z, &weight.z, &bone.w, &weight.w);
            assert(n == 8);
            bone_ids.push_back(bone);
            weights.push_back(weight);
            break;
        }
        case 'f':
        {
            for(int i = 0; i < 3; ++i)
            {
                int id;
                int n = fscanf(file, " %d ", &id);
                assert(n == 1);
                indices.push_back(id);
            }
            break;
        }
        case 'b':
        {
            int pid;
            mat4 m;
            int n = fscanf(file, " %d ", &pid);
            assert(n == 1);

            for(int i = 0; i < 16; ++i)
            {
                n = fscanf(file, " %f ", m.data + i);
                assert(n == 1);
            }
            parent_ids.push_back(pid);
            bp_model_from_bone.push_back(m);
            break;
        }
        case 's':
        {
            vec3 tr;
            vec4 rot;
            int n;
            n = fscanf(file, " %f %f %f ", &tr.x, &tr.y, &tr.z);
            assert(n == 3);
            n = fscanf(file, " %f %f %f %f ", &rot.w, &rot.x, &rot.y, &rot.z);
            translations.push_back(tr);
            rotations.push_back(rot);
            break;
        }
        default:
            assert(code == EOF);
            done = true;
            break;
        }
    }
    assert(positions.size());
    cmd.vertex_count = indices.size();
    assert(cmd.vertex_count % 3 == 0);
    cmd.positions = (vec3*)malloc(sizeof(vec3) * cmd.vertex_count);
    cmd.normals = (vec3*)malloc(sizeof(vec3) * cmd.vertex_count);
    cmd.bone_ids = (vec4*)malloc(sizeof(vec4) * cmd.vertex_count);
    cmd.weights = (vec4*)malloc(sizeof(vec4) * cmd.vertex_count);

    for(int i = 0; i < cmd.vertex_count; ++i)
    {
        int id = indices[i];
        cmd.positions[i] = positions[id];
        cmd.normals[i] = normals[id];
        cmd.bone_ids[i] = bone_ids[id];
        cmd.weights[i] = weights[id];
    }

    cmd.bone_count = parent_ids.size();
    cmd.parent_ids = (int*)malloc(sizeof(int) * cmd.bone_count);
    cmd.bp_model_from_bone = (mat4*)malloc(sizeof(mat4) * cmd.bone_count);

    for(int i = 0; i < cmd.bone_count; ++i)
    {
        cmd.parent_ids[i] = parent_ids[i];
        cmd.bp_model_from_bone[i] = bp_model_from_bone[i];
    }

    cmd.sample_count = translations.size() / cmd.bone_count;
    cmd.translations = (vec3*)malloc(sizeof(vec3) * translations.size());
    cmd.rotations = (vec4*)malloc(sizeof(vec4) * translations.size());

    for(int i = 0; i < (int)translations.size(); ++i)
    {
        cmd.translations[i] = translations[i];
        cmd.rotations[i] = rotations[i];
    }

    cmd.anim_time = 0.f;
    cmd.anim_duration = 5.f;
    cmd.skinning_matrices = (mat4*)malloc(sizeof(mat4) * cmd.bone_count);
    cmd.cp_model_from_bone = (mat4*)malloc(sizeof(mat4) * cmd.bone_count);
}

void update_skinning_matrices(RenderCmd& cmd, float dt)
{
    cmd.anim_time += dt;

    if(cmd.anim_time >= cmd.anim_duration)
        cmd.anim_time -= cmd.anim_duration;

    float frames_progress = (cmd.anim_time / cmd.anim_duration) * (cmd.sample_count - 1);
    int sample_lhs_id = (int)frames_progress;
    float t = frames_progress - sample_lhs_id;

    for(int i = 0; i < cmd.bone_count; ++i)
    {
        int base = i * cmd.sample_count;
        vec3 tr_lhs = cmd.translations[base + sample_lhs_id];
        vec3 tr_rhs = cmd.translations[base + sample_lhs_id + 1];
        vec4 rot_lhs = cmd.rotations[base + sample_lhs_id];
        vec4 rot_rhs = cmd.rotations[base + sample_lhs_id + 1];

        // interpolate through the shorter path
        if(dot(rot_lhs, rot_rhs) < 0)
            rot_lhs = -1 * rot_lhs;

        vec3 tr = (1-t)*tr_lhs + t*tr_rhs;
        vec4 rot = (1-t)*rot_lhs + t*rot_rhs;
        rot = normalize(rot); // linear interpolation does not preserve length (quat_to_mat4() requires a unit quaternion)
        mat4 parent_from_bone = translate(tr) * quat_to_mat4(rot);
        mat4 cp_model_from_parent = identity4();

        if(i > 0)
        {
            int pid = cmd.parent_ids[i];
            assert(pid < i);
            cp_model_from_parent = cmd.cp_model_from_bone[pid];
        }
        mat4 cp_model_from_bone = cp_model_from_parent * parent_from_bone;
        mat4 bone_from_bp_model = invert_coord_change(cmd.bp_model_from_bone[i]);
        cmd.skinning_matrices[i] = cp_model_from_bone * bone_from_bp_model;
        cmd.cp_model_from_bone[i] = cp_model_from_bone;
    }
}

GLuint _program;
GLuint _program_debug;

void gl_draw(RenderCmd& cmd)
{
    if(cmd.debug)
    {
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
    }
    else
    {
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
    }

    GLuint vbo;
    GLuint vao;
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cmd.vertex_count * (2*sizeof(vec3) + 2*sizeof(vec4)), nullptr, GL_STREAM_DRAW);

    GLintptr offset = 0;
    int bytes3 = sizeof(vec3) * cmd.vertex_count;
    int bytes4 = sizeof(vec4) * cmd.vertex_count;

    glBufferSubData(GL_ARRAY_BUFFER, offset, bytes3, cmd.positions);
    offset += bytes3;

    if(!cmd.debug)
    {
        glBufferSubData(GL_ARRAY_BUFFER, offset, bytes3, cmd.normals);
        offset += bytes3;
        glBufferSubData(GL_ARRAY_BUFFER, offset, bytes4, cmd.bone_ids);
        offset += bytes4;
        glBufferSubData(GL_ARRAY_BUFFER, offset, bytes4, cmd.weights);
    }

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    offset = 0;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)offset);
    offset += bytes3;
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)offset);
    offset += bytes3;
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)offset);
    offset += bytes4;
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, (void*)offset);

    GLuint program = cmd.debug ? _program_debug : _program;
    glUseProgram(program);

    GLint view_loc = glGetUniformLocation(program, "view");
    GLint proj_loc = glGetUniformLocation(program, "proj");
    GLint model_loc = glGetUniformLocation(program, "model");
    glUniformMatrix4fv(view_loc, 1, GL_TRUE, cmd.view.data);
    glUniformMatrix4fv(proj_loc, 1, GL_TRUE, cmd.proj.data);
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, cmd.model_transform.data);

    if(!cmd.debug)
    {
        GLint light_int_loc = glGetUniformLocation(program, "light_intensity");
        GLint light_dir_loc = glGetUniformLocation(program, "light_dir");
        GLint ambient_int_loc = glGetUniformLocation(program, "ambient_intensity");
        GLint eye_pos_loc = glGetUniformLocation(program, "eye_pos");
        GLint diffuse_color_loc = glGetUniformLocation(program, "diffuse_color");
        GLint specular_color_loc = glGetUniformLocation(program, "specular_color");
        GLint specular_exp_loc = glGetUniformLocation(program, "specular_exp");
        GLint skinning_matrices_loc = glGetUniformLocation(program, "skinning_matrices");
        glUniform3fv(light_int_loc, 1, &cmd.light_intensity.x);
        glUniform3fv(light_dir_loc, 1, &cmd.light_dir.x);
        glUniform1f(ambient_int_loc, cmd.ambient_intensity);
        glUniform3fv(eye_pos_loc, 1, &cmd.eye_pos.x);
        glUniform3fv(diffuse_color_loc, 1, &cmd.diffuse_color.x);
        glUniform3fv(specular_color_loc, 1, &cmd.specular_color.x);
        glUniform1f(specular_exp_loc, cmd.specular_exp);
        glUniformMatrix4fv(skinning_matrices_loc, cmd.bone_count, GL_TRUE, cmd.skinning_matrices[0].data);
    }

    glDrawArrays(GL_TRIANGLES, 0, cmd.vertex_count);
    // note: this is super important
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

RenderCmd test_triangle()
{
    vec3* positions = (vec3*)malloc(3 * sizeof(vec3));
    positions[0] = {-0.3,0,0};
    positions[1] = {0.3,0,0};
    positions[2] = {0,1,0};
    RenderCmd cmd;
    cmd.positions = positions;
    cmd.vertex_count = 3;
    cmd.model_transform = identity4();
    cmd.debug = true;
    return cmd;
}

mat4 gl_from_blender()
{
    return rotate_x(-pi/2) * rotate_z(-pi/2);
}

int main()
{
    if(SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        assert(false);
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    //SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 1000, 1000, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    SDL_Window* window = SDL_CreateWindow("demo", 0, 0, 1000, 1000, SDL_WINDOW_OPENGL);
    assert(window);
    SDL_GLContext context =  SDL_GL_CreateContext(window);
    assert(context);

    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
    {
        assert(false);
    }

    _program = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(_program, vert_shader);
        glAttachShader(_program, frag_shader);
        glLinkProgram(_program);
    }
    _program_debug = glCreateProgram();
    {
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vert_shader, 1, &src_vert_deb, nullptr);
        glCompileShader(vert_shader);
        glShaderSource(frag_shader, 1, &src_frag_deb, nullptr);
        glCompileShader(frag_shader);
        glAttachShader(_program_debug, vert_shader);
        glAttachShader(_program_debug, frag_shader);
        glLinkProgram(_program_debug);
    }
    bool quit = false;
    bool enable_move = false;
    bool draw_debug_bones = false;
    vec3 camera_pos = {0.f, 1.f, 2.f};
    float pitch = 0;
    float yaw = 0;
    Uint64 prev_counter = SDL_GetPerformanceCounter();
    int active_id = 0;
    RenderCmd cmd[3];

    cmd[0].light_intensity = {0.4, 0.4, 0.4};
    cmd[0].light_dir = normalize(vec3{0, 0.5, 1});
    cmd[0].ambient_intensity = 0.01;
    cmd[0].model_transform = identity4();
    cmd[0].diffuse_color = {0.6, 0.3, 0};
    cmd[0].specular_color = {1, 0, 0};
    cmd[0].specular_exp = 60;
    cmd[0].debug = false;
    cmd[0].model_transform = rotate_y(pi/4) * gl_from_blender();

    cmd[1] = cmd[0];
	cmd[2] = cmd[0];
    load_model("anim_data_human", cmd[0]);
    load_model("anim_data_squat", cmd[1]);
    load_model("anim_data_snake", cmd[2]);

    RenderCmd cmd_deb = test_triangle();
    cmd_deb.model_transform = scale({0.05,0.05,0.05});

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
                    draw_debug_bones = !draw_debug_bones;
                    break;
                case SDLK_2:
                    int count = sizeof(cmd) / sizeof(RenderCmd);
                    active_id += 1;

                    if(active_id == count)
                        active_id = 0;
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

        Uint64 current_counter = SDL_GetPerformanceCounter();
        float dt = (current_counter - prev_counter) / (double)SDL_GetPerformanceFrequency();
        prev_counter = current_counter;

        RenderCmd& cmd_active = cmd[active_id];

        update_skinning_matrices(cmd_active, dt);

        cmd_active.view = lookat(camera_pos, yaw, pitch);
        cmd_active.proj = perspective(60, (float)width/height, 0.1f, 100.f);
        cmd_active.eye_pos = camera_pos;

        cmd_deb.view = cmd_active.view;
        cmd_deb.proj = cmd_active.proj;
        cmd_deb.eye_pos = cmd_active.eye_pos;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl_draw(cmd_active);

        if(draw_debug_bones)
        {
            for(int i = 0; i < cmd_active.bone_count; ++i)
            {
                RenderCmd cmd_bone = cmd_deb;
                cmd_bone.model_transform = cmd_active.model_transform * cmd_active.cp_model_from_bone[i] * cmd_bone.model_transform;
                gl_draw(cmd_bone);
            }
        }
        SDL_GL_SwapWindow(window);
    }
    SDL_Quit();
    return 0;
}
