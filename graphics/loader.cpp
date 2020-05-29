#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "main.hpp"

void load(vec3*& out_positions, vec3*& out_normals, int& out_vertex_count)
{
    const aiScene* scene = aiImportFile("model.obj", 0);

    if(!scene)
    {
        printf("failed to load a model\n");
        exit(1);
    }
    assert(scene->HasMeshes());
    aiMesh& mesh = *scene->mMeshes[0];
    assert(mesh.HasFaces());
    assert(mesh.HasNormals());
    std::vector<vec3> positions;
    std::vector<vec3> normals;
    positions.reserve(3*mesh.mNumFaces);
    normals.reserve(3*mesh.mNumFaces);

    for(int fid = 0; fid < (int)mesh.mNumFaces; ++fid)
    {
        aiFace face = mesh.mFaces[fid];
        assert(face.mNumIndices == 3);

        for(int i = 0; i < 3; ++i)
        {
            int idx = face.mIndices[i];
            aiVector3D vert = mesh.mVertices[idx];
            aiVector3D normal = mesh.mNormals[idx];
            positions.push_back(vec3{vert.x, vert.y, vert.z});
            normals.push_back(vec3{normal.x, normal.y, normal.z});
        }
    }
    int bytes = sizeof(vec3) * positions.size();
    out_positions = (vec3*)malloc(bytes);
    out_normals = (vec3*)malloc(bytes);
    memcpy(out_positions, positions.data(), bytes);
    memcpy(out_normals, normals.data(), bytes);
    out_vertex_count = positions.size();
    aiReleaseImport(scene);
}
