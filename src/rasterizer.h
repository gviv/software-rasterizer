#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <vector>

#include "math_util.h"

struct Bitmap
{
    u32* pixels;
    i32 width;
    i32 height;
};

// The vertex layout is fixed for now.
struct Vertex
{
    v3 position;
    v2 texCoords;
    v3 normal;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
};

struct Model
{
    std::vector<Mesh> meshes;
};

#define VERTEX_SHADER(name) v4 name(const Vertex& vertex, const void** uniforms, f32* out)
typedef VERTEX_SHADER(VertexShader);

#define FRAGMENT_SHADER(name) v4 name(const void** uniforms, f32* in)
typedef FRAGMENT_SHADER(FragmentShader);

constexpr int NB_MAX_VARYINGS = 32;
constexpr int NB_MAX_UNIFORMS = 16;

struct Shader
{
    VertexShader* vertexShader;
    const void* vertexUniforms[NB_MAX_UNIFORMS];

    FragmentShader* fragmentShader;
    const void* fragmentUniforms[NB_MAX_UNIFORMS];

    // Number of values passed from the vertex shader to the fragment shader.
    int nbVaryings;
};

#endif
