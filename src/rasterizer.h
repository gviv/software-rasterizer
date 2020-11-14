#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <vector>

#include "math_util.h"

// Enable or disable SIMD for the vertex/fragment processing.
// If SIMD_VERTEX is on, SIMD_FRAGMENT must be on too.
#define SIMD_VERTEX   1
#define SIMD_FRAGMENT 1

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

struct Vertex_x8
{
    v3_x8 position;
    v2_x8 texCoords;
    v3_x8 normal;
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

#if SIMD_VERTEX
#define VERTEX_SHADER(name) v4_x8 name(const Vertex_x8& vertex, const void** uniforms, f32_x8* out)
#else
#define VERTEX_SHADER(name) v4 name(const Vertex& vertex, const void** uniforms, f32* out)
#endif
typedef VERTEX_SHADER(VertexShader);

#if SIMD_FRAGMENT
#define FRAGMENT_SHADER(name) v4_x8 name(const void** uniforms, f32_x8* in)
#else
#define FRAGMENT_SHADER(name) v4 name(const void** uniforms, f32* in)
#endif
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

void render(
    Mesh& mesh,
    Shader shader,
    u32* pixels,
    i32 bufferWidth,
    i32 bufferHeight,
    f32* depthBuffer);

#endif
