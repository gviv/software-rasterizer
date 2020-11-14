#include <memory>
#include <voxium.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>

#include "obj_loader.cpp"
#include "rasterizer.cpp"

// 0: Triangle
// 1: Icosphere with basic shading
#define SCENE 1

#if SCENE == 0
VERTEX_SHADER(simpleVertexShader)
{
    out[0] = vertex.position.x;
    out[1] = vertex.position.y;
    out[2] = vertex.position.z;

#if SIMD_VERTEX
    return {vertex.position, makeF32_x8(1.f)};
#else
    return {vertex.position, 1.f};
#endif
}

FRAGMENT_SHADER(simpleFragmentShader)
{
#if SIMD_FRAGMENT
    return {in[0], in[1], in[2], makeF32_x8(1.f)};
#else
    return {in[0], in[1], in[2], 1.f};
#endif
}
#elif SCENE == 1
VERTEX_SHADER(basicMaterialVertexShader)
{
#if SIMD_VERTEX
    const m4_x8& modelToWorld = *static_cast<const m4_x8*>(uniforms[0]);
    const m4_x8& worldToView  = *static_cast<const m4_x8*>(uniforms[1]);
    const m4_x8& projection   = *static_cast<const m4_x8*>(uniforms[2]);
    const m4_x8& modelToWorldNormal = *static_cast<const m4_x8*>(uniforms[3]);

    v3_x8 normalWorldSpace = multMat44Vec3(modelToWorldNormal, vertex.normal);
    v3_x8 vertexWorldSpace = multMat44Point3(modelToWorld, vertex.position);
    v3_x8 vertexViewSpace = multMat44Point3(worldToView, vertexWorldSpace);
#else
    const m4& modelToWorld = *static_cast<const m4*>(uniforms[0]);
    const m4& worldToView  = *static_cast<const m4*>(uniforms[1]);
    const m4& projection   = *static_cast<const m4*>(uniforms[2]);
    const m4& modelToWorldNormal = *static_cast<const m4*>(uniforms[3]);

    v3 normalWorldSpace = multMat44Vec3(modelToWorldNormal, vertex.normal);
    v3 vertexWorldSpace = multMat44Point3(modelToWorld, vertex.position);
    v3 vertexViewSpace = multMat44Point3(worldToView, vertexWorldSpace);
#endif


    out[0] = normalWorldSpace.x;
    out[1] = normalWorldSpace.y;
    out[2] = normalWorldSpace.z;

    out[3] = vertexWorldSpace.x;
    out[4] = vertexWorldSpace.y;
    out[5] = vertexWorldSpace.z;

#if SIMD_VERTEX
    return projection * v4_x8{vertexViewSpace, makeF32_x8(1.f)};
#else
    return projection * v4{vertexViewSpace, 1.f};
#endif
}

FRAGMENT_SHADER(basicMaterialFragmentShader)
{
#if SIMD_FRAGMENT
    const v3_x8& lightPosWorld = *static_cast<const v3_x8*>(uniforms[0]);
    v3_x8 normalWorldSpace{in[0], in[1], in[2]};
    v3_x8 vertexWorldSpace{in[3], in[4], in[5]};
    v3_x8 objColor = makeV3_x8({1.f, 1.f, 0.f});
    v3_x8 lightColor = makeV3_x8(v3{1.f});
    // TODO(gviv): I want a regular `normalize` function.
    v3_x8 lightDir = (lightPosWorld - vertexWorldSpace).normalize();
    normalWorldSpace.normalize();

    f32_x8 zero = _mm256_setzero_ps();
    v3_x8 ambient = makeV3_x8(v3{.2f});
    v3_x8 diffuse =
        objColor *
        (ambient + lightColor * _mm256_max_ps(zero, lightDir.dot(normalWorldSpace)));

    return {diffuse, makeF32_x8(1.f)};
#else
    const v3& lightPosWorld = *static_cast<const v3*>(uniforms[0]);
    v3 normalWorldSpace{in[0], in[1], in[2]};
    v3 vertexWorldSpace{in[3], in[4], in[5]};
    v3 objColor{1.f, 1.f, 0.f};
    v3 lightColor{1.f};
    // TODO(gviv): I want a regular `normalize` function.
    v3 lightDir = (lightPosWorld - vertexWorldSpace).normalize();
    normalWorldSpace.normalize();

    v3 ambient{.2f};
    v3 diffuse =
        objColor *
        (ambient + lightColor * max(0.f, lightDir.dot(normalWorldSpace)));

    return {diffuse, 1.f};
#endif
}
#endif

struct Camera
{
    v3 position;
    v3 right;
    v3 up;
    v3 forward;
};

class App : public vx::BaseApp
{
#if SCENE == 0
    Mesh triangle;
    Shader simpleShader;
#elif SCENE == 1
    Mesh icosphere;
    Shader basicMaterialShader;
    v3 modelPosition{0.f, 0.f, 0.f};
    v3 modelRotation;
    v3 modelScale{1.f, 1.f, 1.f};
    Camera camera;
#if SIMD_FRAGMENT
    v3_x8 lightPos = makeV3_x8({0.f, 10.f, 5.f});
#else
    v3 lightPos{0.f, 10.f, 5.f};
#endif
    float fovDegrees = 60.f;
    float n = .5f;
    float f = 600.f;
#endif
    std::unique_ptr<f32[]> depthBuffer;

public:
    void init()
    {
        depthBuffer = std::make_unique<f32[]>(vx::width() * vx::height());

#if SCENE == 0
        // Init triangle
        Vertex vertex0 = {
            {.25f, .25f, 0.f},
            {},
            {},
        };
        Vertex vertex1 = {
            {.75f, .25f, 0.f},
            {},
            {},
        };
        Vertex vertex2 = {
            {.25f, .75f, 0.f},
            {},
            {},
        };
        triangle.vertices.push_back(vertex0);
        triangle.vertices.push_back(vertex1);
        triangle.vertices.push_back(vertex2);
        triangle.indices.push_back(0);
        triangle.indices.push_back(1);
        triangle.indices.push_back(2);

        // Init shader
        simpleShader.vertexShader = &simpleVertexShader;
        simpleShader.fragmentShader = &simpleFragmentShader;
        simpleShader.nbVaryings = 3;
#elif SCENE == 1
        Model model;
        if (!loadObj("icosphere.obj", model)) {
            std::cerr << "Error loading OBJ file" << std::endl;
            // TODO(gviv): Add a "Voxium way" of exiting the program.
            std::exit(1);
        }
        icosphere = model.meshes[0];

        // Place the camera so we can see "something" and not some dark void
        v3 minVertex{FLT_MAX, FLT_MAX, FLT_MAX};
        v3 maxVertex{FLT_MIN, FLT_MIN, FLT_MIN};
        for (const auto& v : icosphere.vertices)
        {
            minVertex.x = min(v.position.x, minVertex.x);
            minVertex.y = min(v.position.y, minVertex.y);
            minVertex.z = min(v.position.z, minVertex.z);
            maxVertex.x = max(v.position.x, maxVertex.x);
            maxVertex.y = max(v.position.y, maxVertex.y);
            maxVertex.z = max(v.position.z, maxVertex.z);
        }

        camera.position = (minVertex + maxVertex) * .5f;
        camera.position.z = 3.f;

        camera.right = {1.f, 0.f, 0.f};
        camera.up = {0.f, 1.f, 0.f};
        camera.forward = {0.f, 0.f, 1.f};

        // Init shader
        basicMaterialShader.vertexShader = &basicMaterialVertexShader;
        basicMaterialShader.fragmentShader = &basicMaterialFragmentShader;
        basicMaterialShader.nbVaryings = 6;
#endif
    }

    void update(float dt)
    {
#if SCENE == 1
        float camSpeed = 10.f;
        modelRotation.x += dt * .5f;
        modelRotation.z += dt;

        if (vx::isDown(vx::Key::SHIFT))
        {
            camSpeed = 100.f;
        }

        if (vx::isDown(vx::Key::W))
        {
            camera.position.z -= camSpeed * dt;
        }
        else if (vx::isDown(vx::Key::S))
        {
            camera.position.z += camSpeed * dt;
        }
#endif
    }

    void draw()
    {
        // We want to directly operate on the screen's pixels, so we don't have
        // to render in some offscreen buffer first and blit it afterwards.
        u32* pixels = vx::getPixels();
        int bufferWidth = vx::width();
        int bufferHeight = vx::height();

        // Clear depth buffer
        for (int i = 0; i < bufferWidth * bufferHeight; ++i)
        {
            depthBuffer[i] = 10000.f;
        }

#if SCENE == 0
        // Render the triangle!
        render(triangle, simpleShader, pixels, bufferWidth, bufferHeight, depthBuffer.get());
#elif SCENE == 1
        // TODO(gviv): Maybe we want some helper functions that return
        // transformation matrices.
#if SIMD_VERTEX
        m4_x8 modelToWorld =
            // Translate
            makeM4_x8({1.f, 0.f, 0.f, modelPosition.x,
               0.f, 1.f, 0.f, modelPosition.y,
               0.f, 0.f, 1.f, modelPosition.z,
               0.f, 0.f, 0.f, 1.f})
            *
            // Rotate around z
            makeM4_x8({cos(modelRotation.z), -sin(modelRotation.z), 0.f, 0.f,
               sin(modelRotation.z), cos(modelRotation.z), 0.f, 0.f,
               0.f, 0.f, 1.f, 0.f,
               0.f, 0.f, 0.f, 1.f})
            *
            // Rotate around x
            makeM4_x8({1.f, 0.f, 0.f, 0.f,
               0.f, cos(modelRotation.x), -sin(modelRotation.x), 0.f,
               0.f, sin(modelRotation.x), cos(modelRotation.x), 0.f,
               0.f, 0.f, 0.f, 1.f})
            *
            // Scale
            makeM4_x8({modelScale.x, 0.f, 0.f, 0.f,
               0.f, modelScale.y, 0.f, 0.f,
               0.f, 0.f, modelScale.z, 0.f,
               0.f, 0.f, 0.f, 1.f});

        m4_x8 worldToView = fastInverse(makeM4_x8({
            camera.right.x, camera.up.x, camera.forward.x, camera.position.x,
            camera.right.y, camera.up.y, camera.forward.y, camera.position.y,
            camera.right.z, camera.up.z, camera.forward.z, camera.position.z,
            0.f, 0.f, 0.f, 1.f
        }));
#else
        m4 modelToWorld =
            // Translate
            m4{1.f, 0.f, 0.f, modelPosition.x,
               0.f, 1.f, 0.f, modelPosition.y,
               0.f, 0.f, 1.f, modelPosition.z,
               0.f, 0.f, 0.f, 1.f}
            *
            // Rotate around z
            m4{cos(modelRotation.z), -sin(modelRotation.z), 0.f, 0.f,
               sin(modelRotation.z), cos(modelRotation.z), 0.f, 0.f,
               0.f, 0.f, 1.f, 0.f,
               0.f, 0.f, 0.f, 1.f}
            *
            // Rotate around x
            m4{1.f, 0.f, 0.f, 0.f,
               0.f, cos(modelRotation.x), -sin(modelRotation.x), 0.f,
               0.f, sin(modelRotation.x), cos(modelRotation.x), 0.f,
               0.f, 0.f, 0.f, 1.f}
            *
            // Scale
            m4{modelScale.x, 0.f, 0.f, 0.f,
               0.f, modelScale.y, 0.f, 0.f,
               0.f, 0.f, modelScale.z, 0.f,
               0.f, 0.f, 0.f, 1.f};

        m4 worldToView = fastInverse(m4{
            camera.right.x, camera.up.x, camera.forward.x, camera.position.x,
            camera.right.y, camera.up.y, camera.forward.y, camera.position.y,
            camera.right.z, camera.up.z, camera.forward.z, camera.position.z,
            0.f, 0.f, 0.f, 1.f
        });
#endif

        f32 t = tan(fovDegrees * PI / 360.f) * n;
        f32 r = t * (f32)bufferWidth / bufferHeight;
        f32 l = -r;
        f32 b = -t;

        // Remaps points from [l, r][b, t][n, f] to [0, 1]^3
#if SIMD_VERTEX
        m4_x8 projection = makeM4_x8({
            n / (r - l), 0.f, l / (r - l), 0.f,
            0.f, n / (t - b), b / (t - b), 0.f,
            0.f, 0.f, f / (n - f), n * f / (n - f),
            0.f, 0.f, -1.f, 0.f
        });
#else
        m4 projection{
            n / (r - l), 0.f, l / (r - l), 0.f,
            0.f, n / (t - b), b / (t - b), 0.f,
            0.f, 0.f, f / (n - f), n * f / (n - f),
            0.f, 0.f, -1.f, 0.f
        };
#endif

        auto modelToWorldNormal = transpose(fastInverse(modelToWorld));

        // TODO(gviv): If we only pass uniforms by pointer, we can set them once
        // at startup.
        basicMaterialShader.vertexUniforms[0] = &modelToWorld;
        basicMaterialShader.vertexUniforms[1] = &worldToView;
        basicMaterialShader.vertexUniforms[2] = &projection;
        basicMaterialShader.vertexUniforms[3] = &modelToWorldNormal;

        basicMaterialShader.fragmentUniforms[0] = &lightPos;

        render(icosphere, basicMaterialShader, pixels, bufferWidth, bufferHeight, depthBuffer.get());
#endif

        // Print some debug stats
        // TODO(gviv): Add some string formatting function like Python's
        // `format` or something.
        char str[256];
        std::snprintf(str, 256, "Nb tris outside: %d", g_nbTrianglesOutside);
        vx::text(str, 0, 0);

        std::snprintf(str, 256, "Nb tris backfacing: %d", g_nbTrianglesBackfacing);
        vx::text(str, 0, 10);

        std::snprintf(str, 256, "Nb tris clipped: %d", g_nbTrianglesClipped);
        vx::text(str, 0, 20);

        std::snprintf(str, 256, "FPS: %.2f", vx::framerate());
        vx::text(str, 0, 30);

#if SCENE == 1
        std::snprintf(str, 256, "Cam pos: (%.2f, %.2f, %.2f)",
                      camera.position.x, camera.position.y, camera.position.z);
        vx::text(str, 0, 40);
#endif

        g_nbTrianglesOutside = 0;
        g_nbTrianglesBackfacing = 0;
        g_nbTrianglesClipped = 0;
    }
};

int main()
{
    App app{};
    vx::run(app, 1024, 1024, 512, 512);

    return 0;
}
