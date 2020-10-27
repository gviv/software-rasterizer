#include <memory>
#include <voxium.h>
#include <cstdio>

#include "obj_loader.cpp"
#include "rasterizer.cpp"

VERTEX_SHADER(simpleVertexShader)
{
    out[0] = vertex.position.x;
    out[1] = vertex.position.y;
    out[2] = vertex.position.z;

    return {vertex.position, 1.f};
}

FRAGMENT_SHADER(simpleFragmentShader)
{
    return {in[0], in[1], in[2], 1.f};
}

class App : public vx::BaseApp
{
    Mesh triangle;
    Shader simpleShader;
    std::unique_ptr<f32[]> depthBuffer;

public:
    void init()
    {
        depthBuffer = std::make_unique<f32[]>(vx::width() * vx::height());

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
    }

    void draw()
    {
        // Clear depth buffer
        for (int i = 0; i < vx::width() * vx::height(); ++i)
        {
            depthBuffer[i] = 10000.f;
        }

        // We want to directly operate on the screen's pixels, so we don't have
        // to render in some offscreen buffer first and blit it afterwards.
        u32* pixels = vx::getPixels();

        // Render the triangle!
        render(triangle, simpleShader, pixels, vx::width(), vx::height(), depthBuffer.get());

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
    }
};

int main()
{
    App app{};
    vx::run(app, 1024, 1024, 512, 512);

    return 0;
}
