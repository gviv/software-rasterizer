#include <assert.h>

#include "rasterizer.h"

int g_nbTrianglesOutside = 0;
int g_nbTrianglesBackfacing = 0;
int g_nbTrianglesClipped = 0;

template<typename T> inline
T bary(const T& v0, const T& v1MinusV0, const T& v2MinusV0, f32 w1, f32 w2)
{
    return v0 + w1 * v1MinusV0 + w2 * v2MinusV0;
}

void rasterize(
    Shader& shader,
    u32* pixels,
    f32* vertexShaderOutput[3],
    i32 bufferWidth,
    i32 bufferHeight,
    f32* depthBuffer,
    v4 verticesHomogeneousClipSpace[3])
{
    v2i verticesScreenSpace[3];
    i32 maxWidth = bufferWidth - 1;
    i32 maxHeight = bufferHeight - 1;

    for (int i = 0; i < 3; ++i)
    {
        f32& w = verticesHomogeneousClipSpace[i].w;

        // TODO(gviv): Test division by zero?
        w = 1.f / w;
        verticesHomogeneousClipSpace[i].x *= w;
        verticesHomogeneousClipSpace[i].y *= w;
        verticesHomogeneousClipSpace[i].z *= w;

        // We are now in the canonical view volume (a.k.a. normalized
        // device coordinates) ([0, 1]^3).
        verticesScreenSpace[i] = {
            (i32)((verticesHomogeneousClipSpace[i].x) * (f32)maxWidth + .5f),
            (i32)((1.f - verticesHomogeneousClipSpace[i].y) * (f32)maxHeight + .5f)
        };
    }

    // NOTE: `verticesHomogeneousClipSpace` now represents the vertices in
    // normalized devices coordinates!

    // Compute the double area of the triangle. If it's negative,
    // the triangle is backfacing.
    int doubleTriangleArea =
        (verticesScreenSpace[0].x - verticesScreenSpace[2].x) *
        (verticesScreenSpace[2].y - verticesScreenSpace[1].y) -
        (verticesScreenSpace[2].x - verticesScreenSpace[1].x) *
        (verticesScreenSpace[0].y - verticesScreenSpace[2].y);

    if (doubleTriangleArea <= 0)
    {
        ++g_nbTrianglesBackfacing;
        return;
    }
    f32 oneOverDoubleArea = 1.f / doubleTriangleArea;
    v2i bbMin{maxWidth, maxHeight};
    v2i bbMax{0, 0};

    // Compute the triangle's bounding box
    bbMin.x = min(
        verticesScreenSpace[0].x,
        min(verticesScreenSpace[1].x, verticesScreenSpace[2].x));
    bbMin.y = min(
        verticesScreenSpace[0].y,
        min(verticesScreenSpace[1].y, verticesScreenSpace[2].y));
    bbMax.x = max(
        verticesScreenSpace[0].x,
        max(verticesScreenSpace[1].x, verticesScreenSpace[2].x));
    bbMax.y = max(
        verticesScreenSpace[0].y,
        max(verticesScreenSpace[1].y, verticesScreenSpace[2].y));

    // Clip to screen
    if (bbMin.x < 0)         bbMin.x = 0;
    if (bbMin.y < 0)         bbMin.y = 0;
    if (bbMax.x > maxWidth)  bbMax.x = maxWidth;
    if (bbMax.y > maxHeight) bbMax.y = maxHeight;

    f32 baryData[NB_MAX_VARYINGS][3];
    const f32* vertexShaderOutput0Ptr = vertexShaderOutput[0];
    const f32* vertexShaderOutput1Ptr = vertexShaderOutput[1];
    const f32* vertexShaderOutput2Ptr = vertexShaderOutput[2];

    f32 oneOverZView0 = verticesHomogeneousClipSpace[0].w;
    f32 oneOverZView1 = verticesHomogeneousClipSpace[1].w;
    f32 oneOverZView2 = verticesHomogeneousClipSpace[2].w;

    f32 oneOverZView0DoubleArea = oneOverZView0 * oneOverDoubleArea;
    f32 oneOverZView1DoubleArea = oneOverZView1 * oneOverDoubleArea;
    f32 oneOverZView2DoubleArea = oneOverZView2 * oneOverDoubleArea;

    for (int attributeIndex = 0;
         attributeIndex < shader.nbVaryings;
         ++attributeIndex)
    {
        // Predivide every attributes for perspective-correct interpolation.
        f32 v0 = *vertexShaderOutput0Ptr++ * oneOverZView0;
        f32 v1 = *vertexShaderOutput1Ptr++ * oneOverZView1;
        f32 v2 = *vertexShaderOutput2Ptr++ * oneOverZView2;

        // We multiply by oneOverDoubleArea to normalize the barycentric
        // weights that will be used in the bary interpolation.
        baryData[attributeIndex][0] = v0;
        baryData[attributeIndex][1] = (v1 - v0) * oneOverDoubleArea;
        baryData[attributeIndex][2] = (v2 - v0) * oneOverDoubleArea;
    }

    f32 fragmentShaderInput[NB_MAX_VARYINGS];
    // NOTE: Since these terms are used for linear interpolation, we don't need
    // to multiply them by oneOverZView, unlike what we do for perspective-
    // correct interpolation.
    f32 zNDC0 = verticesHomogeneousClipSpace[0].z;
    f32 zNDC1MinusZNDC0 = (verticesHomogeneousClipSpace[1].z - zNDC0) * oneOverDoubleArea;
    f32 zNDC2MinusZNDC0 = (verticesHomogeneousClipSpace[2].z - zNDC0) * oneOverDoubleArea;

    // TODO(gviv): Sub-pixel accuracy.

    // Initialization of the barycentric weights.
    int stepXW0 = (verticesScreenSpace[2].y - verticesScreenSpace[1].y);
    int stepXW1 = (verticesScreenSpace[0].y - verticesScreenSpace[2].y);
    int stepXW2 = (verticesScreenSpace[1].y - verticesScreenSpace[0].y);

    int stepYW0 = (verticesScreenSpace[1].x - verticesScreenSpace[2].x);
    int stepYW1 = (verticesScreenSpace[2].x - verticesScreenSpace[0].x);
    int stepYW2 = (verticesScreenSpace[0].x - verticesScreenSpace[1].x);

    int w0Row = (bbMin.x - verticesScreenSpace[1].x) * stepXW0 +
                (bbMin.y - verticesScreenSpace[1].y) * stepYW0;
    int w1Row = (bbMin.x - verticesScreenSpace[2].x) * stepXW1 +
                (bbMin.y - verticesScreenSpace[2].y) * stepYW1;
    int w2Row = (bbMin.x - verticesScreenSpace[0].x) * stepXW2 +
                (bbMin.y - verticesScreenSpace[0].y) * stepYW2;

    u32* row = pixels + bbMin.y * bufferWidth + bbMin.x;
    for (int y = bbMin.y; y <= bbMax.y; ++y)
    {
        u32* pixel = row;
        int w0 = w0Row;
        int w1 = w1Row;
        int w2 = w2Row;

        for (int x = bbMin.x; x <= bbMax.x; ++x)
        {
            if ((w0 | w1 | w2) >= 0)
            {
                // We're inside the triangle
                f32 z = bary(zNDC0, zNDC1MinusZNDC0, zNDC2MinusZNDC0, (f32)w1, (f32)w2);

                if (z < depthBuffer[y * bufferWidth + x])
                {
                    // We passed the depth test.
                    depthBuffer[y * bufferWidth + x] = z;
                    f32 zFragment =
                        1.f / ((f32) w0 * oneOverZView0DoubleArea +
                                (f32) w1 * oneOverZView1DoubleArea +
                                (f32) w2 * oneOverZView2DoubleArea);

                    // Perform perspective-correct barycentric
                    // interpolation for each attributes output by
                    // the vertex shader.
                    // TODO(gviv): Every attribute is perspective-correctly
                    // interpolated at the moment. We could also support
                    // linear attributes, that wouldn't be perspective-corrected.
                    for (int i = 0; i < shader.nbVaryings; ++i)
                    {
                        fragmentShaderInput[i] =
                            zFragment * bary(baryData[i][0], baryData[i][1],
                                                baryData[i][2], (f32)w1,
                                                (f32)w2);
                    }

                    // TODO(gviv): Maybe we should try to avoid the
                    // function call with some crazy templates?
                    v4 finalColor = shader.fragmentShader(
                        shader.fragmentUniforms, fragmentShaderInput);

                    // TODO(gviv): Gamma correction.

                    finalColor.r = clamp01(finalColor.r);
                    finalColor.g = clamp01(finalColor.g);
                    finalColor.b = clamp01(finalColor.b);
                    finalColor.a = clamp01(finalColor.a);

                    v4 pixelColor = finalColor * 255.f;
                    *pixel =
                        ((u32)(pixelColor.r + .5f)) |
                        ((u32)(pixelColor.g + .5f) << 8) |
                        ((u32)(pixelColor.b + .5f) << 16) |
                        ((u32)(pixelColor.a + .5f) << 24);
                }
            }

            w0 += stepXW0;
            w1 += stepXW1;
            w2 += stepXW2;
            ++pixel;
        }

        w0Row += stepYW0;
        w1Row += stepYW1;
        w2Row += stepYW2;
        row += bufferWidth;
    }
}

void render(
    Mesh& mesh,
    Shader shader,
    u32* pixels,
    i32 bufferWidth,
    i32 bufferHeight,
    f32* depthBuffer)
{
    for (int vertexIndex = 0;
        vertexIndex < mesh.indices.size();
        vertexIndex += 3)
    {
        v4 verticesHomogeneousClipSpace[3];
        f32 vertexShader0Output[NB_MAX_VARYINGS];
        f32 vertexShader1Output[NB_MAX_VARYINGS];
        f32 vertexShader2Output[NB_MAX_VARYINGS];

        f32* vertexShaderOutputs[3]{
            vertexShader0Output,
            vertexShader1Output,
            vertexShader2Output,
        };

        for (int i = 0; i < 3; ++i)
        {
            const Vertex& vertex = mesh.vertices[vertexIndex + i];
            verticesHomogeneousClipSpace[i] = shader.vertexShader(
                vertex, shader.vertexUniforms, vertexShaderOutputs[i]);
        }

        //
        // Frustum culling
        //
        f32 x0 = verticesHomogeneousClipSpace[0].x;
        f32 y0 = verticesHomogeneousClipSpace[0].y;
        f32 z0 = verticesHomogeneousClipSpace[0].z;
        f32 w0 = verticesHomogeneousClipSpace[0].w;

        f32 x1 = verticesHomogeneousClipSpace[1].x;
        f32 y1 = verticesHomogeneousClipSpace[1].y;
        f32 z1 = verticesHomogeneousClipSpace[1].z;
        f32 w1 = verticesHomogeneousClipSpace[1].w;

        f32 x2 = verticesHomogeneousClipSpace[2].x;
        f32 y2 = verticesHomogeneousClipSpace[2].y;
        f32 z2 = verticesHomogeneousClipSpace[2].z;
        f32 w2 = verticesHomogeneousClipSpace[2].w;

        // This stores the signed distance between the vertices and each
        // clipping plane (dot product). The right, top and far dot products are
        // inverted compared to the plane's equations because we want it to be
        // positive inside the frustum (so negative outside).
        f32 planeDistances[3][6]{
            // First vertex
            {
                x0,      // Left plane   (x = 0)
                w0 - x0, // Right plane  (x = w)
                y0,      // Bottom plane (y = 0)
                w0 - y0, // Top plane    (y = w)
                z0,      // Near plane   (z = 0)
                w0 - z0  // Far plane    (z = w)
            },

            // Second vertex
            {
                x1,
                w1 - x1,
                y1,
                w1 - y1,
                z1,
                w1 - z1
            },

            // Third vertex
            {
                x2,
                w2 - x2,
                y2,
                w2 - y2,
                z2,
                w2 - z2
            }
        };

        // The following out codes contains, at the bit i:
        //     1 if the vertex if outside the plane i
        //     0 otherwise
        // We assume that we're outside if planeDistance is negative.
        i32 outCodes[3]{
            // First vertex
            (planeDistances[0][0] < 0.f) << 0 |
            (planeDistances[0][1] < 0.f) << 1 |
            (planeDistances[0][2] < 0.f) << 2 |
            (planeDistances[0][3] < 0.f) << 3 |
            (planeDistances[0][4] < 0.f) << 4 |
            (planeDistances[0][5] < 0.f) << 5,

            // Second vertex
            (planeDistances[1][0] < 0.f) << 0 |
            (planeDistances[1][1] < 0.f) << 1 |
            (planeDistances[1][2] < 0.f) << 2 |
            (planeDistances[1][3] < 0.f) << 3 |
            (planeDistances[1][4] < 0.f) << 4 |
            (planeDistances[1][5] < 0.f) << 5,

            // Third vertex
            (planeDistances[2][0] < 0.f) << 0 |
            (planeDistances[2][1] < 0.f) << 1 |
            (planeDistances[2][2] < 0.f) << 2 |
            (planeDistances[2][3] < 0.f) << 3 |
            (planeDistances[2][4] < 0.f) << 4 |
            (planeDistances[2][5] < 0.f) << 5
        };

        if (outCodes[0] & outCodes[1] & outCodes[2])
        {
            // The triangle is not in the frustum at all, we can skip it.
            g_nbTrianglesOutside++;
            continue;
        }

        if (!(outCodes[0] | outCodes[1] | outCodes[2]))
        {
            // The triangle in entirely inside the frustum, we can
            // render it directly.
            rasterize(shader, pixels, vertexShaderOutputs, bufferWidth,
                      bufferHeight, depthBuffer, verticesHomogeneousClipSpace);
            continue;
        }

        //
        // Clipping
        //
        auto clipEdge = [](f32 distPoint0ToPlane, f32 distPoint1ToPlane)
        {
            // We assume that both distances have different signs (otherwise,
            // the edge needn't be clipped in the first place).
            // Returns the position of the intersection point, between 0 and 1.
            return distPoint0ToPlane / (distPoint0ToPlane - distPoint1ToPlane);
        };

        // Linearly interpolated attributes, generated during clipping.
        // The indices 0, 1, 2 correspond to the first clipped triangle,
        // and the indices 0, 2, 3 correspond to the second clipped
        // triangle (if any).
        // TODO(gviv): We could avoid storing 4 arrays containing every vertex
        // attributes, because we can override the vertexShaderOutputs. We only
        // need to store one array of attributes, corresponding to the index 3.
        // Doing this would save some memory, but might force us to test if we
        // should write the current interpolated attributes inside the original
        // vertexShaderOutputs or into the new array.
        f32 outAttributes[4][NB_MAX_VARYINGS];
        v4 outVerticesHomogeneousClipSpace[4];

        auto lerpAttributes = [&](int startIndex, int endIndex, int outputVertexIndex)
        {
            f32 t = clipEdge(planeDistances[startIndex][4], planeDistances[endIndex][4]);

            for (int attributeIndex = 0;
                 attributeIndex < shader.nbVaryings;
                 ++attributeIndex)
            {
                outAttributes[outputVertexIndex][attributeIndex] =
                    lerp(vertexShaderOutputs[startIndex][attributeIndex],
                         vertexShaderOutputs[endIndex][attributeIndex],
                         t);
            }
            outVerticesHomogeneousClipSpace[outputVertexIndex] =
                lerp(verticesHomogeneousClipSpace[startIndex],
                     verticesHomogeneousClipSpace[endIndex],
                     t);
        };

        int nbCreatedVertices = 0;

        // This only clips against the near plane at the moment (hard-coded in
        // the comparisons).
        int startIndex = 2;
        for (int endIndex = 0; endIndex < 3; ++endIndex)
        {
            const v4& startVertex = verticesHomogeneousClipSpace[startIndex];
            const v4& endVertex = verticesHomogeneousClipSpace[endIndex];

            // If z is inside the frustum
            if (!(outCodes[endIndex] & (1 << 4)))
            {
                // If z is outside
                if (outCodes[startIndex] & (1 << 4))
                {
                    // We crossed the near plane, so we interpolate
                    // the vertices and other attributes
                    lerpAttributes(startIndex, endIndex, nbCreatedVertices);
                    nbCreatedVertices++;
                }

                // The endVertex is in the view volume, so we add it,
                // along with its attributes
                memcpy(outAttributes[nbCreatedVertices],
                       vertexShaderOutputs[endIndex],
                       shader.nbVaryings * sizeof(f32));
                outVerticesHomogeneousClipSpace[nbCreatedVertices] =
                    verticesHomogeneousClipSpace[endIndex];
                nbCreatedVertices++;
            }
            // If z is inside the frustum
            else if (!(outCodes[startIndex] & (1 << 4)))
            {
                // We crossed the near plane
                lerpAttributes(startIndex, endIndex, nbCreatedVertices);
                nbCreatedVertices++;
            }

            startIndex = endIndex;
        }

        f32* interpolatedAttributes[3];
        v4 interpolatedHomogeneousClipSpace[3];

        // TODO(gviv): Drop triangle if one vertex is (0, 0, 0, 0).

        if (nbCreatedVertices == 4)
        {
            // We clipped two triangles

            // The first triangle's attributes are directly those interpolated
            // in outAttributes, at indices 0, 1, 2
            interpolatedAttributes[0] = outAttributes[0];
            interpolatedAttributes[1] = outAttributes[1];
            interpolatedAttributes[2] = outAttributes[2];
            interpolatedHomogeneousClipSpace[0] =
                outVerticesHomogeneousClipSpace[0];
            interpolatedHomogeneousClipSpace[1] =
                outVerticesHomogeneousClipSpace[1];
            interpolatedHomogeneousClipSpace[2] =
                outVerticesHomogeneousClipSpace[2];

            // Rasterize it
            rasterize(shader,
                pixels, interpolatedAttributes, bufferWidth, bufferHeight,
                depthBuffer,
                interpolatedHomogeneousClipSpace);

            // TODO(gviv): `rasterize` should not modify the memory pointed to
            // by interpolatedAttributes, because we use the same memory for the
            // second triangle! (I don't want to bother with some crazy `const`
            // nonsense, at least for now)

            // The second triangle's attributes are those interpolated
            // in outAttributes, at indices 0, 2, 3
            interpolatedAttributes[0] = outAttributes[0];
            interpolatedAttributes[1] = outAttributes[2];
            interpolatedAttributes[2] = outAttributes[3];
            interpolatedHomogeneousClipSpace[0] =
                outVerticesHomogeneousClipSpace[0];
            interpolatedHomogeneousClipSpace[1] =
                outVerticesHomogeneousClipSpace[2];
            interpolatedHomogeneousClipSpace[2] =
                outVerticesHomogeneousClipSpace[3];

            // Rasterize it
            rasterize(shader,
                pixels, interpolatedAttributes, bufferWidth, bufferHeight,
                depthBuffer,
                interpolatedHomogeneousClipSpace);
        }
        else
        {
            // We already discarded non-visible triangles, so
            // the current triangle is necessarily in the visible side
            // of the near plane, and thus the clipping has necessarily
            // generated 3 vertices, corresponding to one triangle.
            // This triangle might have been actually clipped against
            // the near plane if it crossed it, or it's just the
            // original one (non-clipped) if it didn't.
            assert(nbCreatedVertices == 3);

            // We clipped one triangle.
            // Prepare the triangle's attributes.
            interpolatedAttributes[0] = outAttributes[0];
            interpolatedAttributes[1] = outAttributes[1];
            interpolatedAttributes[2] = outAttributes[2];
            interpolatedHomogeneousClipSpace[0] =
                outVerticesHomogeneousClipSpace[0];
            interpolatedHomogeneousClipSpace[1] =
                outVerticesHomogeneousClipSpace[1];
            interpolatedHomogeneousClipSpace[2] =
                outVerticesHomogeneousClipSpace[2];

            // Rasterize it
            rasterize(shader,
                pixels, interpolatedAttributes, bufferWidth, bufferHeight,
                depthBuffer,
                interpolatedHomogeneousClipSpace);
        }

        ++g_nbTrianglesClipped;
    }
}
