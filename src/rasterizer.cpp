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

inline
f32_x8 bary(f32_x8 v0, f32_x8 v1MinusV0, f32_x8 v2MinusV0, f32_x8 w1, f32_x8 w2)
{
    return _mm256_fmadd_ps(w2, v2MinusV0, _mm256_fmadd_ps(w1, v1MinusV0, v0));
}

// NOTE: We are in a very early development stage, still experimenting. The #ifs
// mess will be refactored later!
void rasterize(
    Shader& shader,
    u32* pixels,
#if SIMD_VERTEX
    f32_x8* vertexShaderOutput[3],
#else
    f32* vertexShaderOutput[3],
#endif
    i32 bufferWidth,
    i32 bufferHeight,
    f32* depthBuffer,
#if SIMD_VERTEX
    u32 invisibleTrianglesMask,
    v4_x8 verticesHomogeneousClipSpace[3])
#else
    v4 verticesHomogeneousClipSpace[3])
#endif
{
#if SIMD_VERTEX
    v2i_x8 verticesScreenSpace[3];
#else
    v2i verticesScreenSpace[3];
#endif

#if SIMD_VERTEX
    i32_x8 maxWidth = makeI32_x8(bufferWidth - 1);
    i32_x8 maxHeight = makeI32_x8(bufferHeight - 1);
    f32_x8 maxWidth_f32 = _mm256_cvtepi32_ps(maxWidth);
    f32_x8 maxHeight_f32 = _mm256_cvtepi32_ps(maxHeight);
#else
    i32 maxWidth = bufferWidth - 1;
    i32 maxHeight = bufferHeight - 1;
#endif

#if SIMD_VERTEX || SIMD_FRAGMENT
    i32_x8 zero_256i = _mm256_setzero_si256();
    f32_x8 one = makeF32_x8(1.f);
#endif

    for (int i = 0; i < 3; ++i)
    {
#if SIMD_VERTEX
        f32_x8& w = verticesHomogeneousClipSpace[i].w;
#else
        f32& w = verticesHomogeneousClipSpace[i].w;
#endif

        // TODO(gviv): Test division by zero?
#if SIMD_VERTEX
        // We have to perform full inversion because `_mm256_rcp_ps` is not
        // accurate enough when the near and far plane are too distant from
        // each other.
        w = one / w;
#else
        w = 1.f / w;
#endif
        verticesHomogeneousClipSpace[i].x *= w;
        verticesHomogeneousClipSpace[i].y *= w;
        verticesHomogeneousClipSpace[i].z *= w;

        // We are now in the canonical view volume (a.k.a. normalized
        // device coordinates) ([0, 1]^3).

#if SIMD_VERTEX
        // TODO(gviv): Does `_mm256_cvtps_epi32` round? It doesn't really
        // matter now, we'll revise this anyway to support sub-pixel precision.
        verticesScreenSpace[i] = {
            _mm256_cvtps_epi32(verticesHomogeneousClipSpace[i].x * maxWidth_f32),
            _mm256_cvtps_epi32((one - verticesHomogeneousClipSpace[i].y) * maxHeight_f32)
        };
#else
        verticesScreenSpace[i] = {
            (i32)((verticesHomogeneousClipSpace[i].x) * (f32)maxWidth + .5f),
            (i32)((1.f - verticesHomogeneousClipSpace[i].y) * (f32)maxHeight + .5f)
        };
#endif
    }

    // NOTE: `verticesHomogeneousClipSpace` now represents the vertices in
    // normalized devices coordinates!

    // Compute the double area of the triangle. If it's negative,
    // the triangle is backfacing.
#if SIMD_VERTEX
    i32_x8 doubleTriangleArea =
        (verticesScreenSpace[0].x - verticesScreenSpace[2].x) *
        (verticesScreenSpace[2].y - verticesScreenSpace[1].y) -
        (verticesScreenSpace[2].x - verticesScreenSpace[1].x) *
        (verticesScreenSpace[0].y - verticesScreenSpace[2].y);

    // TODO(gviv): If we have less than 8 valid triangles, we should only skip
    // if all the *valid* ones are backfacing. Otherwise, if some non-valid
    // triangles are not backfacing, we would miss the opportunity to early-out.
    // However, there are always 8 valid triangles, except when we are at the
    // very end of the vertices (when we are rendering the last 24 vertices), so
    // it might not be worth taking care of this every time, only to have the
    // opportunity to early-out for the last 24 vertices. Furthermore, when
    // we don't have 24 vertices (because we are at the end, and the number of
    // vertices is not divisible by 24), the position of the non-valid triangles
    // is set to 0. After all transformations, the x and y are likely to end
    // up being the same, so the triangle will have zero area (and so
    // considered backfacing), unless we have some weird transformations that
    // scale x and y differently. All this to say that we might not care.

    // Return if all 8 triangles are backfacing or have zero area.
    i32_x8 areaMask = _mm256_cmpgt_epi32(doubleTriangleArea, zero_256i);
    if (_mm256_testz_si256(areaMask, areaMask))
    {
        // TODO(gviv): We only increment this if *all eight* triangles are
        // backfacing, but we should also count if *some of them* are, if we
        // want a precise count.
        g_nbTrianglesBackfacing += 8;
        return;
    }
#else
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
#endif

#if SIMD_FRAGMENT
#if SIMD_VERTEX
    f32_x8 oneOverDoubleArea = one / _mm256_cvtepi32_ps(doubleTriangleArea);
#else
    f32_x8 oneOverDoubleArea = one / makeF32_x8((f32)doubleTriangleArea);
#endif
#else
    f32 oneOverDoubleArea = 1.f / doubleTriangleArea;
#endif

#if SIMD_VERTEX
    v2i_x8 bbMin;
    v2i_x8 bbMax;

    // Compute the triangle's bounding box
    bbMin.x = _mm256_min_epi32(
        _mm256_min_epi32(verticesScreenSpace[0].x, verticesScreenSpace[1].x),
        verticesScreenSpace[2].x);
    bbMin.y = _mm256_min_epi32(
        _mm256_min_epi32(verticesScreenSpace[0].y, verticesScreenSpace[1].y),
        verticesScreenSpace[2].y);
    bbMax.x = _mm256_max_epi32(
        _mm256_max_epi32(verticesScreenSpace[0].x, verticesScreenSpace[1].x),
        verticesScreenSpace[2].x);
    bbMax.y = _mm256_max_epi32(
        _mm256_max_epi32(verticesScreenSpace[0].y, verticesScreenSpace[1].y),
        verticesScreenSpace[2].y);

    // Clip to screen
    bbMin.x = _mm256_max_epi32(zero_256i, bbMin.x);
    bbMin.y = _mm256_max_epi32(zero_256i, bbMin.y);
    bbMax.x = _mm256_min_epi32(maxWidth, bbMax.x);
    bbMax.y = _mm256_min_epi32(maxHeight, bbMax.y);
#else
    v2i bbMin;
    v2i bbMax;

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
#endif

#if SIMD_FRAGMENT
    f32_x8 baryData[NB_MAX_VARYINGS][3];
#else
    f32 baryData[NB_MAX_VARYINGS][3];
#endif

#if SIMD_VERTEX
    const f32_x8* vertexShaderOutput0Ptr = vertexShaderOutput[0];
    const f32_x8* vertexShaderOutput1Ptr = vertexShaderOutput[1];
    const f32_x8* vertexShaderOutput2Ptr = vertexShaderOutput[2];
#else
    const f32* vertexShaderOutput0Ptr = vertexShaderOutput[0];
    const f32* vertexShaderOutput1Ptr = vertexShaderOutput[1];
    const f32* vertexShaderOutput2Ptr = vertexShaderOutput[2];
#endif

#if SIMD_FRAGMENT
#if SIMD_VERTEX
    f32_x8 oneOverZView0 = verticesHomogeneousClipSpace[0].w;
    f32_x8 oneOverZView1 = verticesHomogeneousClipSpace[1].w;
    f32_x8 oneOverZView2 = verticesHomogeneousClipSpace[2].w;
#else
    f32_x8 oneOverZView0 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[0].w);
    f32_x8 oneOverZView1 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[1].w);
    f32_x8 oneOverZView2 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[2].w);
#endif
#else
    f32 oneOverZView0 = verticesHomogeneousClipSpace[0].w;
    f32 oneOverZView1 = verticesHomogeneousClipSpace[1].w;
    f32 oneOverZView2 = verticesHomogeneousClipSpace[2].w;
#endif

#if SIMD_FRAGMENT
    f32_x8 oneOverZView0DoubleArea = oneOverZView0 * oneOverDoubleArea;
    f32_x8 oneOverZView1DoubleArea = oneOverZView1 * oneOverDoubleArea;
    f32_x8 oneOverZView2DoubleArea = oneOverZView2 * oneOverDoubleArea;
#else
    f32 oneOverZView0DoubleArea = oneOverZView0 * oneOverDoubleArea;
    f32 oneOverZView1DoubleArea = oneOverZView1 * oneOverDoubleArea;
    f32 oneOverZView2DoubleArea = oneOverZView2 * oneOverDoubleArea;
#endif

    for (int attributeIndex = 0;
         attributeIndex < shader.nbVaryings;
         ++attributeIndex)
    {
        // Predivide every attributes for perspective-correct interpolation.
#if SIMD_FRAGMENT
    #if SIMD_VERTEX
        f32_x8 output0_x8 = *(vertexShaderOutput0Ptr++);
        f32_x8 output1_x8 = *(vertexShaderOutput1Ptr++);
        f32_x8 output2_x8 = *(vertexShaderOutput2Ptr++);
    #else
        f32_x8 output0_x8 = _mm256_broadcast_ss(vertexShaderOutput0Ptr++);
        f32_x8 output1_x8 = _mm256_broadcast_ss(vertexShaderOutput1Ptr++);
        f32_x8 output2_x8 = _mm256_broadcast_ss(vertexShaderOutput2Ptr++);
    #endif

        f32_x8 v0 = output0_x8 * oneOverZView0;
        f32_x8 v1 = output1_x8 * oneOverZView1;
        f32_x8 v2 = output2_x8 * oneOverZView2;
#else
        f32 v0 = *vertexShaderOutput0Ptr++ * oneOverZView0;
        f32 v1 = *vertexShaderOutput1Ptr++ * oneOverZView1;
        f32 v2 = *vertexShaderOutput2Ptr++ * oneOverZView2;
#endif

        // We multiply by oneOverDoubleArea to normalize the barycentric
        // weights that will be used in the bary interpolation.
        baryData[attributeIndex][0] = v0;
        baryData[attributeIndex][1] = (v1 - v0) * oneOverDoubleArea;
        baryData[attributeIndex][2] = (v2 - v0) * oneOverDoubleArea;
    }

#if SIMD_FRAGMENT
    f32_x8 fragmentShaderInput[NB_MAX_VARYINGS];
#else
    f32 fragmentShaderInput[NB_MAX_VARYINGS];
#endif
    // NOTE: Since these terms are used for linear interpolation, we don't need
    // to multiply them by oneOverZView, unlike what we do for perspective-
    // correct interpolation.
#if SIMD_FRAGMENT
#if SIMD_VERTEX
    f32_x8 zNDC0 = verticesHomogeneousClipSpace[0].z;
    f32_x8 zNDC1 = verticesHomogeneousClipSpace[1].z;
    f32_x8 zNDC2 = verticesHomogeneousClipSpace[2].z;
#else
    f32_x8 zNDC0 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[0].z);
    f32_x8 zNDC1 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[1].z);
    f32_x8 zNDC2 = _mm256_broadcast_ss(&verticesHomogeneousClipSpace[2].z);
#endif

    f32_x8 zNDC1MinusZNDC0 = (zNDC1 - zNDC0) * oneOverDoubleArea;
    f32_x8 zNDC2MinusZNDC0 = (zNDC2 - zNDC0) * oneOverDoubleArea;
#else
    f32 zNDC0 = verticesHomogeneousClipSpace[0].z;
    f32 zNDC1MinusZNDC0 = (verticesHomogeneousClipSpace[1].z - zNDC0) * oneOverDoubleArea;
    f32 zNDC2MinusZNDC0 = (verticesHomogeneousClipSpace[2].z - zNDC0) * oneOverDoubleArea;
#endif

#if SIMD_FRAGMENT
    f32_x8 twoFiftyFive = makeF32_x8(255.f);
    i32_x8 bufferWidth_x8 = makeI32_x8(bufferWidth);
    i32_x8 offsetX = makeI32_x8(0, 1, 2, 3, 4, 5, 6, 7);
#endif

    // TODO(gviv): Sub-pixel accuracy.

#if SIMD_VERTEX
    for (int lane = 0; lane < 8; ++lane)
    {
        // Skip triangle if it's outside the frustum.
        if (invisibleTrianglesMask & (1 << lane)) continue;
#endif

    // Initialization of the barycentric weights.
#if SIMD_FRAGMENT
    #if SIMD_VERTEX
        // TODO(gviv): The `.m256i_i32[]` syntax is non-portable (MSVC only).
        i32_x8 verticesScreenSpace0X = makeI32_x8(verticesScreenSpace[0].x.m256i_i32[lane]);
        i32_x8 verticesScreenSpace1X = makeI32_x8(verticesScreenSpace[1].x.m256i_i32[lane]);
        i32_x8 verticesScreenSpace2X = makeI32_x8(verticesScreenSpace[2].x.m256i_i32[lane]);

        i32_x8 verticesScreenSpace0Y = makeI32_x8(verticesScreenSpace[0].y.m256i_i32[lane]);
        i32_x8 verticesScreenSpace1Y = makeI32_x8(verticesScreenSpace[1].y.m256i_i32[lane]);
        i32_x8 verticesScreenSpace2Y = makeI32_x8(verticesScreenSpace[2].y.m256i_i32[lane]);
    #else
        i32_x8 verticesScreenSpace0X = makeI32_x8(verticesScreenSpace[0].x);
        i32_x8 verticesScreenSpace1X = makeI32_x8(verticesScreenSpace[1].x);
        i32_x8 verticesScreenSpace2X = makeI32_x8(verticesScreenSpace[2].x);

        i32_x8 verticesScreenSpace0Y = makeI32_x8(verticesScreenSpace[0].y);
        i32_x8 verticesScreenSpace1Y = makeI32_x8(verticesScreenSpace[1].y);
        i32_x8 verticesScreenSpace2Y = makeI32_x8(verticesScreenSpace[2].y);
    #endif

        i32_x8 stepXW0 = verticesScreenSpace2Y - verticesScreenSpace1Y;
        i32_x8 stepXW1 = verticesScreenSpace0Y - verticesScreenSpace2Y;
        i32_x8 stepXW2 = verticesScreenSpace1Y - verticesScreenSpace0Y;

        i32_x8 stepYW0 = verticesScreenSpace1X - verticesScreenSpace2X;
        i32_x8 stepYW1 = verticesScreenSpace2X - verticesScreenSpace0X;
        i32_x8 stepYW2 = verticesScreenSpace0X - verticesScreenSpace1X;

    #if SIMD_VERTEX
        i32_x8 bbMinX = makeI32_x8(bbMin.x.m256i_i32[lane]) + offsetX;
        i32_x8 bbMinY = makeI32_x8(bbMin.y.m256i_i32[lane]);
    #else
        i32_x8 bbMinX = makeI32_x8(bbMin.x) + offsetX;
        i32_x8 bbMinY = makeI32_x8(bbMin.y);
    #endif

        i32_x8 w0Row = (bbMinX - verticesScreenSpace1X) * stepXW0 +
                       (bbMinY - verticesScreenSpace1Y) * stepYW0;
        i32_x8 w1Row = (bbMinX - verticesScreenSpace2X) * stepXW1 +
                       (bbMinY - verticesScreenSpace2Y) * stepYW1;
        i32_x8 w2Row = (bbMinX - verticesScreenSpace0X) * stepXW2 +
                       (bbMinY - verticesScreenSpace0Y) * stepYW2;

        i32_x8 eight = makeI32_x8(8);
        stepXW0 *= eight;
        stepXW1 *= eight;
        stepXW2 *= eight;
#else
        int stepXW0 = verticesScreenSpace[2].y - verticesScreenSpace[1].y;
        int stepXW1 = verticesScreenSpace[0].y - verticesScreenSpace[2].y;
        int stepXW2 = verticesScreenSpace[1].y - verticesScreenSpace[0].y;

        int stepYW0 = verticesScreenSpace[1].x - verticesScreenSpace[2].x;
        int stepYW1 = verticesScreenSpace[2].x - verticesScreenSpace[0].x;
        int stepYW2 = verticesScreenSpace[0].x - verticesScreenSpace[1].x;

        int w0Row = (bbMin.x - verticesScreenSpace[1].x) * stepXW0 +
                    (bbMin.y - verticesScreenSpace[1].y) * stepYW0;
        int w1Row = (bbMin.x - verticesScreenSpace[2].x) * stepXW1 +
                    (bbMin.y - verticesScreenSpace[2].y) * stepYW1;
        int w2Row = (bbMin.x - verticesScreenSpace[0].x) * stepXW2 +
                    (bbMin.y - verticesScreenSpace[0].y) * stepYW2;
#endif

#if SIMD_VERTEX
        f32_x8 zNDC0_curLane = makeF32_x8(zNDC0.m256_f32[lane]);
        f32_x8 zNDC1MinusZNDC0_curLane = makeF32_x8(zNDC1MinusZNDC0.m256_f32[lane]);
        f32_x8 zNDC2MinusZNDC0_curLane = makeF32_x8(zNDC2MinusZNDC0.m256_f32[lane]);

        f32_x8 oneOverZView0DoubleArea_curLane = makeF32_x8(oneOverZView0DoubleArea.m256_f32[lane]);
        f32_x8 oneOverZView1DoubleArea_curLane = makeF32_x8(oneOverZView1DoubleArea.m256_f32[lane]);
        f32_x8 oneOverZView2DoubleArea_curLane = makeF32_x8(oneOverZView2DoubleArea.m256_f32[lane]);

        f32_x8 baryData_curLane[NB_MAX_VARYINGS][3];
        for (int i = 0; i < shader.nbVaryings; ++i)
        {
            baryData_curLane[i][0] = makeF32_x8(baryData[i][0].m256_f32[lane]);
            baryData_curLane[i][1] = makeF32_x8(baryData[i][1].m256_f32[lane]);
            baryData_curLane[i][2] = makeF32_x8(baryData[i][2].m256_f32[lane]);
        }
#endif

#if SIMD_VERTEX
        u32* row = pixels + bbMin.y.m256i_i32[lane] * bufferWidth + bbMin.x.m256i_i32[lane];
        for (int y = bbMin.y.m256i_i32[lane]; y <= bbMax.y.m256i_i32[lane]; ++y)
#else
        u32* row = pixels + bbMin.y * bufferWidth + bbMin.x;
        for (int y = bbMin.y; y <= bbMax.y; ++y)
#endif
        {
            u32* pixel = row;
#if SIMD_FRAGMENT
            i32_x8 w0 = w0Row;
            i32_x8 w1 = w1Row;
            i32_x8 w2 = w2Row;
#else
            int w0 = w0Row;
            int w1 = w1Row;
            int w2 = w2Row;
#endif

#if SIMD_FRAGMENT
    #if SIMD_VERTEX
            for (int x = bbMin.x.m256i_i32[lane];
                 x <= bbMax.x.m256i_i32[lane];
                 x += 8)
    #else
            for (int x = bbMin.x; x <= bbMax.x; x += 8)
    #endif
#else
            for (int x = bbMin.x; x <= bbMax.x; ++x)
#endif
            {
#if SIMD_FRAGMENT
                i32_x8 weightsMask = w0 | w1 | w2;
                if (_mm256_movemask_ps(_mm256_castsi256_ps(weightsMask)) != 0xFF)
#else
                if ((w0 | w1 | w2) >= 0)
#endif
                {
                // We're inside the triangle
#if SIMD_FRAGMENT
                    f32_x8 w1_f32 = _mm256_cvtepi32_ps(w1);
                    f32_x8 w2_f32 = _mm256_cvtepi32_ps(w2);
    #if SIMD_VERTEX
                    f32_x8 z = bary(zNDC0_curLane, zNDC1MinusZNDC0_curLane, zNDC2MinusZNDC0_curLane, w1_f32, w2_f32);
    #else
                    f32_x8 z = bary(zNDC0, zNDC1MinusZNDC0, zNDC2MinusZNDC0, w1_f32, w2_f32);
    #endif
#else
                    f32 z = bary(zNDC0, zNDC1MinusZNDC0, zNDC2MinusZNDC0, (f32)w1, (f32)w2);
#endif

#if SIMD_FRAGMENT
                    // TODO(gviv): Why does MSVC emit `vmovups` while I want
                    // to do an *aligned* load??
                    f32_x8 storedDepth = _mm256_load_ps(depthBuffer + y * bufferWidth + x);
                    f32_x8 depthMask = _mm256_cmp_ps(z, storedDepth, _CMP_LT_OS);

                    if (!_mm256_testz_ps(depthMask, depthMask))
#else
                    if (z < depthBuffer[y * bufferWidth + x])
#endif
                    {
                        // We passed the depth test.
#if SIMD_FRAGMENT
                        i32_x8 writeMask = _mm256_andnot_si256(
                            weightsMask, _mm256_castps_si256(depthMask));

                        // TODO(gviv): Can we avoid doing this for each pixel?
                        // It's only needed if we're rendering the eight rightmost
                        // pixels of the screen.
                        i32_x8 clipMask = _mm256_cmpgt_epi32(
                            bufferWidth_x8, makeI32_x8(x) + offsetX);
                        writeMask &= clipMask;

                        storedDepth = _mm256_blendv_ps(storedDepth, z, _mm256_castsi256_ps(writeMask));
                        // TODO(gviv): Same here, what are you doing MSVC?
                        _mm256_store_ps(depthBuffer + y * bufferWidth + x, storedDepth);

                        f32_x8 w0_f32 = _mm256_cvtepi32_ps(w0);
    #if SIMD_VERTEX
                        // TODO(gviv): Safe to use `_mm256_rcp_ps` here?
                        f32_x8 zFragment = _mm256_rcp_ps(_mm256_fmadd_ps(
                            w0_f32, oneOverZView0DoubleArea_curLane,
                            _mm256_fmadd_ps(
                                w1_f32, oneOverZView1DoubleArea_curLane,
                                w2_f32 * oneOverZView2DoubleArea_curLane)));
    #else
                        f32_x8 zFragment = _mm256_rcp_ps(_mm256_fmadd_ps(
                            w0_f32, oneOverZView0DoubleArea,
                            _mm256_fmadd_ps(
                                w1_f32, oneOverZView1DoubleArea,
                                w2_f32 * oneOverZView2DoubleArea)));
    #endif
#else
                        depthBuffer[y * bufferWidth + x] = z;
                        f32 zFragment =
                            1.f / ((f32) w0 * oneOverZView0DoubleArea +
                                   (f32) w1 * oneOverZView1DoubleArea +
                                   (f32) w2 * oneOverZView2DoubleArea);
#endif

                        // Perform perspective-correct barycentric
                        // interpolation for each attributes output by
                        // the vertex shader.
                        // TODO(gviv): Every attribute is perspective-correctly
                        // interpolated at the moment. We could also support
                        // linear attributes, that wouldn't be perspective-corrected.
                        for (int i = 0; i < shader.nbVaryings; ++i)
                        {
#if SIMD_FRAGMENT
    #if SIMD_VERTEX
                            fragmentShaderInput[i] =
                                zFragment * bary(baryData_curLane[i][0],
                                                 baryData_curLane[i][1],
                                                 baryData_curLane[i][2], w1_f32,
                                                 w2_f32);
    #else
                            fragmentShaderInput[i] =
                                zFragment * bary(baryData[i][0], baryData[i][1],
                                                 baryData[i][2], w1_f32,
                                                 w2_f32);
    #endif
#else
                            fragmentShaderInput[i] =
                                zFragment * bary(baryData[i][0], baryData[i][1],
                                                 baryData[i][2], (f32)w1,
                                                 (f32)w2);
#endif
                        }

#if SIMD_FRAGMENT
                        v4_x8 pixelColor = shader.fragmentShader(
                            shader.fragmentUniforms, fragmentShaderInput);

                        pixelColor.r = clamp01(pixelColor.r) * twoFiftyFive;
                        pixelColor.b = clamp01(pixelColor.b) * twoFiftyFive;
                        pixelColor.g = clamp01(pixelColor.g) * twoFiftyFive;
                        pixelColor.a = clamp01(pixelColor.a) * twoFiftyFive;

                        // TODO(gviv): Gamma correction.

                        // TODO(gviv): Ensure this actually rounds.
                        i32_x8 r_i32 = _mm256_cvtps_epi32(pixelColor.r);
                        i32_x8 g_i32 = _mm256_cvtps_epi32(pixelColor.g);
                        i32_x8 b_i32 = _mm256_cvtps_epi32(pixelColor.b);
                        i32_x8 a_i32 = _mm256_cvtps_epi32(pixelColor.a);

                        i32_x8 pixelsToWrite =
                            r_i32 |
                            _mm256_slli_epi32(g_i32, 8) |
                            _mm256_slli_epi32(b_i32, 16) |
                            _mm256_slli_epi32(a_i32, 24);

                        // TODO(gviv): Instead of `maskstore`, is it better to
                        // blend and store (like the depth)?
                        _mm256_maskstore_epi32(reinterpret_cast<int*>(pixel),
                                               writeMask, pixelsToWrite);
#else
                        // TODO(gviv): Maybe we should try to avoid the
                        // function call with some crazy templates?
                        v4 finalColor = shader.fragmentShader(
                            shader.fragmentUniforms, fragmentShaderInput);

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
#endif
                    }
                }

                w0 += stepXW0;
                w1 += stepXW1;
                w2 += stepXW2;
#if SIMD_FRAGMENT
                pixel += 8;
#else
                ++pixel;
#endif
            }

            w0Row += stepYW0;
            w1Row += stepYW1;
            w2Row += stepYW2;
            row += bufferWidth;
        }
#if SIMD_VERTEX
    }
#endif
}

void render(
    Mesh& mesh,
    Shader shader,
    u32* pixels,
    i32 bufferWidth,
    i32 bufferHeight,
    f32* depthBuffer)
{
#if SIMD_VERTEX
    i32_x8 verticesSize = makeI32_x8((i32)mesh.vertices.size());
    f32_x8 zero = _mm256_setzero_ps();
    i32_x8 sizeofVertex = makeI32_x8(sizeof(Vertex));

    for (int vertexIndex = 0;
        vertexIndex < mesh.indices.size();
        vertexIndex += 24)
#else
    for (int vertexIndex = 0;
        vertexIndex < mesh.indices.size();
        vertexIndex += 3)
#endif
    {
#if SIMD_VERTEX
        v4_x8 verticesHomogeneousClipSpace[3];
        f32_x8 vertexShader0Output[NB_MAX_VARYINGS];
        f32_x8 vertexShader1Output[NB_MAX_VARYINGS];
        f32_x8 vertexShader2Output[NB_MAX_VARYINGS];

        f32_x8* vertexShaderOutputs[3]{
            vertexShader0Output,
            vertexShader1Output,
            vertexShader2Output,
        };
#else
        v4 verticesHomogeneousClipSpace[3];
        f32 vertexShader0Output[NB_MAX_VARYINGS];
        f32 vertexShader1Output[NB_MAX_VARYINGS];
        f32 vertexShader2Output[NB_MAX_VARYINGS];

        f32* vertexShaderOutputs[3]{
            vertexShader0Output,
            vertexShader1Output,
            vertexShader2Output,
        };
#endif

        for (int vertexIndexInTriangle = 0;
            vertexIndexInTriangle < 3;
            ++vertexIndexInTriangle)
        {
#if SIMD_VERTEX
            // The vertices are organized as follows.
            // Initially, all vertices are contiguous, and each triplet of
            // vertices defines a triangle:
            // 0 1 2  3 4 5  6 7 8...
            // -----  -----  -----
            // tri 0  tri 1  tri 2
            //
            // We want to perform computations on eight triangles at a time. To
            // do so, we reorganize the vertices in three groups of eight values
            // (one per vertex). The first index of a triangle goes in the first
            // group, the second in the second and so forth.
            // Incoming vertices:
            // 0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15   16 17 18 19 20 21 22 23
            // - - -
            //
            // Reorganized vertices:
            // 0 3 6 9 12 15 18 21   1 4 7 10 13 16 19 22   2 5 8 11 14 17 20 23
            // -                     -                      -
            //
            // We can then send the three reorganized groups to the vertex
            // shader, that will perform calculations on eight triangles at a
            // time.
            //
            // TODO(gviv): Maybe we could organize the attributes to be SIMD-
            // friendly beforehand, so we can just load the data directly.
            i32_x8 index = makeI32_x8(
                vertexIndex + vertexIndexInTriangle + 0,
                vertexIndex + vertexIndexInTriangle + 3,
                vertexIndex + vertexIndexInTriangle + 6,
                vertexIndex + vertexIndexInTriangle + 9,
                vertexIndex + vertexIndexInTriangle + 12,
                vertexIndex + vertexIndexInTriangle + 15,
                vertexIndex + vertexIndexInTriangle + 18,
                vertexIndex + vertexIndexInTriangle + 21);
            f32_x8 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(verticesSize, index));
            index *= sizeofVertex;

            // TODO(gviv): Is the mask really necessary to ensure that we don't
            // load outside our allocated memory?
            v3_x8 position{
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].position.x, index, mask, 1),
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].position.y, index, mask, 1),
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].position.z, index, mask, 1),
            };
            v2_x8 texCoords{
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].texCoords.x, index, mask, 1),
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].texCoords.y, index, mask, 1),
            };
            v3_x8 normal{
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].normal.x, index, mask, 1),
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].normal.y, index, mask, 1),
                _mm256_mask_i32gather_ps(zero, &mesh.vertices[0].normal.z, index, mask, 1),
            };

            // TODO(gviv): This is not very flexible, we should rather pass
            // the individual attributes to the vertex shader, to enable custom
            // vertex formats.
            Vertex_x8 vertex{position, texCoords, normal};
            verticesHomogeneousClipSpace[vertexIndexInTriangle] =
                shader.vertexShader(vertex, shader.vertexUniforms,
                                    vertexShaderOutputs[vertexIndexInTriangle]);
#else
            const Vertex& vertex = mesh.vertices[vertexIndex + vertexIndexInTriangle];
            verticesHomogeneousClipSpace[vertexIndexInTriangle] =
                shader.vertexShader(vertex, shader.vertexUniforms,
                                    vertexShaderOutputs[vertexIndexInTriangle]);
#endif
        }

        //
        // Frustum culling
        //
#if SIMD_VERTEX
        f32_x8 x0 = verticesHomogeneousClipSpace[0].x;
        f32_x8 y0 = verticesHomogeneousClipSpace[0].y;
        f32_x8 z0 = verticesHomogeneousClipSpace[0].z;
        f32_x8 w0 = verticesHomogeneousClipSpace[0].w;

        f32_x8 x1 = verticesHomogeneousClipSpace[1].x;
        f32_x8 y1 = verticesHomogeneousClipSpace[1].y;
        f32_x8 z1 = verticesHomogeneousClipSpace[1].z;
        f32_x8 w1 = verticesHomogeneousClipSpace[1].w;

        f32_x8 x2 = verticesHomogeneousClipSpace[2].x;
        f32_x8 y2 = verticesHomogeneousClipSpace[2].y;
        f32_x8 z2 = verticesHomogeneousClipSpace[2].z;
        f32_x8 w2 = verticesHomogeneousClipSpace[2].w;
#else
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
#endif

        // This stores the signed distance between the vertices and each
        // clipping plane (dot product). The right, top and far dot products are
        // inverted compared to the plane's equations because we want it to be
        // positive inside the frustum (so negative outside).
#if SIMD_VERTEX
        f32_x8 planeDistances[3][6]{
#else
        f32 planeDistances[3][6]{
#endif
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

#if SIMD_VERTEX
        i32_x8 outCodes[3][6]{
            // First vertex
            {
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][0], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][1], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][2], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][3], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][4], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[0][5], zero, _CMP_LT_OQ)),
            },

            // Second vertex
            {
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][0], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][1], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][2], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][3], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][4], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[1][5], zero, _CMP_LT_OQ)),
            },

            // Third vertex
            {
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][0], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][1], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][2], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][3], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][4], zero, _CMP_LT_OQ)),
                _mm256_castps_si256(_mm256_cmp_ps(planeDistances[2][5], zero, _CMP_LT_OQ)),
            }
        };
#else
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
#endif

#if SIMD_VERTEX
        // A triangle is outside the frustum if each vertices share at least one
        // side outside the frustum.
        i32_x8 trianglesMask =
            (outCodes[0][0] & outCodes[1][0] & outCodes[2][0]) |
            (outCodes[0][1] & outCodes[1][1] & outCodes[2][1]) |
            (outCodes[0][2] & outCodes[1][2] & outCodes[2][2]) |
            (outCodes[0][3] & outCodes[1][3] & outCodes[2][3]) |
            (outCodes[0][4] & outCodes[1][4] & outCodes[2][4]) |
            (outCodes[0][5] & outCodes[1][5] & outCodes[2][5]);

        u32 invisibleTrianglesMask =
            (u32)_mm256_movemask_ps(_mm256_castsi256_ps(trianglesMask));

        // TODO(gviv): When there are less than 8 triangles to process (some
        // triangles have been masked out), the attributes corresponding to
        // those triangles, passed to the vertex shader, are set to 0. In this
        // case, are we sure that they wouldn't be visible anyway so this test
        // is still valid? In other words, we must ensure that when a triangle
        // is masked out because it's outside the range of the vertices list,
        // it doesn't end up being visible because its homogeneous position
        // is now in the frustum, after being passed to the vertex shader
        // (which operates on all 8 triangles, regardless of whether they are
        // valid).
        if (invisibleTrianglesMask == 0xFF)
#else
        if (outCodes[0] & outCodes[1] & outCodes[2])
#endif
        {
#if SIMD_VERTEX
            // None of the 8 triangles are in the frustum, we can skip them.
            // TODO(gviv): Count the triangles outside not only when all of them
            // are outside, but also when *some* of them are outside.
            g_nbTrianglesOutside += 8;
#else
            // The triangle is not in the frustum at all, we can skip it.
            g_nbTrianglesOutside++;
#endif
            continue;
        }

#if SIMD_VERTEX
        // TODO(gviv): We should only do this if *all* of the triangles are
        // *entirely* inside the frustum (but since we don't have SIMD clipping
        // yet, we enter here either way).
#else
        if (!(outCodes[0] | outCodes[1] | outCodes[2]))
#endif
        {
#if SIMD_VERTEX
            rasterize(shader, pixels, vertexShaderOutputs, bufferWidth,
                      bufferHeight, depthBuffer, invisibleTrianglesMask,
                      verticesHomogeneousClipSpace);
#else
            // The triangle in entirely inside the frustum, we can
            // render it directly.
            rasterize(shader, pixels, vertexShaderOutputs, bufferWidth,
                      bufferHeight, depthBuffer, verticesHomogeneousClipSpace);
#endif
            continue;
        }

        //
        // Clipping
        //
        // TODO(gviv): SIMD clipping.
#if !SIMD_VERTEX
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
#endif
    }
}
