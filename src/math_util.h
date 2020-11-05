#ifndef MATH_UTIL_H_
#define MATH_UTIL_H_

#include <intrin.h>
#include <cstdint>
#include <cmath>

using i8  = std::int8_t;
using u8  = std::uint8_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

using f32_x8 = __m256;
using i32_x8 = __m256i;

// TODO(gviv): Add these in Voxium.
constexpr float PI = 3.14159265358979323846f;

template<typename T> inline constexpr
const T& min(const T& a, const T& b)
{
    return a > b ? b : a;
}

template<typename T> inline constexpr
const T& max(const T& a, const T& b)
{
    return a < b ? b : a;
}
//

inline
i32_x8 makeI32_x8(i32 value)
{
    return _mm256_set1_epi32(value);
}

inline
i32_x8 makeI32_x8(i32 e7, i32 e6, i32 e5, i32 e4, i32 e3, i32 e2, i32 e1, i32 e0)
{
    return _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
}

inline
i32_x8 operator+(const i32_x8& a, const i32_x8& b)
{
    return _mm256_add_epi32(a, b);
}

inline
i32_x8 operator-(const i32_x8& a, const i32_x8& b)
{
    return _mm256_sub_epi32(a, b);
}

inline
i32_x8 operator-(const i32_x8& a)
{
    return _mm256_xor_si256(a, _mm256_castps_si256(_mm256_set1_ps(-0.f)));
}

inline
i32_x8 operator*(const i32_x8& a, const i32_x8& b)
{
    return _mm256_mullo_epi32(a, b);
}

inline
i32_x8 operator/(const i32_x8& a, const i32_x8& b)
{
    return _mm256_div_epi32(a, b);
}

inline
i32_x8& operator+=(i32_x8& self, const i32_x8& other)
{
    self = self + other;

    return self;
}

inline
i32_x8& operator-=(i32_x8& self, const i32_x8& other)
{
    self = self - other;

    return self;
}

inline
i32_x8& operator*=(i32_x8& self, const i32_x8& other)
{
    self = self * other;

    return self;
}

inline
i32_x8& operator/=(i32_x8& self, const i32_x8& other)
{
    self = self / other;

    return self;
}

inline
i32_x8 operator|(const i32_x8& a, const i32_x8& b)
{
    return _mm256_or_si256(a, b);
}

inline
i32_x8 operator&(const i32_x8& a, const i32_x8& b)
{
    return _mm256_and_si256(a, b);
}

inline
i32_x8& operator|=(i32_x8& self, const i32_x8& other)
{
    self = self | other;

    return self;
}

inline
i32_x8& operator&=(i32_x8& self, const i32_x8& other)
{
    self = self & other;

    return self;
}

inline
f32_x8 makeF32_x8(f32 v)
{
    return _mm256_set1_ps(v);
}

inline
f32_x8 makeF32_x8(f32 e7, f32 e6, f32 e5, f32 e4, f32 e3, f32 e2, f32 e1, f32 e0)
{
    return _mm256_setr_ps(e7, e6, e5, e4, e3, e2, e1, e0);
}

inline
f32_x8 operator+(const f32_x8& a, const f32_x8& b)
{
    return _mm256_add_ps(a, b);
}

inline
f32_x8 operator-(const f32_x8& a, const f32_x8& b)
{
    return _mm256_sub_ps(a, b);
}

inline
f32_x8 operator-(const f32_x8& a)
{
    return _mm256_xor_ps(a, _mm256_set1_ps(-0.f));
}

inline
f32_x8 operator*(const f32_x8& a, const f32_x8& b)
{
    return _mm256_mul_ps(a, b);
}

inline
f32_x8 operator/(const f32_x8& a, const f32_x8& b)
{
    return _mm256_div_ps(a, b);
}

inline
f32_x8& operator+=(f32_x8& self, const f32_x8& other)
{
    self = self + other;

    return self;
}

inline
f32_x8& operator-=(f32_x8& self, const f32_x8& other)
{
    self = self - other;

    return self;
}

inline
f32_x8& operator*=(f32_x8& self, const f32_x8& other)
{
    self = self * other;

    return self;
}

inline
f32_x8& operator/=(f32_x8& self, const f32_x8& other)
{
    self = self / other;

    return self;
}

inline
f32 clamp01(f32 a)
{
    return min(max(0.f, a), 1.f);
}

inline
f32_x8 clamp01(const f32_x8& a)
{
    const f32_x8 one = makeF32_x8(1.f);
    const f32_x8 zero = _mm256_setzero_ps();

    return _mm256_min_ps(_mm256_max_ps(zero, a), one);
}

inline
f32_x8 abs(const f32_x8& a)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.f), a);
}

template<typename T> inline
T lerp(T a, T b, f32 t)
{
    return (1 - t) * a + b * t;
}

template<typename T> inline
T lerp(T a, T b, f32_x8 t)
{
    const f32_x8 one = makeF32_x8(1.f);

    return (one - t) * a + b * t;
}

//
// Vector2
//
template<typename T>
struct v2_
{
    union
    {
        struct
        {
            T x, y;
        };
        struct
        {
            T s, t;
        };
    };

    v2_() = default;
    explicit v2_(const T& value) : x{value}, y{value} {}
    v2_(const T& x, const T& y) : x{x}, y{y} {}

    v2_& normalize()
    {
        // TODO(gviv): Instead of computing the length, we should compute the
        // length squared and only sqrt it if it's non-zero (and check for
        // divide-by-zero!).
        T len = length();

        *this /= len;

        return *this;
    }

    T dot(const v2_& other) const
    {
        return x * other.x + y * other.y;
    }

    T length() const;

    T length2() const
    {
        return dot(*this);
    }
};

template<typename T> inline
v2_<T> operator+(const v2_<T>& vec1, const v2_<T>& vec2)
{
    return {
        vec1.x + vec2.x,
        vec1.y + vec2.y,
    };
}

template<typename T> inline
v2_<T> operator-(const v2_<T>& vec1, const v2_<T>& vec2)
{
    return {
        vec1.x - vec2.x,
        vec1.y - vec2.y,
    };
}

template<typename T> inline
v2_<T> operator-(const v2_<T>& vec)
{
    return {
        -vec.x,
        -vec.y,
    };
}

template<typename T> inline
v2_<T> operator*(const v2_<T>& vec1, const v2_<T>& vec2)
{
    return {
        vec1.x * vec2.x,
        vec1.y * vec2.y,
    };
}

template<typename T> inline
v2_<T> operator/(const v2_<T>& vec1, const v2_<T>& vec2)
{
    return {
        vec1.x / vec2.x,
        vec1.y / vec2.y,
    };
}

template<typename T> inline
v2_<T> operator+(const v2_<T>& vec, const T& value)
{
    return vec + v2_<T>{value, value};
}

template<typename T> inline
v2_<T> operator+(const T& value, const v2_<T>& vec)
{
    return vec + value;
}

template<typename T> inline
v2_<T> operator-(const v2_<T>& vec, const T& value)
{
    return vec - v2_<T>{value, value};
}

template<typename T> inline
v2_<T> operator-(const T& value, const v2_<T>& vec)
{
    return v2_<T>{value, value} - vec;
}

template<typename T> inline
v2_<T> operator*(const v2_<T>& vec, const T& value)
{
    return vec * v2_<T>{value, value};
}

template<typename T> inline
v2_<T> operator*(const T& value, const v2_<T>& vec)
{
    return vec * value;
}

template<typename T> inline
v2_<T> operator/(const v2_<T>& vec, const T& value)
{
    return vec / v2_<T>{value, value};
}


template<typename T> inline
v2_<T>& operator+=(v2_<T>& self, const v2_<T>& other)
{
    self = self + other;

    return self;
}

template<typename T> inline
v2_<T>& operator-=(v2_<T>& self, const v2_<T>& other)
{
    self = self - other;

    return self;
}

template<typename T> inline
v2_<T>& operator*=(v2_<T>& self, const v2_<T>& other)
{
    self = self * other;

    return self;
}

template<typename T> inline
v2_<T>& operator*=(v2_<T>& self, const T& value)
{
    self = self * v2_<T>{value, value};

    return self;
}

template<typename T> inline
v2_<T>& operator/=(v2_<T>& self, const v2_<T>& other)
{
    self = self / other;

    return self;
}

template<typename T> inline
v2_<T>& operator/=(v2_<T>& self, const T& value)
{
    self = self / v2_<T>{value, value};

    return self;
}

inline
v2_<f32_x8>& v2_<f32_x8>::normalize()
{
    f32_x8 len = _mm256_rsqrt_ps(length2());

    *this *= len;

    return *this;
}

inline
f32_x8 v2_<f32_x8>::length() const
{
    return _mm256_sqrt_ps(length2());
}

inline
f32 v2_<f32>::length() const
{
    return std::sqrtf(length2());
}

//
// Vector3
//
template<typename T>
struct v3_
{
    union
    {
        struct
        {
            T x, y, z;
        };
        struct
        {
            T r, g, b;
        };
        struct
        {
            T s, t, p;
        };
    };

    v3_() = default;
    explicit v3_(const T& value) : x{value}, y{value}, z{value} {}
    v3_(const T& x, const T& y, const T& z) : x{x}, y{y}, z{z} {}

    v3_ cross(const v3_& other) const
    {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    v3_& normalize()
    {
        // TODO(gviv): Instead of computing the length, we should compute the
        // length squared and only sqrt it if it's non-zero (and check for
        // divide-by-zero!).
        T len = length();

        *this /= len;

        return *this;
    }

    T dot(const v3_& other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    T length() const;

    T length2() const
    {
        return dot(*this);
    }
};

template<typename T> inline
v3_<T> operator+(const v3_<T>& vec1, const v3_<T>& vec2)
{
    return {
        vec1.x + vec2.x,
        vec1.y + vec2.y,
        vec1.z + vec2.z,
    };
}

template<typename T> inline
v3_<T> operator-(const v3_<T>& vec1, const v3_<T>& vec2)
{
    return {
        vec1.x - vec2.x,
        vec1.y - vec2.y,
        vec1.z - vec2.z,
    };
}

template<typename T> inline
v3_<T> operator-(const v3_<T>& vec)
{
    return {
        -vec.x,
        -vec.y,
        -vec.z
    };
}

template<typename T> inline
v3_<T> operator*(const v3_<T>& vec1, const v3_<T>& vec2)
{
    return {
        vec1.x * vec2.x,
        vec1.y * vec2.y,
        vec1.z * vec2.z,
    };
}

template<typename T> inline
v3_<T> operator/(const v3_<T>& vec1, const v3_<T>& vec2)
{
    return {
        vec1.x / vec2.x,
        vec1.y / vec2.y,
        vec1.z / vec2.z,
    };
}

template<typename T> inline
v3_<T> operator+(const v3_<T>& vec, const T& value)
{
    return vec + v3_<T>{value, value, value};
}

template<typename T> inline
v3_<T> operator+(const T& value, const v3_<T>& vec)
{
    return vec + value;
}

template<typename T> inline
v3_<T> operator-(const v3_<T>& vec, const T& value)
{
    return vec - v3_<T>{value, value, value};
}

template<typename T> inline
v3_<T> operator-(const T& value, const v3_<T>& vec)
{
    return v3_<T>{value, value, value} - vec;
}

template<typename T> inline
v3_<T> operator*(const v3_<T>& vec, const T& value)
{
    return vec * v3_<T>{value, value, value};
}

template<typename T> inline
v3_<T> operator*(const T& value, const v3_<T>& vec)
{
    return vec * value;
}

template<typename T> inline
v3_<T> operator/(const v3_<T>& vec, const T& value)
{
    return vec / v3_<T>{value, value, value};
}


template<typename T> inline
v3_<T>& operator+=(v3_<T>& self, const v3_<T>& other)
{
    self = self + other;

    return self;
}

template<typename T> inline
v3_<T>& operator-=(v3_<T>& self, const v3_<T>& other)
{
    self = self - other;

    return self;
}

template<typename T> inline
v3_<T>& operator*=(v3_<T>& self, const v3_<T>& other)
{
    self = self * other;

    return self;
}

template<typename T> inline
v3_<T>& operator*=(v3_<T>& self, const T& value)
{
    self = self * v3_<T>{value, value, value};

    return self;
}

template<typename T> inline
v3_<T>& operator/=(v3_<T>& self, const v3_<T>& other)
{
    self = self / other;

    return self;
}

template<typename T> inline
v3_<T>& operator/=(v3_<T>& self, const T& value)
{
    self = self / v3_<T>{value, value, value};

    return self;
}

inline
v3_<f32_x8>& v3_<f32_x8>::normalize()
{
    f32_x8 len = _mm256_rsqrt_ps(length2());

    *this *= len;

    return *this;
}

inline
f32_x8 v3_<f32_x8>::length() const
{
    return _mm256_sqrt_ps(length2());
}

inline
f32 v3_<f32>::length() const
{
    return std::sqrtf(length2());
}

template<typename T> inline
v3_<T> cross(const v3_<T>& vec1, const v3_<T>& vec2)
{
    return vec1.cross(vec2);
}

//
// Vector4
//
template<typename T>
struct v4_
{
    union
    {
        struct
        {
            T x, y, z, w;
        };
        struct
        {
            T r, g, b, a;
        };
        struct
        {
            T s, t, p, q;
        };
    };

    v4_() = default;
    explicit v4_(const T& value) : x{value}, y{value}, z{value}, w{value} {}
    v4_(const v3_<T>& vec, const T& w) : x{vec.x}, y{vec.y}, z{vec.z}, w{w} {}
    v4_(const T& x, const T& y, const T& z, const T& w) : x{x}, y{y}, z{z}, w{w} {}

    v4_& normalize()
    {
        // TODO(gviv): Instead of computing the length, we should compute the
        // length squared and only sqrt it if it's non-zero (and check for
        // divide-by-zero!).
        T len = length();

        *this /= len;

        return *this;
    }

    T dot(const v4_& other) const
    {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    T length() const;

    T length2() const
    {
        return dot(*this);
    }
};

template<typename T> inline
v4_<T> operator+(const v4_<T>& vec1, const v4_<T>& vec2)
{
    return {
        vec1.x + vec2.x,
        vec1.y + vec2.y,
        vec1.z + vec2.z,
        vec1.w + vec2.w,
    };
}

template<typename T> inline
v4_<T> operator-(const v4_<T>& vec1, const v4_<T>& vec2)
{
    return {
        vec1.x - vec2.x,
        vec1.y - vec2.y,
        vec1.z - vec2.z,
        vec1.w - vec2.w,
    };
}

template<typename T> inline
v4_<T> operator-(const v4_<T>& vec)
{
    return {
        -vec.x,
        -vec.y,
        -vec.z,
        -vec.w,
    };
}

template<typename T> inline
v4_<T> operator*(const v4_<T>& vec1, const v4_<T>& vec2)
{
    return {
        vec1.x * vec2.x,
        vec1.y * vec2.y,
        vec1.z * vec2.z,
        vec1.w * vec2.w,
    };
}

template<typename T> inline
v4_<T> operator/(const v4_<T>& vec1, const v4_<T>& vec2)
{
    return {
        vec1.x / vec2.x,
        vec1.y / vec2.y,
        vec1.z / vec2.z,
        vec1.w / vec2.w,
    };
}

template<typename T> inline
v4_<T>& operator+=(v4_<T>& self, const v4_<T>& other)
{
    self = self + other;

    return self;
}

template<typename T> inline
v4_<T> operator+(const v4_<T>& vec, const T& value)
{
    return vec + v4_<T>{value, value, value, value};
}

template<typename T> inline
v4_<T> operator+(const T& value, const v4_<T>& vec)
{
    return vec + value;
}

template<typename T> inline
v4_<T> operator-(const v4_<T>& vec, const T& value)
{
    return vec - v4_<T>{value, value, value, value};
}

template<typename T> inline
v4_<T> operator-(const T& value, const v4_<T>& vec)
{
    return v4_<T>{value, value, value, value} - vec;
}

template<typename T> inline
v4_<T> operator*(const v4_<T>& vec, const T& value)
{
    return vec * v4_<T>{value, value, value, value};
}

template<typename T> inline
v4_<T> operator*(const T& value, const v4_<T>& vec)
{
    return vec * value;
}

template<typename T> inline
v4_<T> operator/(const v4_<T>& vec, const T& value)
{
    return vec / v4_<T>{value, value, value, value};
}

template<typename T> inline
v4_<T>& operator-=(v4_<T>& self, const v4_<T>& other)
{
    self = self - other;

    return self;
}

template<typename T> inline
v4_<T>& operator*=(v4_<T>& self, const v4_<T>& other)
{
    self = self * other;

    return self;
}

template<typename T> inline
v4_<T>& operator*=(v4_<T>& self, const T& value)
{
    self = self * v4_<T>{value, value, value, value};

    return self;
}

template<typename T> inline
v4_<T>& operator/=(v4_<T>& self, const v4_<T>& other)
{
    self = self / other;

    return self;
}

template<typename T> inline
v4_<T>& operator/=(v4_<T>& self, const T& value)
{
    self = self / v4_<T>{value, value, value, value};

    return self;
}

inline
v4_<f32_x8>& v4_<f32_x8>::normalize()
{
    f32_x8 oneOverLength = _mm256_rsqrt_ps(length2());

    *this *= oneOverLength;

    return *this;
}

inline
f32_x8 v4_<f32_x8>::length() const
{
    return _mm256_sqrt_ps(length2());
}

inline
f32 v4_<f32>::length() const
{
    return std::sqrtf(length2());
}

using v2_x8 = v2_<f32_x8>;
using v3_x8 = v3_<f32_x8>;
using v4_x8 = v4_<f32_x8>;

using v2i_x8 = v2_<i32_x8>;
using v3i_x8 = v3_<i32_x8>;
using v4i_x8 = v4_<i32_x8>;

using v2 = v2_<f32>;
using v3 = v3_<f32>;
using v4 = v4_<f32>;

using v2i = v2_<i32>;
using v3i = v3_<i32>;
using v4i = v4_<i32>;

inline
v3_x8 makeV3_8x(const v3& vec)
{
    return {makeF32_x8(vec.x), makeF32_x8(vec.y), makeF32_x8(vec.z)};
}

inline
v4_x8 makeV4_x8(const f32& x, const f32& y, const f32& z, const f32& w)
{
    return {makeF32_x8(x), makeF32_x8(y), makeF32_x8(z), makeF32_x8(w)};
}

template<typename T>
struct m4_
{
    T e[4][4];

    static m4_ id;

    T* operator[](u8 i)
    {
        return e[i];
    }

    const T* operator[](u8 i) const
    {
        return e[i];
    }
};

using m4 = m4_<f32>;
using m4_x8 = m4_<f32_x8>;

m4 m4::id{
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f,
};

m4_x8 m4_x8::id{
    makeF32_x8(1.f), makeF32_x8(0.f), makeF32_x8(0.f), makeF32_x8(0.f),
    makeF32_x8(0.f), makeF32_x8(1.f), makeF32_x8(0.f), makeF32_x8(0.f),
    makeF32_x8(0.f), makeF32_x8(0.f), makeF32_x8(1.f), makeF32_x8(0.f),
    makeF32_x8(0.f), makeF32_x8(0.f), makeF32_x8(0.f), makeF32_x8(1.f),
};

inline
v4_x8 operator*(const m4_x8& mat, const v4_x8& vec)
{
    return {
        _mm256_fmadd_ps(mat[0][0], vec.x, _mm256_fmadd_ps(mat[0][1], vec.y, _mm256_fmadd_ps(mat[0][2], vec.z, mat[0][3] * vec.w))),
        _mm256_fmadd_ps(mat[1][0], vec.x, _mm256_fmadd_ps(mat[1][1], vec.y, _mm256_fmadd_ps(mat[1][2], vec.z, mat[1][3] * vec.w))),
        _mm256_fmadd_ps(mat[2][0], vec.x, _mm256_fmadd_ps(mat[2][1], vec.y, _mm256_fmadd_ps(mat[2][2], vec.z, mat[2][3] * vec.w))),
        _mm256_fmadd_ps(mat[3][0], vec.x, _mm256_fmadd_ps(mat[3][1], vec.y, _mm256_fmadd_ps(mat[3][2], vec.z, mat[3][3] * vec.w)))
    };
}

inline
m4_x8 operator*(const m4_x8& a, const m4_x8& b)
{
    m4_x8 result;

    result[0][0] = _mm256_fmadd_ps(a[0][0], b[0][0], _mm256_fmadd_ps(a[0][1], b[1][0], _mm256_fmadd_ps(a[0][2], b[2][0], a[0][3] * b[3][0])));
    result[0][1] = _mm256_fmadd_ps(a[0][0], b[0][1], _mm256_fmadd_ps(a[0][1], b[1][1], _mm256_fmadd_ps(a[0][2], b[2][1], a[0][3] * b[3][1])));
    result[0][2] = _mm256_fmadd_ps(a[0][0], b[0][2], _mm256_fmadd_ps(a[0][1], b[1][2], _mm256_fmadd_ps(a[0][2], b[2][2], a[0][3] * b[3][2])));
    result[0][3] = _mm256_fmadd_ps(a[0][0], b[0][3], _mm256_fmadd_ps(a[0][1], b[1][3], _mm256_fmadd_ps(a[0][2], b[2][3], a[0][3] * b[3][3])));

    result[1][0] = _mm256_fmadd_ps(a[1][0], b[0][0], _mm256_fmadd_ps(a[1][1], b[1][0], _mm256_fmadd_ps(a[1][2], b[2][0], a[1][3] * b[3][0])));
    result[1][1] = _mm256_fmadd_ps(a[1][0], b[0][1], _mm256_fmadd_ps(a[1][1], b[1][1], _mm256_fmadd_ps(a[1][2], b[2][1], a[1][3] * b[3][1])));
    result[1][2] = _mm256_fmadd_ps(a[1][0], b[0][2], _mm256_fmadd_ps(a[1][1], b[1][2], _mm256_fmadd_ps(a[1][2], b[2][2], a[1][3] * b[3][2])));
    result[1][3] = _mm256_fmadd_ps(a[1][0], b[0][3], _mm256_fmadd_ps(a[1][1], b[1][3], _mm256_fmadd_ps(a[1][2], b[2][3], a[1][3] * b[3][3])));

    result[2][0] = _mm256_fmadd_ps(a[2][0], b[0][0], _mm256_fmadd_ps(a[2][1], b[1][0], _mm256_fmadd_ps(a[2][2], b[2][0], a[2][3] * b[3][0])));
    result[2][1] = _mm256_fmadd_ps(a[2][0], b[0][1], _mm256_fmadd_ps(a[2][1], b[1][1], _mm256_fmadd_ps(a[2][2], b[2][1], a[2][3] * b[3][1])));
    result[2][2] = _mm256_fmadd_ps(a[2][0], b[0][2], _mm256_fmadd_ps(a[2][1], b[1][2], _mm256_fmadd_ps(a[2][2], b[2][2], a[2][3] * b[3][2])));
    result[2][3] = _mm256_fmadd_ps(a[2][0], b[0][3], _mm256_fmadd_ps(a[2][1], b[1][3], _mm256_fmadd_ps(a[2][2], b[2][3], a[2][3] * b[3][3])));

    result[3][0] = _mm256_fmadd_ps(a[3][0], b[0][0], _mm256_fmadd_ps(a[3][1], b[1][0], _mm256_fmadd_ps(a[3][2], b[2][0], a[3][3] * b[3][0])));
    result[3][1] = _mm256_fmadd_ps(a[3][0], b[0][1], _mm256_fmadd_ps(a[3][1], b[1][1], _mm256_fmadd_ps(a[3][2], b[2][1], a[3][3] * b[3][1])));
    result[3][2] = _mm256_fmadd_ps(a[3][0], b[0][2], _mm256_fmadd_ps(a[3][1], b[1][2], _mm256_fmadd_ps(a[3][2], b[2][2], a[3][3] * b[3][2])));
    result[3][3] = _mm256_fmadd_ps(a[3][0], b[0][3], _mm256_fmadd_ps(a[3][1], b[1][3], _mm256_fmadd_ps(a[3][2], b[2][3], a[3][3] * b[3][3])));

    return result;
}

inline
v4 operator*(const m4& mat, const v4& vec)
{
    return {
        mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z + mat[0][3] * vec.w,
        mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z + mat[1][3] * vec.w,
        mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z + mat[2][3] * vec.w,
        mat[3][0] * vec.x + mat[3][1] * vec.y + mat[3][2] * vec.z + mat[3][3] * vec.w
    };
}

inline
m4 operator*(const m4& a, const m4& b)
{
    m4 result;

    result[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
    result[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
    result[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
    result[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];

    result[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
    result[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
    result[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
    result[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];

    result[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
    result[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
    result[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
    result[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];

    result[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
    result[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
    result[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
    result[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];

    return result;
}

// Only works for transformation matrices!
inline
m4 fastInverse(const m4 &a)
{
    return {
        a[0][0], a[1][0], a[2][0], -a[0][3] * a[0][0] - a[1][3] * a[1][0] - a[2][3] * a[2][0],
        a[0][1], a[1][1], a[2][1], -a[0][3] * a[0][1] - a[1][3] * a[1][1] - a[2][3] * a[2][1],
        a[0][2], a[1][2], a[2][2], -a[0][3] * a[0][2] - a[1][3] * a[1][2] - a[2][3] * a[2][2],
        0.f, 0.f, 0.f, 1.f
    };
}

// Only works for transformation matrices!
inline
m4_x8 fastInverse(const m4_x8 &a)
{
    const f32_x8 zero = _mm256_setzero_ps();
    return {
        a[0][0], a[1][0], a[2][0], -a[0][3] * a[0][0] - a[1][3] * a[1][0] - a[2][3] * a[2][0],
        a[0][1], a[1][1], a[2][1], -a[0][3] * a[0][1] - a[1][3] * a[1][1] - a[2][3] * a[2][1],
        a[0][2], a[1][2], a[2][2], -a[0][3] * a[0][2] - a[1][3] * a[1][2] - a[2][3] * a[2][2],
        zero, zero, zero, 1.f
    };
}

inline
m4_x8 makeM4_x8(
    f32 e00, f32 e01, f32 e02, f32 e03,
    f32 e10, f32 e11, f32 e12, f32 e13,
    f32 e20, f32 e21, f32 e22, f32 e23,
    f32 e30, f32 e31, f32 e32, f32 e33)
{
    return {
        makeF32_x8(e00), makeF32_x8(e01), makeF32_x8(e02), makeF32_x8(e03),
        makeF32_x8(e10), makeF32_x8(e11), makeF32_x8(e12), makeF32_x8(e13),
        makeF32_x8(e20), makeF32_x8(e21), makeF32_x8(e22), makeF32_x8(e23),
        makeF32_x8(e30), makeF32_x8(e31), makeF32_x8(e32), makeF32_x8(e33),
    };
}

// Assumes that `p` is a point, so it's affected by the translation part of the
// matrix. Do not support projective transformations.
template<typename T> inline
v3_<T> multMat44Point3(const m4_<T>& mat, const v3_<T>& p)
{
    return {
        mat[0][0] * p.x + mat[0][1] * p.y + mat[0][2] * p.z + mat[0][3],
        mat[1][0] * p.x + mat[1][1] * p.y + mat[1][2] * p.z + mat[1][3],
        mat[2][0] * p.x + mat[2][1] * p.y + mat[2][2] * p.z + mat[2][3]
    };
}

// Assumes that `vec` is a vector, so it's not affected by the translation part
// of the matrix. Do not support projective transformations.
template<typename T> inline
v3_<T> multMat44Vec3(const m4_<T>& mat, const v3_<T>& vec)
{
    return {
        mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z,
        mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z,
        mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z
    };
}

#endif
