#ifndef VOXIUM_H_
#define VOXIUM_H_

#include <vector>
#include <string>

namespace vx {

class Voxmap;

struct Color {
    float r;
    float g;
    float b;
    float a;

    Color() = default;
    Color(float r, float g, float b);
    Color(float r, float g, float b, float a);

    Color& multiply(float n);

    static const Color TRANSPARENT, WHITE, BLACK, VERYLIGHTGRAY, LIGHTGRAY,
        GRAY, DARKGRAY, VERYDARKGRAY, VERYLIGHTRED, LIGHTRED, RED, DARKRED,
        VERYDARKRED, VERYLIGHTGREEN, LIGHTGREEN, GREEN, DARKGREEN,
        VERYDARKGREEN, VERYLIGHTBLUE, LIGHTBLUE, BLUE, DARKBLUE, VERYDARKBLUE,
        VERYLIGHTMAGENTA, LIGHTMAGENTA, MAGENTA, DARKMAGENTA, VERYDARKMAGENTA,
        VERYLIGHTCYAN, LIGHTCYAN, CYAN, DARKCYAN, VERYDARKCYAN, VERYLIGHTYELLOW,
        LIGHTYELLOW, YELLOW, DARKYELLOW, VERYDARKYELLOW, VERYLIGHTORANGE,
        LIGHTORANGE, ORANGE, DARKORANGE, VERYDARKORANGE, VERYLIGHTBROWN,
        LIGHTBROWN, BROWN, DARKBROWN, VERYDARKBROWN, VERYLIGHTPINK, LIGHTPINK,
        PINK, DARKPINK, VERYDARKPINK, VERYLIGHTPURPLE, LIGHTPURPLE, PURPLE,
        DARKPURPLE, VERYDARKPURPLE;
};

struct col {
    unsigned char r, g, b, a;
};

class Voxmap_ {
   public:
    int width;
    int height;
    int depth;

    Voxmap_() = default;

    Voxmap_(int width, int height, int depth);

    void resize(int width, int height, int depth);
    void setVoxel(int x, int y, int z, const Color& voxel);
    void setVoxelUnchecked(int x, int y, int z, const Color& voxel);
    Color getColor(int x, int y, int z) const;
    Color getColorUnchecked(int x, int y, int z) const;
    unsigned int* getPixels();

    std::vector<col> colors;
};

template<typename T>
struct Vector2_ {
    T x, y;

    Vector2_() = default;
    explicit Vector2_(T x);
    Vector2_(T x, T y);

    T dot(const Vector2_<T>& other) const;

    Vector2_<T> perp() const {
        return {-y, x};
    }

    bool equals(const Vector2_<T>& other) const;
    bool equals(T x, T y) const;

    T length() const;
    T length2() const;

    Vector2_& normalize();

    Vector2_& operator+=(const Vector2_& other);
    Vector2_& operator+=(T n);

    Vector2_& operator-=(const Vector2_& other);
    Vector2_& operator-=(T n);

    Vector2_& operator*=(const Vector2_& other);
    Vector2_& operator*=(T n);

    Vector2_& operator/=(const Vector2_& other);
    Vector2_& operator/=(T n);

    Vector2_& set(const Vector2_& other);
    Vector2_& set(T x, T y);

    Vector2_& add(const Vector2_& other);
    Vector2_& add(T x, T y);
    Vector2_& add(T n);

    Vector2_& subtract(const Vector2_& other);
    Vector2_& subtract(T x, T y);
    Vector2_& subtract(T n);

    Vector2_& multiply(const Vector2_& other);
    Vector2_& multiply(T x, T y);
    Vector2_& multiply(T n);

    Vector2_& divide(const Vector2_& other);
    Vector2_& divide(T x, T y);
    Vector2_& divide(T n);
};

template<typename T>
Vector2_<T> operator+(const Vector2_<T>& v1, const Vector2_<T>& v2);
template<typename T>
Vector2_<T> operator+(const Vector2_<T>& vec, T n);
template<typename T>
Vector2_<T> operator+(T n, const Vector2_<T>& vec);

template<typename T>
Vector2_<T> operator-(const Vector2_<T>& vec);
template<typename T>
Vector2_<T> operator-(const Vector2_<T>& v1, const Vector2_<T>& v2);
template<typename T>
Vector2_<T> operator-(const Vector2_<T>& vec, float n);
template<typename T>
Vector2_<T> operator-(T n, const Vector2_<T>& vec);

template<typename T>
Vector2_<T> operator*(const Vector2_<T>& v1, const Vector2_<T>& v2);
template<typename T>
Vector2_<T> operator*(const Vector2_<T>& vec, T n);
template<typename T>
Vector2_<T> operator*(T n, const Vector2_<T>& vec);

template<typename T>
Vector2_<T> operator/(const Vector2_<T>& v1, const Vector2_<T>& v2);
template<typename T>
Vector2_<T> operator/(const Vector2_<T>& vec, T n);
template<typename T>
Vector2_<T> operator/(T n, const Vector2_<T>& vec);

template<typename T> inline
Vector2_<T> perp(const Vector2_<T>& v) {
    return {-v.y, v.x};
}

template<typename T>
struct Vector3_ {
    T x, y, z;

    Vector3_() = default;
    Vector3_(T x) : x{x}, y{x}, z{x} {}
    Vector3_(T x, T y, T z) : x{x}, y{y}, z{z} {}

    bool equals(const Vector3_<T>& other) {
        return abs(x - other.x) <= VECTOR_COMPARISON_EPS &&
               abs(y - other.y) <= VECTOR_COMPARISON_EPS &&
               abs(z - other.z) <= VECTOR_COMPARISON_EPS;
    }

    Vector3_<T>& set(T x, T y, T z) {
        this->x = x;
        this->y = y;
        this->z = z;

        return *this;
    }

    Vector3_<T>& normalize() {
        if (length2() > 0.f) {
            *this /= length();
        }

        return *this;
    }

    T length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    T dot(const Vector3_<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    T length2() const {
        return x * x + y * y + z * z;
    }

    Vector3_<T>& operator+=(const Vector3_<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    Vector3_<T>& operator-=(const Vector3_<T>& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;

        return *this;
    }

    Vector3_<T>& operator*=(const Vector3_<T>& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;

        return *this;
    }

    Vector3_<T>& operator*=(T n) {
        x *= n;
        y *= n;
        z *= n;

        return *this;
    }

    Vector3_<T>& operator/=(const Vector3_<T>& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;

        return *this;
    }

    Vector3_<T>& operator/=(T n) {
        x /= n;
        y /= n;
        z /= n;

        return *this;
    }

    Vector3_<T>& add(const Vector3_<T>& other) {
        return *this += other;
    }

    Vector3_<T>& subtract(const Vector3_<T>& other) {
        return *this -= other;
    }

    Vector3_<T>& multiply(const Vector3_<T>& other) {
        return *this *= other;
    }

    Vector3_<T>& multiply(T n) {
        return *this *= n;
    }

    Vector3_<T>& divide(const Vector3_<T>& other) {
        return *this /= other;
    }

    Vector3_<T>& divide(T n) {
        return *this /= n;
    }

    Vector3_<T> cross(const Vector3_<T>& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }
};

template<typename T>
inline Vector3_<T> operator+(Vector3_<T> v1, const Vector3_<T>& v2) {
    return v1 += v2;
}

template<typename T>
inline Vector3_<T> operator-(const Vector3_<T>& vec) {
    return v3{0.f, 0.f, 0.f} - vec;
}

template<typename T>
inline Vector3_<T> operator-(Vector3_<T> v1, const Vector3_<T>& v2) {
    return v1 -= v2;
}

template<typename T>
inline Vector3_<T> operator*(Vector3_<T> v1, const Vector3_<T>& v2) {
    return v1 *= v2;
}

template<typename T>
inline Vector3_<T> operator*(Vector3_<T> v1, T n) {
    return v1 *= n;
}

template<typename T>
inline Vector3_<T> operator*(T n, Vector3_<T> v1) {
    return v1 *= n;
}

template<typename T>
inline Vector3_<T> operator/(Vector3_<T> v1, const Vector3_<T>& v2) {
    return v1 /= v2;
}

template<typename T>
inline Vector3_<T> operator/(Vector3_<T> v1, T n) {
    return v1 /= n;
}

using Vector2 = Vector2_<float>;

enum class Mouse { LEFT, MIDDLE, RIGHT, LAST = RIGHT };

enum class Key {
    NUM_LOCK = static_cast<int>(Mouse::LAST) + 1,
    ESCAPE,
    SPACE,
    BACKSPACE,
    PAGE_UP,
    PAGE_DOWN,
    TAB,
    ENTER,
    SHIFT,
    LEFT,
    UP,
    RIGHT,
    DOWN,

    ZERO = 48,
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,

    A = 65,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    LAST = Z
};

class BaseApp {
   public:
    virtual void init() {}
    virtual void update(float dt) {}
    virtual void draw() {}

    virtual void onKeyDown(Key key) {}
    virtual void onKeyUp(Key key) {}
    virtual void onMouseDown(Mouse button) {}
    virtual void onMouseUp(Mouse button) {}
    virtual void onMouseMove(float x, float y) {}
    virtual void onFileDrop(const std::string& path) {}
};

void run(BaseApp& app, int clientWidth, int clientHeight, int voxelScreenWidth,
         int voxelScreenHeight, int voxelScreenDepth = 1);
void run(int windowWidth, int windowHeight, int voxelScreenWidth,
         int voxelScreenHeight, int voxelScreenDepth = 1);

template<typename SubType>
class DrawCommand {
   public:
    SubType& red(float value);
    SubType& green(float value);
    SubType& blue(float value);
    SubType& alpha(float value);

    Voxmap_* buffer;
    Color color;
};

class ClearCommand : public DrawCommand<ClearCommand>{
   public:
    void clear(Color color);
};

class RectangleCommand : public DrawCommand<RectangleCommand> {
   public:
    RectangleCommand(const RectangleCommand&) = delete;
    RectangleCommand(RectangleCommand&&) = delete;
    RectangleCommand& operator=(const RectangleCommand&) = delete;
    RectangleCommand& operator=(RectangleCommand&&) = delete;

    RectangleCommand& fill();
    RectangleCommand& rotate(float angle);
    RectangleCommand& center(const Vector2& center);
    RectangleCommand& scale(float scalev);
    RectangleCommand& texture(const Voxmap& voxmap);
    RectangleCommand& normalMap(const Voxmap& voxmap);

public:
    float x;
    float y;
    float width;
    float height;
    bool filled;
    float angle;
    float scaleValue = 1.f;
    Vector2 centerPoint;
    const Voxmap_* bitmap;
    const Voxmap_* nmap;

    ~RectangleCommand();

    friend RectangleCommand rectangle(float, float, float, float, Color);
};

class CircleCommand : public DrawCommand<CircleCommand> {
   public:
    CircleCommand& fill();

    CircleCommand(const CircleCommand&) = delete;
    CircleCommand(CircleCommand&&) = delete;
    CircleCommand& operator=(const CircleCommand&) = delete;
    CircleCommand& operator=(CircleCommand&&) = delete;

public:
    float x;
    float y;
    float radius;
    bool filled;

    ~CircleCommand();

    friend CircleCommand circle(float, float, float, Color);
};


class LineCommand : public DrawCommand<LineCommand> {
    public:
    float x1;
    float y1;
    float x2;
    float y2;

    LineCommand(const LineCommand&) = delete;
    LineCommand(LineCommand&&) = delete;
    LineCommand& operator=(const LineCommand&) = delete;
    LineCommand& operator=(LineCommand&&) = delete;

    ~LineCommand();

    friend LineCommand line(float, float, float, float, Color);
};

class TextCommand : public DrawCommand<TextCommand> {
   public:
    TextCommand& scale(float value);

    std::string str;
    float x;
    float y;
    float scaleValue = 1.f;

    ~TextCommand();

    friend TextCommand text(const std::string&, float, float, Color);
};

class PixelCommand : public DrawCommand<PixelCommand> {
public:
    float x;
    float y;

    PixelCommand(const PixelCommand&) = delete;
    PixelCommand(PixelCommand&&) = delete;
    PixelCommand& operator=(const PixelCommand&) = delete;
    PixelCommand& operator=(PixelCommand&&) = delete;

    ~PixelCommand();

    friend PixelCommand pixel(float, float, Color);
};

class VoxelCommand : public DrawCommand<VoxelCommand> {
public:
    float x;
    float y;
    float z;

    VoxelCommand(const VoxelCommand&) = delete;
    VoxelCommand(VoxelCommand&&) = delete;
    VoxelCommand& operator=(const VoxelCommand&) = delete;
    VoxelCommand& operator=(VoxelCommand&&) = delete;

    ~VoxelCommand();

    friend VoxelCommand voxel(float, float, float, Color);
};

class BlitCommand : public DrawCommand<BlitCommand> {
   public:
    Voxmap_* source;
    int x, y, z;

    ~BlitCommand();

    friend BlitCommand blit(Voxmap& source, int x, int y, int z);
};

class VoxmapCommand : public DrawCommand<VoxmapCommand> {
public:
    VoxmapCommand& rotate(float angle);
    VoxmapCommand& center(const Vector2& center);
    VoxmapCommand& scale(float scalev);


    Voxmap_* source;
    float x;
    float y;
    float angle;
    float scaleValue = 1.f;
    Vector2 centerPoint;

    ~VoxmapCommand();

    friend VoxmapCommand voxmap(Voxmap& source, float x, float y);
};


class Voxmap {
   public:
    Voxmap() = default;
    Voxmap(const std::string& uri);

    Voxmap(int width, int height, int depth);

    int width() const;

    int height() const;

    int depth() const;

    Color getVoxel(int x, int y, int z) const;

    VoxelCommand voxel(float x, float y, float z, Color color);

    Voxmap_ internalVoxmap;
};



void clear(Color color = Color::BLACK);
PixelCommand pixel(int x, int y, Color color);
PixelCommand pixel(Vector2 pos, Color color);

VoxelCommand voxel(float x, float y, float z, Color color);

RectangleCommand rectangle(float x, float y, float width, float height,
                            Color color);
RectangleCommand rectangle(Vector2 pos, Vector2 dimension, Color color);

CircleCommand circle(float x, float y, float radius, Color color);
CircleCommand circle(Vector2 pos, float radius, Color color);

LineCommand line(float x1, float y1, float x2, float y2, Color color);
LineCommand line(Vector2 start, Vector2 end, Color color);

TextCommand text(const std::string& str, float x, float y,
                  Color color = Color::WHITE);

VoxmapCommand voxmap(Voxmap& source, float x, float y);

BlitCommand blit(Voxmap& source, int x, int y, int z);

Color getVoxel(int x, int y, int z);

void flip();
void flip(float framerate);
void flipUnsync();
void flipUnsync(float framerate);
float deltaTime();

float framerate();
void framerate(float value);

void screenSize(int width, int height, int depth = 1);
void windowSize(int width, int height);

unsigned int* getPixels();

int windowWidth();
int width();
int height();
int depth();
float seconds();

bool isDown(Key key);
bool isDown(Mouse button);
bool isUp(Key key);
bool isUp(Mouse button);

float mouseXWindow();
float mouseX();
float mouseY();
Vector2 mousePos();

}  // namespace vx

#endif  // VOXIUM_H_
