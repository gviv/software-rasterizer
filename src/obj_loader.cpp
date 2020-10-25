#include <fstream>
#include <string>
#include <vector>

#include "rasterizer.h"

namespace
{

inline
bool isSpace(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

inline
bool isAtEnd(const char* str)
{
    return *str == '\0';
}

struct Lexer
{
    const char* current;

    template<typename ...Args>
    bool eat(Args&&... args)
    {
        const char* start = current;
        bool success = (eatOne(std::forward<Args>(args)) && ...);

        if (!success)
        {
            // We could not get a match between the requested arguments and the
            // next characters in the stream, so we rollback to the starting
            // location.
            current = start;
            return false;
        }

        return true;
    }

private:
    void skipSpaces()
    {
        while (!isAtEnd(current) && isSpace(*current)) ++current;
    }

    template<typename T>
    bool eatOne(T&& arg)
    {
        skipSpaces();

        if constexpr (std::is_same_v<std::decay_t<T>, const char*>)
        {
            return nextStringEquals(std::forward<T>(arg));
        }
        else
        {
            return tryToExtract(std::forward<T>(arg));
        }
    }

    template<typename T>
    bool tryToExtract(T&&)
    {
        T ignored;

        return tryToExtract(ignored);
    }

    // Advances in the stream, seeking for the requested type and writes its
    // value in `out` if any.
    template<typename T>
    bool tryToExtract(T& out)
    {
        bool success = false;

        if constexpr (std::is_same_v<T, std::string>)
        {
            // Whatever follows will be considered a string.
            success = true;
            out = lexString();
        }
        else if constexpr (std::is_same_v<T, f32> || std::is_same_v<T, i32>)
        {
            // The following token should be a number.
            char* charAfterNumber;
            auto number = [this, &charAfterNumber]()
            {
                if constexpr (std::is_same_v<T, f32>)
                {
                    return std::strtof(current, &charAfterNumber);
                }
                else
                {
                    return (i32)std::strtol(current, &charAfterNumber, 10);
                }
            }();

            if (charAfterNumber != current)
            {
                // A number has been read successfully.
                success = true;
                out = number;
                current = charAfterNumber;
            }
        }
        else
        {
            static_assert(false, "Unsupported type for extraction");
        }

        return success;
    }

    bool nextStringEquals(const char* str)
    {
        while (!isAtEnd(str) && !isAtEnd(current))
        {
            if (*current++ != *str++) return false;
        }

        return true;
    }

    std::string lexString()
    {
        if (isAtEnd(current)) return {};
        const char* start = current;

        // TODO(gviv): When we want to lex a string, we should eat everything
        // until the end of the line, not until a space character (because
        // in statements like `mtllib filename`, the filename can be whatever,
        // including spaces).
        while (!isSpace(*current) && !isAtEnd(current)) ++current;

        std::size_t size = (std::size_t)(current - start);

        return {start, size};
    }

};

template<typename T> inline
T extractAttribute(const std::vector<T>& attributes, i32 index)
{
    return attributes[index < 0 ? attributes.size() + index : index - 1];
}

inline
Vertex makeVertex(
    const std::vector<v3>& positions,
    const std::vector<v2>& texCoords,
    const std::vector<v3>& normals,
    i32 v, i32 vt, i32 vn)
{
    Vertex result{};

    result.position = extractAttribute(positions, v);
    result.texCoords = extractAttribute(texCoords, vt);
    result.normal = extractAttribute(normals, vn);

    return result;
}

inline
Vertex makeVertex(
    const std::vector<v3>& positions,
    const std::vector<v2>& texCoords,
    i32 v, i32 vt)
{
    Vertex result{};

    result.position = extractAttribute(positions, v);
    result.texCoords = extractAttribute(texCoords, vt);

    return result;
}

inline
Vertex makeVertex(
    const std::vector<v3>& positions,
    const std::vector<v3>& normals,
    i32 v, i32 vn)
{
    Vertex result{};

    result.position = extractAttribute(positions, v);
    result.normal = extractAttribute(normals, vn);

    return result;
}

inline
Vertex makeVertex(const std::vector<v3>& positions, i32 v)
{
    Vertex result{};

    result.position = extractAttribute(positions, v);

    return result;
}

inline
v3 computeNormal(const v3& pos1, const v3& pos2, const v3& pos3)
{
    return (pos2 - pos1).cross(pos3 - pos1);
}

inline
void addVertices(
    Mesh& mesh,
    const Vertex& v1,
    const Vertex& v2,
    const Vertex& v3,
    u32& curIndex)
{
    mesh.vertices.push_back(v1);
    mesh.vertices.push_back(v2);
    mesh.vertices.push_back(v3);
    mesh.indices.push_back(curIndex++);
    mesh.indices.push_back(curIndex++);
    mesh.indices.push_back(curIndex++);
}

inline
void addVertices(
    Mesh& mesh,
    const Vertex& v1,
    const Vertex& v2,
    const Vertex& v3,
    const Vertex& v4,
    u32& curIndex)
{
    // Triangulate the four vertices.
    mesh.vertices.push_back(v1);
    mesh.vertices.push_back(v2);
    mesh.vertices.push_back(v3);
    mesh.indices.push_back(curIndex);
    mesh.indices.push_back(curIndex + 1);
    mesh.indices.push_back(curIndex + 2);

    mesh.vertices.push_back(v1);
    mesh.vertices.push_back(v3);
    mesh.vertices.push_back(v4);
    mesh.indices.push_back(curIndex);
    mesh.indices.push_back(curIndex + 2);
    mesh.indices.push_back(curIndex + 3);

    curIndex += 4;
}

}  // namespace

Model loadObj(const char* filename)
{
    Lexer lexer;
    std::ifstream is{filename};
    std::string line;
    std::vector<v3> positions;
    std::vector<v2> texCoords;
    std::vector<v3> normals;

    // TODO(gviv): The vertices should share indices. Right now, each vertex has
    // its own index.
    u32 curVertexIndex = 0;

    Model model;
    model.meshes.resize(1);

    // TODO(gviv): Support line continuation.
    while (std::getline(is, line))
    {
        lexer.current = line.c_str();

        if (lexer.eat("vt"))
        {
            f32 u, v = 0.f;

            if (!lexer.eat(u))
            {
                // Error: Expected texture coordinate.
            }
            lexer.eat(v);

            texCoords.push_back({u, v});

            if (lexer.eat(f32{}))
            {
                // Error: 3D texture coordinates unsupported.
            }
        }
        else if (lexer.eat("vn"))
        {
            f32 x, y, z;

            if (!lexer.eat(x, y, z))
            {
                // Error: Expected normal coordinates.
            }

            normals.push_back({x, y, z});
        }
        else if (lexer.eat("v"))
        {
            f32 x, y, z;

            if (!lexer.eat(x, y, z))
            {
                // Error: Expected vertex positions.
            }

            positions.push_back({x, y, z});

            if (lexer.eat(f32{}))
            {
                // Error: 4D vertices unsupported.
            }
        }
        else if (lexer.eat("f"))
        {
            i32 v1, v2, v3, v4;
            i32 vt1, vt2, vt3, vt4;
            i32 vn1, vn2, vn3, vn4;

            if (lexer.eat(v1, "/", vt1, "/", vn1))
            {
                // Format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 [v4/vt4/vn4]
                if (!lexer.eat(
                        v2, "/", vt2, "/", vn2,
                        v3, "/", vt3, "/", vn3))
                {
                    // Error: Invalid triplet format (expected `v/vt/vn`).
                }

                Vertex vertex1 =
                    makeVertex(positions, texCoords, normals, v1, vt1, vn1);
                Vertex vertex2 =
                    makeVertex(positions, texCoords, normals, v2, vt2, vn2);
                Vertex vertex3 =
                    makeVertex(positions, texCoords, normals, v3, vt3, vn3);

                if (lexer.eat(v4, "/", vt4, "/", vn4))
                {
                    Vertex vertex4 =
                        makeVertex(positions, texCoords, normals, v4, vt4, vn4);
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, vertex4, curVertexIndex);
                }
                else
                {
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, curVertexIndex);
                }
            }
            else if (lexer.eat(v1, v2, v3))
            {
                // Format: f v1 v2 v3 [v4]
                Vertex vertex1 = makeVertex(positions, v1);
                Vertex vertex2 = makeVertex(positions, v2);
                Vertex vertex3 = makeVertex(positions, v3);

                ::v3 faceNormal = computeNormal(
                    vertex1.position, vertex2.position, vertex3.position);
                vertex1.normal = faceNormal;
                vertex2.normal = faceNormal;
                vertex3.normal = faceNormal;

                if (lexer.eat(v4))
                {
                    Vertex vertex4 = makeVertex(positions, v4);
                    vertex4.normal = faceNormal;
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, vertex4, curVertexIndex);
                }
                else
                {
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, curVertexIndex);
                }
            }
            else if (lexer.eat(v1, "//", vn1))
            {
                // Format: f v1//vn1 v2//vn2 v3//vn3 [v4//vn4]
                if (!lexer.eat(
                        v2, "//", vn2,
                        v3, "//", vn3))
                {
                    // Error: Invalid triplet format (expected `v//vn`).
                }

                Vertex vertex1 = makeVertex(positions, normals, v1, vn1);
                Vertex vertex2 = makeVertex(positions, normals, v2, vn2);
                Vertex vertex3 = makeVertex(positions, normals, v3, vn3);

                if (lexer.eat(v4, "//", vn4))
                {
                    Vertex vertex4 = makeVertex(positions, normals, v4, vn4);
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, vertex4, curVertexIndex);
                }
                else
                {
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, curVertexIndex);
                }
            }
            else if (lexer.eat(v1, "/", vt1))
            {
                // Format: f v1/vt1 v2/vt2 v3/vt3 [v4/vt4]
                if (!lexer.eat(
                        v2, "/", vt2,
                        v3, "/", vt3))
                {
                    // Error: Invalid triplet format (expected `v/vt`).
                }

                Vertex vertex1 = makeVertex(positions, texCoords, v1, vt1);
                Vertex vertex2 = makeVertex(positions, texCoords, v2, vt2);
                Vertex vertex3 = makeVertex(positions, texCoords, v3, vt3);

                ::v3 faceNormal = computeNormal(
                    vertex1.position, vertex2.position, vertex3.position);
                vertex1.normal = faceNormal;
                vertex2.normal = faceNormal;
                vertex3.normal = faceNormal;

                if (lexer.eat(v4, "/", vt4))
                {
                    Vertex vertex4 = makeVertex(positions, texCoords, v4, vt4);
                    vertex4.normal = faceNormal;
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, vertex4, curVertexIndex);
                }
                else
                {
                    addVertices(model.meshes[0], vertex1, vertex2, vertex3, curVertexIndex);
                }
            }
        }
    }

    return model;
}
