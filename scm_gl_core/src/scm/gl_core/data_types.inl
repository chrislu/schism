
#include <cassert>

namespace scm {
namespace gl {

inline
int
size_of_type(data_type d)
{
    static int type_sizes[] = {
        1,
        // float
        sizeof(float), sizeof(float) * 2, sizeof(float) * 3, sizeof(float) * 4,
        // matrices
        sizeof(float) * 4, sizeof(float) * 9, sizeof(float) * 16,
        sizeof(float) * 2 * 3, sizeof(float) * 2 * 4, sizeof(float) * 3 * 2,
        sizeof(float) * 3 * 4, sizeof(float) * 4 * 2, sizeof(float) * 4 * 3,
        // int
        sizeof(int), sizeof(int) * 2, sizeof(int) * 3, sizeof(int) * 4,
        // unsigned int
        sizeof(unsigned), sizeof(unsigned) * 2, sizeof(unsigned) * 3, sizeof(unsigned) * 4,
        // bool
        sizeof(unsigned), sizeof(unsigned) * 2, sizeof(unsigned) * 3, sizeof(unsigned) * 4,
        // small int types
        sizeof(short),
        sizeof(unsigned short),
        sizeof(char),
        sizeof(unsigned char)
    };

    assert((sizeof(type_sizes) / sizeof(int)) == TYPE_COUNT);
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);

    return (type_sizes[d]);
}

inline
int
components(data_type d)
{
    static int component_counts[] = {
        1,
        // float
        1, 2, 3, 4,
        // matrices
        4, 9, 16,
        2 * 3, 2 * 4, 3 * 2,
        3 * 4, 4 * 2, 4 * 3,
        // int
        1, 2, 3, 4,
        // unsigned int
        1, 2, 3, 4,
        // bool
        1, 2, 3, 4,
        // small int types
        1,
        1,
        1,
        1
    };

    assert((sizeof(component_counts) / sizeof(int)) == TYPE_COUNT);
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);

    return (component_counts[d]);
}

inline
bool is_integer_type(data_type d)
{
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);
    return (TYPE_INT <= d && d <= TYPE_UBYTE);
}

inline
bool is_float_type(data_type d)
{
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);
    return (TYPE_FLOAT <= d && d <= TYPE_MAT4X3F);
}

} // namespace gl
} // namespace scm