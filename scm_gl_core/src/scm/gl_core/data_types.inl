
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
        sizeof(unsigned), sizeof(unsigned) * 2, sizeof(unsigned) * 3, sizeof(unsigned) * 4
    };

    assert((sizeof(type_sizes) / sizeof(int)) == TYPE_COUNT);
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);

    return (type_sizes[d]);
}

} // namespace gl
} // namespace scm