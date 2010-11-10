
#ifndef SCM_GL_CORE_DATA_FORMATS_INL_INCLUDED
#define SCM_GL_CORE_DATA_FORMATS_INL_INCLUDED

#include <cassert>

#include <boost/static_assert.hpp>

namespace scm {
namespace gl {

inline
bool is_normalized(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_R_8 <= d && d <= FORMAT_SRGBA_8);
}

inline
bool is_unnormalized(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_R_8I <= d && d <= FORMAT_RGBA_32UI);
}

inline
bool is_integer_type(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_R_8 <= d && d <= FORMAT_RGBA_32UI);
}

inline
bool is_float_type(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (   FORMAT_R_16F <= d && d <= FORMAT_R11B11G10F
            || FORMAT_D32F == d
            || FORMAT_D32F_S8 == d);
}

inline
bool is_depth_format(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_D16 <= d && d <= FORMAT_D32F_S8);
}

inline
bool is_stencil_format(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_D24_S8 <= d && d <= FORMAT_D32F_S8);
}

inline
bool is_packed_format(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_RGB9_E5 <= d && d <= FORMAT_R11B11G10F);
}

inline
bool is_plain_format(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (d <= FORMAT_RGBA_32F);
}

inline
bool is_srgb_format(data_format d)
{
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (FORMAT_SRGB_8 <= d && d <= FORMAT_SRGBA_8);
}

inline
int channel_count(data_format d)
{
    static int channel_counts[] = {
        0,
        // normalized integer formats (NORM)
        1, 2, 3, 4, // FORMAT_RGBA_8
        1, 2, 3, 4, // FORMAT_RGBA_16
        1, 2, 3, 4, // FORMAT_RGBA_8S
        1, 2, 3, 4, // FORMAT_RGBA_16S
        // swizzled integer formats
        3, 4,       // FORMAT_BGRA_8
        // srgb integer formats
        3, 4,       // FORMAT_SRGB_8
        // unnormalized integer formats (UNORM)
        1, 2, 3, 4, // FORMAT_RGBA_8I
        1, 2, 3, 4, // FORMAT_RGBA_16I
        1, 2, 3, 4, // FORMAT_RGBA_32I
        1, 2, 3, 4, // FORMAT_RGBA_8UI
        1, 2, 3, 4, // FORMAT_RGBA_16UI
        1, 2, 3, 4, // FORMAT_RGBA_32UI
        // floating point formats
        1, 2, 3, 4, // FORMAT_RGBA_16F
        1, 2, 3, 4, // FORMAT_RGBA_32F
        // special packed formats
        3, 3,
        // depth stencil formats
        1, 1, 1, 1, 2, 2
    };

    BOOST_STATIC_ASSERT((sizeof(channel_counts) / sizeof(int)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (channel_counts[d]);
}

inline
int size_of_channel(data_format d)
{
    static int channel_sizes[] = {
        1,
        // normalized integer formats (NORM)
        1, 1, 1, 1, // FORMAT_RGBA_8
        2, 2, 2, 2, // FORMAT_RGBA_16
        1, 1, 1, 1, // FORMAT_RGBA_8S
        2, 2, 2, 2, // FORMAT_RGBA_16S
        // swizzled integer formats
        1, 1,       // FORMAT_BGRA_8
        // srgb integer formats
        1, 1,       // FORMAT_SRGB_8
        // unnormalized integer formats (UNORM)
        1, 1, 1, 1, // FORMAT_RGBA_8I
        2, 2, 2, 2, // FORMAT_RGBA_16I
        4, 4, 4, 4, // FORMAT_RGBA_32I
        1, 1, 1, 1, // FORMAT_RGBA_8UI
        2, 2, 2, 2, // FORMAT_RGBA_16UI
        4, 4, 4, 4, // FORMAT_RGBA_32UI
        // floating point formats
        2, 2, 2, 2, // FORMAT_RGBA_16F
        4, 4, 4, 4, // FORMAT_RGBA_32F
        // special packed formats
        0, 0,
        // depth stencil formats
        0, 0, 0, 0, 0, 0
    };

    BOOST_STATIC_ASSERT((sizeof(channel_sizes) / sizeof(int)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d <= FORMAT_RGBA_32F);
    return (channel_sizes[d]);
}

inline
int size_of_format(data_format d)
{
    static int format_sizes[] = {
        1,
        // normalized integer formats (NORM)
        1, 2, 3, 4,   // FORMAT_RGBA_8
        2, 4, 6, 8,   // FORMAT_RGBA_16
        1, 2, 3, 4,   // FORMAT_RGBA_8S
        2, 4, 6, 8,   // FORMAT_RGBA_16S
        // swizzled integer formats
        3, 4,         // FORMAT_BGRA_8
        // srgb integer formats
        3, 4,         // FORMAT_SRGB_8
        // unnormalized integer formats (UNORM)
        1, 2, 3, 4,   // FORMAT_RGBA_8I
        2, 4, 6, 8,   // FORMAT_RGBA_16I
        4, 8, 12, 16, // FORMAT_RGBA_32I
        1, 2, 3, 4,   // FORMAT_RGBA_8UI
        2, 4, 6, 8,   // FORMAT_RGBA_16UI
        4, 8, 12, 16, // FORMAT_RGBA_32UI
        // floating point formats
        2, 4, 6, 8,   // FORMAT_RGBA_16F
        4, 8, 12, 16, // FORMAT_RGBA_32F
        // special packed formats
        4, 4,
        // depth stencil formats
        2, 3, 4, 4, 4, 8
    };

    BOOST_STATIC_ASSERT((sizeof(format_sizes) / sizeof(int)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return (format_sizes[d]);
}

inline
int
size_of_depth_component(data_format d)
{
    assert(FORMAT_D16 <= d && d <= FORMAT_D32F_S8);
    switch (d) {
        case FORMAT_D16:        return (2);
        case FORMAT_D24:        return (3);
        case FORMAT_D32:        return (4);
        case FORMAT_D32F:       return (4);
        case FORMAT_D24_S8:     return (3);
        case FORMAT_D32F_S8:    return (4);
        default:                return (0);
    }
}

inline
int size_of_stencil_component(data_format d)
{
    assert(FORMAT_D16 <= d && d <= FORMAT_D32F_S8);
    switch (d) {
        case FORMAT_D16:        return (0);
        case FORMAT_D24:        return (0);
        case FORMAT_D32:        return (0);
        case FORMAT_D32F:       return (0);
        case FORMAT_D24_S8:     return (1);
        case FORMAT_D32F_S8:    return (1);
        default:                return (0);
    }
}

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_DATA_FORMATS_INL_INCLUDED
