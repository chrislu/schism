
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "data_formats.h"

#include <cassert>

#include <boost/static_assert.hpp>

namespace  {

const char* format_strings[] = {
    "NULL",
    // normalized integer formats (NORM)
    "R_8", "RG_8", "RGB_8", "RGBA_8",
    "R_16", "RG_16", "RGB_16", "RGBA_16",
    "R_8S", "RG_8S", "RGB_8S", "RGBA_8S",
    "R_16S", "RG_16S", "RGB_16S", "RGBA_16S",
    // packed integer formats
    "BGR_8", "BGRA_8",
    // srgb integer formats
    "SRGB_8", "SRGBA_8",
    // unnormalized integer formats (UNORM)
    "R_8I", "RG_8I", "RGB_8I", "RGBA_8I",
    "R_16I", "RG_16I", "RGB_16I", "RGBA_16I",
    "R_32I", "RG_32I", "RGB_32I", "RGBA_32I",
    "R_8UI", "RG_8UI", "RGB_8UI", "RGBA_8UI",
    "R_16UI", "RG_16UI", "RGB_16UI", "RGBA_16UI",
    "R_32UI", "RG_32UI", "RGB_32UI", "RGBA_32UI",
    // floating point formats
    "R_16F", "RG_16F", "RGB_16F", "RGBA_16F",
    "R_32F", "RG_32F", "RGB_32F", "RGBA_32F",
    // special packed formats
    "RGB9_E5", "R11B11G10F",
    // compressed formats
    "BC1_RGBA", "BC1_SRGBA",
    "BC2_RGBA", "BC2_SRGBA",
    "BC3_RGBA", "BC3_SRGBA",
    "BC4_R", "BC4_R_S",
    "BC5_RG", "BC5_RG_S",
    "BC6H_RGB_F", "BC6H_RGB_UF",
    "BC7_RGBA", "BC7_SRGBA",
    // depth stencil formats
    "D16", "D24", "D32", "D32F", "D24_S8", "D32F_S8"
};

} // namespace 

namespace scm {
namespace gl {

const char* format_string(data_format d)
{
    BOOST_STATIC_ASSERT((sizeof(format_strings) / sizeof(const char*)) == FORMAT_COUNT);
    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);
    return format_strings[d];
}

} // namespace gl
} // namespace scm
