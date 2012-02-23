
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DATA_FORMATS_H_INCLUDED
#define SCM_GL_CORE_DATA_FORMATS_H_INCLUDED

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

enum data_format {
    // null format
    FORMAT_NULL                 = 0x00u,

    // normalized integer formats (NORM)
    FORMAT_R_8,
    FORMAT_RG_8,
    FORMAT_RGB_8,
    FORMAT_RGBA_8,

    FORMAT_R_16,
    FORMAT_RG_16,
    FORMAT_RGB_16,
    FORMAT_RGBA_16,

    FORMAT_R_8S,
    FORMAT_RG_8S,
    FORMAT_RGB_8S,
    FORMAT_RGBA_8S,

    FORMAT_R_16S,
    FORMAT_RG_16S,
    FORMAT_RGB_16S,
    FORMAT_RGBA_16S,

    // swizzled integer formats
    FORMAT_BGR_8,
    FORMAT_BGRA_8,

    // srgb integer formats
    FORMAT_SRGB_8,
    FORMAT_SRGBA_8,

    // unnormalized integer formats
    FORMAT_R_8I,
    FORMAT_RG_8I,
    FORMAT_RGB_8I,
    FORMAT_RGBA_8I,

    FORMAT_R_16I,
    FORMAT_RG_16I,
    FORMAT_RGB_16I,
    FORMAT_RGBA_16I,

    FORMAT_R_32I,
    FORMAT_RG_32I,
    FORMAT_RGB_32I,
    FORMAT_RGBA_32I,

    FORMAT_R_8UI,
    FORMAT_RG_8UI,
    FORMAT_RGB_8UI,
    FORMAT_RGBA_8UI,

    FORMAT_R_16UI,
    FORMAT_RG_16UI,
    FORMAT_RGB_16UI,
    FORMAT_RGBA_16UI,

    FORMAT_R_32UI,
    FORMAT_RG_32UI,
    FORMAT_RGB_32UI,
    FORMAT_RGBA_32UI,

    // floating point formats
    FORMAT_R_16F,
    FORMAT_RG_16F,
    FORMAT_RGB_16F,
    FORMAT_RGBA_16F,

    FORMAT_R_32F,
    FORMAT_RG_32F,
    FORMAT_RGB_32F,
    FORMAT_RGBA_32F,

    // special packed formats
    FORMAT_RGB9_E5,
    FORMAT_R11B11G10F,

    // compressed formats
    FORMAT_BC1_RGBA,        // DXT1
    FORMAT_BC1_SRGBA,       // DXT1
    FORMAT_BC2_RGBA,        // DXT3
    FORMAT_BC2_SRGBA,       // DXT3
    FORMAT_BC3_RGBA,        // DXT5
    FORMAT_BC3_SRGBA,       // DXT5
    FORMAT_BC4_R,           // RGTC1
    FORMAT_BC4_R_S,         // RGTC1
    FORMAT_BC5_RG,          // RGTC2
    FORMAT_BC5_RG_S,        // RGTC2
    FORMAT_BC6H_RGB_F,      // BPTC
    FORMAT_BC6H_RGB_UF,     // BPTC
    FORMAT_BC7_RGBA,        // BPTC
    FORMAT_BC7_SRGBA,       // BPTC

    // depth stencil formats
    FORMAT_D16,
    FORMAT_D24,
    FORMAT_D32,
    FORMAT_D32F,
    FORMAT_D24_S8,
    FORMAT_D32F_S8,

    FORMAT_COUNT
}; // enum data_format

bool is_normalized(data_format d);
bool is_unnormalized(data_format d);

bool is_integer_type(data_format d);
bool is_float_type(data_format d);

bool is_depth_format(data_format d);
bool is_stencil_format(data_format d);

bool is_packed_format(data_format d);
bool is_plain_format(data_format d);

bool is_srgb_format(data_format d);

bool is_compressed_format(data_format d);

int channel_count(data_format d);
int size_of_channel(data_format d);
int size_of_format(data_format d);
int bit_per_pixel(data_format d);
int compressed_block_size(data_format d);

int size_of_depth_component(data_format d);
int size_of_stencil_component(data_format d);

__scm_export(gl_core) const char* format_string(data_format d);

} // namespace gl
} // namespace scm

#include "data_formats.inl"

#endif // SCM_GL_CORE_DATA_FORMATS_H_INCLUDED
