
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <cassert>

#include <boost/static_assert.hpp>

#include <scm/gl_core/render_device/opengl/gl_core.h>

namespace scm {
namespace gl {
namespace util {

inline
unsigned
gl_internal_format(const data_format d)
{
    static unsigned gl_int_fmts[] = {
        GL_NONE,        // FORMAT_NULL                 = 0x00u,

        // normalized integer formats (NORM)
        GL_R8,          // FORMAT_R_8,
        GL_RG8,         // FORMAT_RG_8,
        GL_RGB8,        // FORMAT_RGB_8,
        GL_RGBA8,       // FORMAT_RGBA_8,

        GL_R16,         // FORMAT_R_16,
        GL_RG16,        // FORMAT_RG_16,
        GL_RGB16,       // FORMAT_RGB_16,
        GL_RGBA16,      // FORMAT_RGBA_16,

        GL_R8_SNORM,    // FORMAT_R_8S,
        GL_RG8_SNORM,   // FORMAT_RG_8S,
        GL_RGB8_SNORM,  // FORMAT_RGB_8S,
        GL_RGBA8_SNORM, // FORMAT_RGBA_8S,

        GL_R16_SNORM,   // FORMAT_R_16S,
        GL_RG16_SNORM,  // FORMAT_RG_16S,
        GL_RGB16_SNORM, // FORMAT_RGB_16S,
        GL_RGBA16_SNORM,// FORMAT_RGBA_16S,

        // swizzled integer formats
        GL_RGB8,        // FORMAT_BGR_8,
        GL_RGBA8,       // FORMAT_BGRA_8,

        // srgb integer formats
        GL_SRGB8,       // FORMAT_SRGB_8,
        GL_SRGB8_ALPHA8,// FORMAT_SRGBA_8,

        // unnormalized integer formats (UNORM)
        GL_R8I,         // FORMAT_R_8I,
        GL_RG8I,        // FORMAT_RG_8I,
        GL_RGB8I,       // FORMAT_RGB_8I,
        GL_RGBA8I,      // FORMAT_RGBA_8I,

        GL_R16I,        // FORMAT_R_16I,
        GL_RG16I,       // FORMAT_RG_16I,
        GL_RGB16I,      // FORMAT_RGB_16I,
        GL_RGBA16I,     // FORMAT_RGBA_16I,

        GL_R32I,        // FORMAT_R_32I,
        GL_RG32I,       // FORMAT_RG_32I,
        GL_RGB32I,      // FORMAT_RGB_32I,
        GL_RGBA32I,     // FORMAT_RGBA_32I,

        GL_R8UI,        // FORMAT_R_8UI,
        GL_RG8UI,       // FORMAT_RG_8UI,
        GL_RGB8UI,      // FORMAT_RGB_8UI,
        GL_RGBA8UI,     // FORMAT_RGBA_8UI,

        GL_R16UI,       // FORMAT_R_16UI,
        GL_RG16UI,      // FORMAT_RG_16UI,
        GL_RGB16UI,     // FORMAT_RGB_16UI,
        GL_RGBA16UI,    // FORMAT_RGBA_16UI,

        GL_R32UI,       // FORMAT_R_32UI,
        GL_RG32UI,      // FORMAT_RG_32UI,
        GL_RGB32UI,     // FORMAT_RGB_32UI,
        GL_RGBA32UI,    // FORMAT_RGBA_32UI,

        // floating point formats
        GL_R16F,        // FORMAT_R_16F,
        GL_RG16F,       // FORMAT_RG_16F,
        GL_RGB16F,      // FORMAT_RGB_16F,
        GL_RGBA16F,     // FORMAT_RGBA_16F,

        GL_R32F,        // FORMAT_R_32F,
        GL_RG32F,       // FORMAT_RG_32F,
        GL_RGB32F,      // FORMAT_RGB_32F,
        GL_RGBA32F,     // FORMAT_RGBA_32F,

        // special packed formats
        GL_RGB9_E5,     // FORMAT_RGB9_E5,
        GL_R11F_G11F_B10F,// FORMAT_R11B11G10F,

        // compressed formats
        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,           //FORMAT_BC1_RGBA,        // DXT1
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,     //FORMAT_BC1_SRGBA,       // DXT1
        GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,           //FORMAT_BC2_RGBA,        // DXT3
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,     //FORMAT_BC2_SRGBA,       // DXT3
        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,           //FORMAT_BC3_RGBA,        // DXT5
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,     //FORMAT_BC3_SRGBA,       // DXT5
        GL_COMPRESSED_RED_RGTC1,                    //FORMAT_BC4_R,           // RGTC1
        GL_COMPRESSED_SIGNED_RED_RGTC1,             //FORMAT_BC4_R_S,         // RGTC1
        GL_COMPRESSED_RG_RGTC2,                     //FORMAT_BC5_RG,          // RGTC2
        GL_COMPRESSED_SIGNED_RG_RGTC2,              //FORMAT_BC5_RG_S,        // RGTC2
        GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB,    //FORMAT_BC6H_RGB_F,      // BPTC
        GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB,  //FORMAT_BC6H_RGB_UF,     // BPTC
        GL_COMPRESSED_RGBA_BPTC_UNORM_ARB,          //FORMAT_BC7_RGBA,        // BPTC
        GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB,    //FORMAT_BC8_SRGBA,       // BPTC

        // depth stencil formats
        GL_DEPTH_COMPONENT16, // FORMAT_D16,
        GL_DEPTH_COMPONENT24, // FORMAT_D24,
        GL_DEPTH_COMPONENT32, // FORMAT_D32,
        GL_DEPTH_COMPONENT32F, // FORMAT_D32F,
        GL_DEPTH24_STENCIL8, // FORMAT_D24_S8,
        GL_DEPTH32F_STENCIL8 // FORMAT_D32F_S8,
    };

    BOOST_STATIC_ASSERT((sizeof(gl_int_fmts) / sizeof(unsigned)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);

    return gl_int_fmts[d];
}

inline
unsigned
gl_base_format(const data_format d)
{
    static unsigned gl_int_bfmts[] = {
        GL_NONE,        // FORMAT_NULL                 = 0x00u,

        // normalized integer formats (NORM)
        GL_RED,         // FORMAT_R_8,
        GL_RG,          // FORMAT_RG_8,
        GL_RGB,         // FORMAT_RGB_8,
        GL_RGBA,        // FORMAT_RGBA_8,

        GL_RED,         // FORMAT_R_16,
        GL_RG,          // FORMAT_RG_16,
        GL_RGB,         // FORMAT_RGB_16,
        GL_RGBA,        // FORMAT_RGBA_16,

        GL_RED,         // FORMAT_R_8S,
        GL_RG,          // FORMAT_RG_8S,
        GL_RGB,         // FORMAT_RGB_8S,
        GL_RGBA,        // FORMAT_RGBA_8S,

        GL_RED,         // FORMAT_R_16S,
        GL_RG,          // FORMAT_RG_16S,
        GL_RGB,         // FORMAT_RGB_16S,
        GL_RGBA,        // FORMAT_RGBA_16S,

        // swizzled integer formats
        GL_BGR,         // FORMAT_BGR_8,
        GL_BGRA,        // FORMAT_BGRA_8,

        // srgb integer formats
        GL_RGB,         // FORMAT_SRGB_8,
        GL_RGBA,        // FORMAT_SRGBA_8,

        // unnormalized integer formats (UNORM)
        GL_RED_INTEGER, // FORMAT_R_8I,
        GL_RG_INTEGER,  // FORMAT_RG_8I,
        GL_RGB_INTEGER, // FORMAT_RGB_8I,
        GL_RGBA_INTEGER,// FORMAT_RGBA_8I,

        GL_RED_INTEGER, // FORMAT_R_16I,
        GL_RG_INTEGER,  // FORMAT_RG_16I,
        GL_RGB_INTEGER, // FORMAT_RGB_16I,
        GL_RGBA_INTEGER,// FORMAT_RGBA_16I,

        GL_RED_INTEGER, // FORMAT_R_32I,
        GL_RG_INTEGER,  // FORMAT_RG_32I,
        GL_RGB_INTEGER, // FORMAT_RGB_32I,
        GL_RGBA_INTEGER,// FORMAT_RGBA_32I,

        GL_RED_INTEGER, // FORMAT_R_8UI,
        GL_RG_INTEGER,  // FORMAT_RG_8UI,
        GL_RGB_INTEGER, // FORMAT_RGB_8UI,
        GL_RGBA_INTEGER,// FORMAT_RGBA_8UI,

        GL_RED_INTEGER, // FORMAT_R_16UI,
        GL_RG_INTEGER,  // FORMAT_RG_16UI,
        GL_RGB_INTEGER, // FORMAT_RGB_16UI,
        GL_RGBA_INTEGER,// FORMAT_RGBA_16UI,

        GL_RED_INTEGER, // FORMAT_R_32UI,
        GL_RG_INTEGER,  // FORMAT_RG_32UI,
        GL_RGB_INTEGER, // FORMAT_RGB_32UI,
        GL_RGBA_INTEGER,// FORMAT_RGBA_32UI,

        // floating point formats
        GL_RED,         // FORMAT_R_16F,
        GL_RG,          // FORMAT_RG_16F,
        GL_RGB,         // FORMAT_RGB_16F,
        GL_RGBA,        // FORMAT_RGBA_16F,

        GL_RED,         // FORMAT_R_32F,
        GL_RG,          // FORMAT_RG_32F,
        GL_RGB,         // FORMAT_RGB_32F,
        GL_RGBA,        // FORMAT_RGBA_32F,

        // special packed formats
        GL_RGB,         // FORMAT_RGB9_E5,
        GL_RGB,         // FORMAT_R11B11G10F,

        // compressed formats
        GL_RGBA,        //FORMAT_BC1_RGBA,        // DXT1
        GL_RGBA,        //FORMAT_BC1_SRGBA,       // DXT1
        GL_RGBA,        //FORMAT_BC2_RGBA,        // DXT3
        GL_RGBA,        //FORMAT_BC2_SRGBA,       // DXT3
        GL_RGBA,        //FORMAT_BC3_RGBA,        // DXT5
        GL_RGBA,        //FORMAT_BC3_SRGBA,       // DXT5
        GL_RED,         //FORMAT_BC4_R,           // RGTC1
        GL_RED,         //FORMAT_BC4_R_S,         // RGTC1
        GL_RG,          //FORMAT_BC5_RG,          // RGTC2
        GL_RG,          //FORMAT_BC5_RG_S,        // RGTC2
        GL_RGB,         //FORMAT_BC6H_RGB_F,      // BPTC
        GL_RGB,         //FORMAT_BC6H_RGB_UF,     // BPTC
        GL_RGBA,        //FORMAT_BC7_RGBA,        // BPTC
        GL_RGBA,        //FORMAT_BC7_SRGBA,       // BPTC

        // depth stencil formats
        GL_DEPTH_COMPONENT, // FORMAT_D16,
        GL_DEPTH_COMPONENT, // FORMAT_D24,
        GL_DEPTH_COMPONENT, // FORMAT_D32,
        GL_DEPTH_COMPONENT, // FORMAT_D32F,
        GL_DEPTH_STENCIL, // FORMAT_D24_S8,
        GL_DEPTH_STENCIL // FORMAT_D32F_S8,
    };

    BOOST_STATIC_ASSERT((sizeof(gl_int_bfmts) / sizeof(unsigned)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);

    return gl_int_bfmts[d];
}

inline
unsigned
gl_base_type(const data_format d)
{
    static unsigned gl_btypes[] = {
        GL_NONE,        // FORMAT_NULL                 = 0x00u,

        // normalized integer formats (NORM)
        GL_UNSIGNED_BYTE,          // FORMAT_R_8,
        GL_UNSIGNED_BYTE,         // FORMAT_RG_8,
        GL_UNSIGNED_BYTE,        // FORMAT_RGB_8,
        GL_UNSIGNED_BYTE,       // FORMAT_RGBA_8,

        GL_UNSIGNED_SHORT,         // FORMAT_R_16,
        GL_UNSIGNED_SHORT,        // FORMAT_RG_16,
        GL_UNSIGNED_SHORT,       // FORMAT_RGB_16,
        GL_UNSIGNED_SHORT,      // FORMAT_RGBA_16,

        GL_BYTE,    // FORMAT_R_8S,
        GL_BYTE,   // FORMAT_RG_8S,
        GL_BYTE,  // FORMAT_RGB_8S,
        GL_BYTE, // FORMAT_RGBA_8S,

        GL_SHORT,   // FORMAT_R_16S,
        GL_SHORT,  // FORMAT_RG_16S,
        GL_SHORT, // FORMAT_RGB_16S,
        GL_SHORT,// FORMAT_RGBA_16S,

        // swizzled integer formats
        GL_UNSIGNED_BYTE,        // FORMAT_BGR_8,
        GL_UNSIGNED_INT_8_8_8_8_REV,       // FORMAT_BGRA_8,

        // srgb integer formats
        GL_UNSIGNED_BYTE,       // FORMAT_SRGB_8,
        GL_UNSIGNED_BYTE,// FORMAT_SRGBA_8,

        // unnormalized integer formats (UNORM)
        GL_BYTE,         // FORMAT_R_8I,
        GL_BYTE,        // FORMAT_RG_8I,
        GL_BYTE,       // FORMAT_RGB_8I,
        GL_BYTE,      // FORMAT_RGBA_8I,

        GL_SHORT,        // FORMAT_R_16I,
        GL_SHORT,       // FORMAT_RG_16I,
        GL_SHORT,      // FORMAT_RGB_16I,
        GL_SHORT,     // FORMAT_RGBA_16I,

        GL_INT,        // FORMAT_R_32I,
        GL_INT,       // FORMAT_RG_32I,
        GL_INT,      // FORMAT_RGB_32I,
        GL_INT,     // FORMAT_RGBA_32I,

        GL_UNSIGNED_BYTE,        // FORMAT_R_8UI,
        GL_UNSIGNED_BYTE,       // FORMAT_RG_8UI,
        GL_UNSIGNED_BYTE,      // FORMAT_RGB_8UI,
        GL_UNSIGNED_BYTE,     // FORMAT_RGBA_8UI,

        GL_UNSIGNED_SHORT,       // FORMAT_R_16UI,
        GL_UNSIGNED_SHORT,      // FORMAT_RG_16UI,
        GL_UNSIGNED_SHORT,     // FORMAT_RGB_16UI,
        GL_UNSIGNED_SHORT,    // FORMAT_RGBA_16UI,

        GL_UNSIGNED_INT,       // FORMAT_R_32UI,
        GL_UNSIGNED_INT,      // FORMAT_RG_32UI,
        GL_UNSIGNED_INT,     // FORMAT_RGB_32UI,
        GL_UNSIGNED_INT,    // FORMAT_RGBA_32UI,

        // floating point formats
        GL_HALF_FLOAT,        // FORMAT_R_16F,
        GL_HALF_FLOAT,       // FORMAT_RG_16F,
        GL_HALF_FLOAT,      // FORMAT_RGB_16F,
        GL_HALF_FLOAT,     // FORMAT_RGBA_16F,

        GL_FLOAT,        // FORMAT_R_32F,
        GL_FLOAT,       // FORMAT_RG_32F,
        GL_FLOAT,      // FORMAT_RGB_32F,
        GL_FLOAT,     // FORMAT_RGBA_32F,

        // special packed formats
        GL_FLOAT,     // FORMAT_RGB9_E5,
        GL_FLOAT,// FORMAT_R11B11G10F,

        // compressed formats
        GL_UNSIGNED_BYTE,   //FORMAT_BC1_RGBA,        // DXT1
        GL_UNSIGNED_BYTE,   //FORMAT_BC1_SRGBA,       // DXT1
        GL_UNSIGNED_BYTE,   //FORMAT_BC2_RGBA,        // DXT3
        GL_UNSIGNED_BYTE,   //FORMAT_BC2_SRGBA,       // DXT3
        GL_UNSIGNED_BYTE,   //FORMAT_BC3_RGBA,        // DXT5
        GL_UNSIGNED_BYTE,   //FORMAT_BC3_SRGBA,       // DXT5
        GL_UNSIGNED_BYTE,   //FORMAT_BC4_R,           // RGTC1
        GL_UNSIGNED_BYTE,   //FORMAT_BC4_R_S,         // RGTC1
        GL_UNSIGNED_BYTE,   //FORMAT_BC5_RG,          // RGTC2
        GL_UNSIGNED_BYTE,   //FORMAT_BC5_RG_S,        // RGTC2
        GL_FLOAT,           // FORMAT_BC6H_RGB_F,      // BPTC
        GL_FLOAT,           // FORMAT_BC6H_RGB_UF,     // BPTC
        GL_UNSIGNED_BYTE,   //FORMAT_BC7_RGBA,        // BPTC
        GL_UNSIGNED_BYTE,   //FORMAT_BC7_SRGBA,       // BPTC

        // depth stencil formats
        GL_UNSIGNED_SHORT, // FORMAT_D16,
        GL_UNSIGNED_INT, // FORMAT_D24,
        GL_UNSIGNED_INT, // FORMAT_D32,
        GL_FLOAT, // FORMAT_D32F,
        GL_UNSIGNED_INT_24_8, // FORMAT_D24_S8,
        GL_FLOAT_32_UNSIGNED_INT_24_8_REV// FORMAT_D32F_S8,
    };

    BOOST_STATIC_ASSERT((sizeof(gl_btypes) / sizeof(unsigned)) == FORMAT_COUNT);

    assert(FORMAT_NULL <= d && d < FORMAT_COUNT);

    return gl_btypes[d];
}

} // namespace util
} // namespace gl
} // namespace scm
