
#ifndef SCM_GL_FORMAT_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_FORMAT_H_INCLUDED_DEVICE_H_INCLUDED

namespace scm {
namespace gl {

enum data_format {
    // null format
    FORMAT_NULL                 = 0x00u,

    // floating point formats
    FORMAT_SCAL_16F,
    FORMAT_VEC2_16F,
    FORMAT_VEC3_16F,
    FORMAT_VEC4_16F,

    FORMAT_SCAL_32F,
    FORMAT_VEC2_32F,
    FORMAT_VEC3_32F,
    FORMAT_VEC4_32F,

    // normalized integer formats
    FORMAT_SCAL_8I_NORM,
    FORMAT_VEC2_8I_NORM,
    FORMAT_VEC3_8I_NORM,
    FORMAT_VEC4_8I_NORM,

    // color formats
    FORMAT_RGB8,
    FORMAT_RGBA8,
    FORMAT_R16F,
    FORMAT_RG16F,
    FORMAT_RGB16F,
    FORMAT_RGBA16F,
    FORMAT_R32F,
    FORMAT_RG32F,
    FORMAT_RGB32F,
    FORMAT_RGBA32F,

    // depth stencil formats
    FORMAT_D24,
    FORMAT_D24_S8,
    FORMAT_D32F

}; // enum data_format

} // namespace gl
} // namespace scm

#endif // SCM_GL_FORMAT_H_INCLUDED_DEVICE_H_INCLUDED
