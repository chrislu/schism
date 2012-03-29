
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_3d.h"

#include <cassert>

#include <scm/gl_classic/utilities/error_checker.h>
#include <scm/gl_classic/utilities/gl_assert.h>

namespace scm {
namespace gl_classic {

texture_3d::texture_3d()
    : texture(GL_TEXTURE_3D, GL_TEXTURE_BINDING_3D),
      _width(0),
      _height(0),
      _depth(0)
{
}

texture_3d::~texture_3d()
{
}

bool texture_3d::image_data(GLint     mip_level,
                            GLint     internal_format,
                            GLsizei   width,
                            GLsizei   height,
                            GLsizei   depth,
                            GLenum    format,
                            GLenum    type,
                            const GLvoid *data)
{
    gl_assert_error("entering texture_3d::image_data()");

    gl_classic::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureImage3DEXT(id(),
                        target(),
                        mip_level,
                        internal_format,
                        width,
                        height,
                        depth,
                        0,
                        format,
                        type,
                        data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexImage3D(target(),
                 mip_level,
                 internal_format,
                 width,
                 height,
                 depth,
                 0,
                 format,
                 type,
                 data);
    assert(ech.ok());
    //if (ech.check_error()) {
    //    std::cout << ech.get_error_string() << std::endl;
    //if ((_last_error = glGetError()) != GL_NO_ERROR) {
    //    return (false);
    //}
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS

    _width  = width;
    _height = height;
    _depth  = depth;

    parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    parameter(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    parameter(GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    parameter(GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE);

    gl_assert_error("exiting texture_3d::image_data()");

    return (true);
}

bool texture_3d::image_sub_data(GLint     mip_level,
                                GLint     off_x,
                                GLint     off_y,
                                GLint     off_z,
                                GLsizei   width,
                                GLsizei   height,
                                GLsizei   depth,
                                GLenum    format,
                                GLenum    type,
                                const GLvoid *data)
{
    gl_assert_error("entering texture_3d::image_sub_data()");

    gl_classic::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureSubImage3DEXT(id(), target(),
                           mip_level,
                           off_x, off_y, off_z,
                           width, height, depth,
                           format, type, data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexSubImage3D(target(),
                    mip_level,
                    off_x, off_y, off_z,
                    width, height, depth,
                    format, type, data);
    assert(ech.ok());
    //if ((_last_error = glGetError()) != GL_NO_ERROR) {
    //    return (false);
    //}
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS

    gl_assert_error("exiting texture_3d::image_sub_data()");

    return (true);
}

} // namespace gl_classic
} // namespace scm
