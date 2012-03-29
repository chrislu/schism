
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_2d.h"

#include <cassert>

#include <scm/gl_classic/utilities/error_checker.h>
#include <scm/gl_classic/utilities/gl_assert.h>

namespace scm {
namespace gl_classic {

texture_2d::texture_2d()
    : texture(GL_TEXTURE_2D, GL_TEXTURE_BINDING_2D),
      _width(0),
      _height(0)
{
}

texture_2d::texture_2d(const GLenum target, const GLenum binding)
    : texture(target, binding),
      _width(0),
      _height(0)
{
}

texture_2d::~texture_2d()
{
}

bool texture_2d::image_data(GLint     mip_level,
                            GLint     internal_format,
                            GLsizei   width,
                            GLsizei   height,
                            GLenum    format,
                            GLenum    type,
                            const GLvoid *data)
{
    gl_assert_error("entering texture_2d::image_data()");

    gl_classic::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureImage2DEXT(id(),
                        target(),
                        mip_level,
                        internal_format,
                        width,
                        height,
                        0,
                        format,
                        type,
                        data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexImage2D(target(),
                 mip_level,
                 internal_format,
                 width,
                 height,
                 0,
                 format,
                 type,
                 data);
    assert(ech.ok());
    //if ((_last_error = glGetError()) != GL_NO_ERROR) {
    //    return (false);
    //}
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS

    if (mip_level == 0) {
        _width  = width;
        _height = height;
    }

    parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    parameter(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    parameter(GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    gl_assert_error("exiting texture_2d::image_data()");

    return (true);
}

bool texture_2d::image_sub_data(GLint     mip_level,
                                GLint     off_x,
                                GLint     off_y,
                                GLsizei   width,
                                GLsizei   height,
                                GLenum    format,
                                GLenum    type,
                                const GLvoid *data)
{
    gl_assert_error("entering texture_2d::image_sub_data()");

    gl_classic::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureSubImage2DEXT(id(), target(),
                           mip_level,
                           off_x, off_y,
                           width, height,
                           format, type, data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexSubImage2D(target(),
                    mip_level,
                    off_x, off_y,
                    width, height,
                    format, type, data);
    assert(ech.ok());
    //if ((_last_error = glGetError()) != GL_NO_ERROR) {
    //    return (false);
    //}
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS

    gl_assert_error("exiting texture_2d::image_sub_data()");

    return (true);
}

} // namespace gl_classic
} // namespace scm
