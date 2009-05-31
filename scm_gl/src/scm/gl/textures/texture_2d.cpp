
#include "texture_2d.h"

#include <cassert>

#include <scm/gl/utilities/error_checker.h>

namespace scm {
namespace gl {

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

const texture_2d& texture_2d::operator=(const texture_2d& rhs)
{
    gl::texture::operator=(rhs);

    this->_width    = rhs._width;
    this->_height   = rhs._height;

    return (*this);
}

bool texture_2d::tex_image(GLint     mip_level,
                           GLint     internal_format,
                           GLsizei   width,
                           GLsizei   height,
                           GLenum    format,
                           GLenum    type,
                           const GLvoid *data)
{
    assert(glGetError() == GL_NONE);

    gl::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureImage2DEXT(texture_id(),
                        texture_target(),
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
    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexImage2D(texture_target(),
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

    _width  = width;
    _height = height;

    tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    tex_parameteri(GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    assert(glGetError() == GL_NONE);

    return (true);
}

bool texture_2d::tex_sub_image(GLint     mip_level,
                               GLint     off_x,
                               GLint     off_y,
                               GLsizei   width,
                               GLsizei   height,
                               GLenum    format,
                               GLenum    type,
                               const GLvoid *data)
{
    assert(glGetError() == GL_NONE);

    gl::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureSubImage2DEXT(texture_id(), texture_target(),
                           mip_level,
                           off_x, off_y,
                           width, height,
                           format, type, data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexSubImage2D(texture_target(),
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

    assert(glGetError() == GL_NONE);

    return (true);
}

} // namespace gl
} // namespace scm
