
#include "texture_2d.h"

#include <cassert>

#include <scm/ogl/utilities/error_checker.h>

namespace scm {
namespace gl {

texture_2d::texture_2d()
    : texture(GL_TEXTURE_2D),
      _width(0),
      _height(0)
{
}

texture_2d::texture_2d(const GLenum target)
    : texture(target),
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
    gl::error_checker ech;

    this->bind();

    glTexImage2D(get_texture_target(),
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

    _width  = width;
    _height = height;

    tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    tex_parameteri(GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    this->unbind();

    return (true);
}

} // namespace gl
} // namespace scm
