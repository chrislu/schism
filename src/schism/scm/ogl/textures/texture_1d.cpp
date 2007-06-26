
#include "texture_1d.h"

namespace scm {
namespace gl {

texture_1d::texture_1d()
    : texture(GL_TEXTURE_1D),
      _width(0)
{
}

texture_1d::~texture_1d()
{
}

const texture_1d& texture_1d::operator=(const texture_1d& rhs)
{
    gl::texture::operator=(rhs);

    this->_width    = rhs._width;

    return (*this);
}

bool texture_1d::tex_image(GLint     mip_level,
                           GLint     internal_format,
                           GLsizei   width,
                           GLenum    format,
                           GLenum    type,
                           const GLvoid *data)
{
    if (get_texture_id() == 0) {
        generate_texture_id();
    }

    if (get_texture_id() == 0) {
        return (false);
    }

    this->bind();

    glTexImage1D(get_texture_target(),
                 mip_level,
                 internal_format,
                 width,
                 0,
                 format,
                 type,
                 data);

    if ((_last_error = glGetError()) != GL_NO_ERROR) {
        return (false);
    }

    _width = width;

    this->unbind();

    return (true);
}

} // namespace gl
} // namespace scm
