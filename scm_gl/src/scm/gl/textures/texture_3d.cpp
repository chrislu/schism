
#include "texture_3d.h"

#include <cassert>

#include <scm/gl/utilities/error_checker.h>

namespace scm {
namespace gl {

texture_3d::texture_3d()
    : texture(GL_TEXTURE_3D),
      _width(0),
      _height(0),
      _depth(0)
{
}

texture_3d::texture_3d(const texture_3d& tex)
    : texture(tex),
      _width(tex._width),
      _height(tex._height),
      _depth(tex._depth)
{
}

texture_3d::~texture_3d()
{
}

const texture_3d& texture_3d::operator=(const texture_3d& rhs)
{
    gl::texture::operator=(rhs);

    this->_width    = rhs._width;
    this->_height   = rhs._height;
    this->_depth    = rhs._depth;

    return (*this);
}

bool texture_3d::tex_image(GLint     mip_level,
                           GLint     internal_format,
                           GLsizei   width,
                           GLsizei   height,
                           GLsizei   depth,
                           GLenum    format,
                           GLenum    type,
                           const GLvoid *data)
{
    gl::error_checker ech;

    this->bind();

    glTexImage3D(get_texture_target(),
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

    _width  = width;
    _height = height;
    _depth  = depth;

    this->unbind();

    return (true);
}

bool texture_3d::tex_sub_image(GLint     mip_level,
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
    this->bind();

    glTexSubImage3D(get_texture_target(),
                    mip_level,
                    off_x, off_y, off_z,
                    width, height, depth,
                    format, type, data);

    if ((_last_error = glGetError()) != GL_NO_ERROR) {
        return (false);
    }

    this->unbind();

    return (true);
}

} // namespace gl
} // namespace scm
