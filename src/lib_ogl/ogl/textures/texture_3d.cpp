
#include "texture_3d.h"

#include <ogl/utilities/error_checker.h>
#include <iostream>

namespace gl
{
    texture_3d::texture_3d()
        : texture(GL_TEXTURE_3D),
          _width(0),
          _height(0),
          _depth(0)
    {
    }
   
    texture_3d::texture_3d(const gl::texture_3d& tex)
        : texture(tex),
          _width(tex._width),
          _height(tex._height),
          _depth(tex._depth)
    {
    }

    texture_3d::~texture_3d()
    {
    }

    const gl::texture_3d& texture_3d::operator=(const gl::texture_3d& rhs)
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
        if (get_texture_id() == 0) {
            generate_texture_id();
        }

        if (get_texture_id() == 0) {
            return (false);
        }

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

        //gl::error_checker ech;

        //if (ech.check_error()) {
        //    std::cout << ech.get_error_string() << std::endl;
        if ((_last_error = glGetError()) != GL_NO_ERROR) {
            return (false);
        }

        _width  = width;
        _height = height;
        _depth  = depth;

        this->unbind();

        return (true);
    }


} // namespace gl



