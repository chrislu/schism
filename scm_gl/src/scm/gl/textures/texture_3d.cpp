
#include "texture_3d.h"

#include <cassert>

#include <scm/gl/utilities/error_checker.h>

namespace scm {
namespace gl {

texture_3d::texture_3d()
    : texture(GL_TEXTURE_3D, GL_TEXTURE_BINDING_3D),
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
    assert(glGetError() == GL_NONE);

    gl::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureImage3DEXT(texture_id(),
                        texture_target(),
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
    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexImage3D(texture_target(),
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

    tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    tex_parameteri(GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    tex_parameteri(GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE);

    assert(glGetError() == GL_NONE);

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
    assert(glGetError() == GL_NONE);

    gl::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureSubImage3DEXT(texture_id(), texture_target(),
                           mip_level,
                           off_x, off_y, off_z,
                           width, height, depth,
                           format, type, data);
    assert(ech.ok());
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexSubImage3D(texture_target(),
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

    assert(glGetError() == GL_NONE);

    return (true);
}

} // namespace gl
} // namespace scm
