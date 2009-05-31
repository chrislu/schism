
#include "texture_1d.h"

#include <cassert>

#include <scm/gl/utilities/error_checker.h>

namespace scm {
namespace gl {

texture_1d::texture_1d()
    : texture(GL_TEXTURE_1D, GL_TEXTURE_BINDING_1D),
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
    assert(glGetError() == GL_NONE);

    gl::error_checker ech;
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureImage1DEXT(texture_id(),
                        texture_target(),
                        mip_level,
                        internal_format,
                        width,
                        0,
                        format,
                        type,
                        data);
    assert(ech.ok());

#else // SCM_GL_USE_DIRECT_STATE_ACCESS

    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexImage1D(texture_target(),
                 mip_level,
                 internal_format,
                 width,
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

    _width = width;

    tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex_parameteri(GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);

    assert(glGetError() == GL_NONE);

    return (true);
}

} // namespace gl
} // namespace scm
