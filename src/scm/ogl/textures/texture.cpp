
#include "texture.h"

namespace scm {
namespace gl {

texture::texture(const GLenum target)
    : _texture_id(0),
      _texture_target(target),
      _last_error(GL_NO_ERROR),
      _occupied_texture_unit(-1)
{
}

texture::texture(const texture& tex)
    : _texture_target(tex._texture_target),
      _texture_id(tex._texture_id),
      _last_error(GL_NO_ERROR),
      _occupied_texture_unit(-1)
{
}

texture::~texture()
{
    delete_texture();
}

const texture& texture::operator=(const texture& rhs)
{
    this->_texture_id       = rhs._texture_id;
    this->_texture_target   = rhs._texture_target;

    return (*this);
}

void texture::delete_texture()
{
    if (_texture_id != 0) {
        glDeleteTextures(1, &_texture_id);
    }
}

void texture::tex_parameteri(GLenum pname, GLint param)
{
    glTexParameteri(_texture_target, pname, param);
}

void texture::bind(int texunit) const
{
    if (texunit >= 0) {
        _occupied_texture_unit = texunit;
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
    }
    glEnable(_texture_target);
    glBindTexture(_texture_target, _texture_id);
}

void texture::unbind() const
{
    if (_occupied_texture_unit >= 0) {
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        _occupied_texture_unit = -1;
    }
    glBindTexture(_texture_target, 0);
    glDisable(_texture_target);
}

void texture::generate_texture_id()
{
    glGenTextures(1, &_texture_id);
}

} // namespace gl
} // namespace scm
