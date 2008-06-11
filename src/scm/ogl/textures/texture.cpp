
#include "texture.h"

#include <cassert>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

texture::texture(const GLenum target)
  : _texture_id(new GLuint),
    _texture_target(target),
    _last_error(GL_NO_ERROR),
    _occupied_texture_unit(-1)
{
    *_texture_id = 0;

    glGenTextures(1, _texture_id.get());
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
    if (_texture_id.unique()) {
        glDeleteTextures(1, _texture_id.get());
    }
}

const texture& texture::operator=(const texture& rhs)
{
    if (_texture_id.unique()) {
        glDeleteTextures(1, _texture_id.get());
    }

    this->_texture_id       = rhs._texture_id;
    this->_texture_target   = rhs._texture_target;

    return (*this);
}

//void texture::delete_texture()
//{
//    if (*_texture_id != 0) {
//        glDeleteTextures(1, _texture_id.get());
//        _texture_id.ResetDC
//    }
//}

void texture::tex_parameteri(GLenum pname, GLint param)
{
    glTexParameteri(_texture_target, pname, param);
}

int texture::get_texture_target() const
{
    return (_texture_target);
}

unsigned texture::get_texture_id() const
{
    checked_lazy_generate_texture_id();

    return (*_texture_id);
}

int texture::get_last_error() const
{
    return (_last_error);
}

void texture::bind(int texunit) const
{
    assert(_texture_id);

    if (texunit >= 0) {
        _occupied_texture_unit = texunit;
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        assert(glGetError() == GL_NONE);
    }
    glEnable(_texture_target);
    assert(glGetError() == GL_NONE);

    glBindTexture(_texture_target, get_texture_id());
    assert(glGetError() == GL_NONE);
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

void texture::checked_lazy_generate_texture_id() const
{
    if (*_texture_id == 0) {
        glGenTextures(1, _texture_id.get());
    }
}

} // namespace gl
} // namespace scm
