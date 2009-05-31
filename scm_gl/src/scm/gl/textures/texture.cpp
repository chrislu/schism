
#include "texture.h"

#include <cassert>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

texture::binding_guard::binding_guard(unsigned target, unsigned binding)
  : _save_active_texture_unit(0),
    _save_texture_object(0),
    _binding(binding),
    _target(target)
{
    assert(glGetError() == GL_NONE);

    glGetIntegerv(GL_ACTIVE_TEXTURE, &_save_active_texture_unit);
    glGetIntegerv(_binding, &_save_texture_object);

    assert(glGetError() == GL_NONE);
}

texture::binding_guard::~binding_guard()
{
    assert(glGetError() == GL_NONE);

    glActiveTexture(_save_active_texture_unit);
    glBindTexture(_target, _save_texture_object);

    assert(glGetError() == GL_NONE);
}

texture::texture(const GLenum target, const GLenum binding)
  : _texture_id(new GLuint),
    _texture_target(target),
    _texture_binding(binding),
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
    assert(glGetError() == GL_NONE);
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureParameteriEXT(texture_id(), _texture_target, pname, param);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(texture_target(), texture_binding());
    bind();
    glTexParameteri(_texture_target, pname, param);
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    assert(glGetError() == GL_NONE);

}

int texture::texture_target() const
{
    return (_texture_target);
}

int texture::texture_binding() const
{
    return (_texture_binding);
}

unsigned texture::texture_id() const
{
    checked_lazy_generate_texture_id();

    return (*_texture_id);
}

int texture::last_error() const
{
    return (_last_error);
}

void texture::bind(int texunit) const
{
    assert(glGetError() == GL_NONE);
    assert(_texture_id);
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    _occupied_texture_unit = texunit > 0 ? texunit : 0;
    glBindMultiTextureEXT(GL_TEXTURE0 + _occupied_texture_unit, _texture_target, texture_id());
    glEnableIndexedEXT(_texture_target, _occupied_texture_unit);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    if (texunit >= 0) {
        _occupied_texture_unit = texunit;
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        assert(glGetError() == GL_NONE);
    }
    glEnable(_texture_target);
    assert(glGetError() == GL_NONE);
    glBindTexture(_texture_target, texture_id());
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    assert(glGetError() == GL_NONE);
}

void texture::unbind() const
{
    assert(glGetError() == GL_NONE);
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glDisableIndexedEXT(_texture_target, _occupied_texture_unit);
    glBindMultiTextureEXT(GL_TEXTURE0 + _occupied_texture_unit, _texture_target, 0);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    if (_occupied_texture_unit >= 0) {
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        _occupied_texture_unit = -1;
    }
    glBindTexture(_texture_target, 0);
    glDisable(_texture_target);
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    assert(glGetError() == GL_NONE);
}

void texture::checked_lazy_generate_texture_id() const
{
    if (*_texture_id == 0) {
        glGenTextures(1, _texture_id.get());
    }
}

} // namespace gl
} // namespace scm
