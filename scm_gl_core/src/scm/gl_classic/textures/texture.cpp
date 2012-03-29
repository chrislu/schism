
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture.h"

#include <cassert>
#include <scm/gl_classic/utilities/gl_assert.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

texture::binding_guard::binding_guard(unsigned target, unsigned binding)
  : _save_active_texture_unit(0),
    _save_texture_object(0),
    _binding(binding),
    _target(target)
{
    gl_assert_error("entering texture::binding_guard::binding_guard()");

    glGetIntegerv(GL_ACTIVE_TEXTURE, &_save_active_texture_unit);
    glGetIntegerv(_binding, &_save_texture_object);

    gl_assert_error("exiting texture::binding_guard::binding_guard()");
}

texture::binding_guard::~binding_guard()
{
    gl_assert_error("entering texture::binding_guard::~binding_guard()");

    glActiveTexture(_save_active_texture_unit);
    glBindTexture(_target, _save_texture_object);

    gl_assert_error("exiting texture::binding_guard::~binding_guard()");
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

texture::~texture()
{
    if (_texture_id.unique()) {
        glDeleteTextures(1, _texture_id.get());
    }
}

void texture::parameter(GLenum pname, int param)
{
    gl_assert_error("entering texture::parameter(GLenum, int)");
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureParameteriEXT(id(), target(), pname, param);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexParameteri(target(), pname, param);
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    gl_assert_error("exiting texture::parameter(GLenum, int)");
}

void texture::parameter(GLenum pname, float param)
{
    gl_assert_error("entering texture::parameter(GLenum, float)");
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glTextureParameterfEXT(id(), target(), pname, param);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    binding_guard guard(target(), binding());
    bind();
    glTexParameterf(target(), pname, param);
    unbind();
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    gl_assert_error("exiting texture::parameter(GLenum, float)");
}

int texture::target() const
{
    return (_texture_target);
}

int texture::binding() const
{
    return (_texture_binding);
}

unsigned texture::id() const
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
    gl_assert_error("entering texture::bind()");
    assert(_texture_id);
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    _occupied_texture_unit = texunit > 0 ? texunit : 0;
    glBindMultiTextureEXT(GL_TEXTURE0 + _occupied_texture_unit, target(), id());
    glEnableIndexedEXT(target(), _occupied_texture_unit);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    if (texunit >= 0) {
        _occupied_texture_unit = texunit;
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        gl_assert_error("texture::bind() after glActiveTexture");
    }
    glEnable(target());
    gl_assert_error("texture::bind() after glEnable(glEnable)");
    glBindTexture(target(), id());
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    gl_assert_error("exiting texture::bind()");
}

void texture::unbind() const
{
    gl_assert_error("entering texture::unbind()");
#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    glBindMultiTextureEXT(GL_TEXTURE0 + _occupied_texture_unit, target(), 0);
    glDisableIndexedEXT(target(), _occupied_texture_unit);
#else // SCM_GL_USE_DIRECT_STATE_ACCESS
    if (_occupied_texture_unit >= 0) {
        glActiveTexture(GL_TEXTURE0 + _occupied_texture_unit);
        _occupied_texture_unit = -1;
    }
    glBindTexture(target(), 0);
    glDisable(target());
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS
    gl_assert_error("exiting texture::unbind()");
}

void texture::checked_lazy_generate_texture_id() const
{
    if (*_texture_id == 0) {
        glGenTextures(1, _texture_id.get());
    }
}

} // namespace gl_classic
} // namespace scm
