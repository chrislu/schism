
#include "texture.h"

namespace gl
{
    texture::texture(const GLenum target)
        : _texture_id(0),
          _texture_target(target),
          _last_error(GL_NO_ERROR)
    {
    }

    texture::texture(const gl::texture& tex)
        : _texture_target(tex._texture_target),
          _texture_id(tex._texture_id),
          _last_error(GL_NO_ERROR)
    {
    }

    texture::~texture()
    {
        delete_texture();
    }

    const gl::texture& texture::operator=(const gl::texture& rhs)
    {
        this->_texture_id = rhs._texture_id;
        this->_texture_target = rhs._texture_target;

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

    void texture::bind() const
    {
        glEnable(_texture_target);
        glBindTexture(_texture_target, _texture_id);
    }

    void texture::unbind() const
    {
        glBindTexture(_texture_target, 0);
        glDisable(_texture_target);
    }

    void texture::generate_texture_id()
    {
        glGenTextures(1, &_texture_id);
    }

} // namespace gl



