
#ifndef TEXTURE_H_INCLUDED
#define TEXTURE_H_INCLUDED

#include <ogl/gl.h>

namespace gl
{
    class texture
    {
    public:
        texture(const gl::texture& tex);
        virtual ~texture();

        const gl::texture& operator=(const gl::texture& rhs);

        void                bind() const;
        void                unbind() const;
        void                delete_texture();

        void                tex_parameteri(GLenum pname, GLint param);

        int                 get_texture_target() const      { return (_texture_target); }
        unsigned            get_texture_id() const          { return (_texture_id); }
        int                 get_last_error() const          { return (_last_error); }

    protected:
        texture(const GLenum target);

        void                generate_texture_id();

    protected:
        GLenum              _last_error;

    private:
        GLenum              _texture_target;
        unsigned            _texture_id;

    }; // class texture

} // namespace gl

#endif // TEXTURE_H_INCLUDED


