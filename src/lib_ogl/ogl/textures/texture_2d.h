
#ifndef TEXTURE_2D_H_INCLUDED
#define TEXTURE_2D_H_INCLUDED

#include <ogl/textures/texture.h>

namespace gl
{
    class texture_2d : public texture
    {
    public:
        texture_2d();
        virtual ~texture_2d();

        const gl::texture_2d& operator=(const gl::texture_2d& rhs);

        unsigned        get_width() const       { return (_width); }
        unsigned        get_height() const      { return (_height); }

        bool            tex_image(GLint     mip_level,
                                  GLint     internal_format,
                                  GLsizei   width,
                                  GLsizei   height,
                                  GLenum    format,
                                  GLenum    type,
                                  const GLvoid *data);


    protected:
        texture_2d(const GLenum target);

        unsigned        _width;
        unsigned        _height;

    private:
    }; // class texture_2d

} // namespace gl

#endif // TEXTURE_2D_H_INCLUDED



