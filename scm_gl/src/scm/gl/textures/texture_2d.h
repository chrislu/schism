
#ifndef TEXTURE_2D_H_INCLUDED
#define TEXTURE_2D_H_INCLUDED

#include <scm/gl/textures/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture_2d : public texture
{
public:
    texture_2d();
    virtual ~texture_2d();

    const gl::texture_2d& operator=(const gl::texture_2d& rhs);

    unsigned        width() const       { return (_width); }
    unsigned        height() const      { return (_height); }

    bool            tex_image(GLint     mip_level,
                              GLint     internal_format,
                              GLsizei   width,
                              GLsizei   height,
                              GLenum    format,
                              GLenum    type,
                              const GLvoid *data);

    bool            tex_sub_image(GLint     mip_level,
                                  GLint     off_x,
                                  GLint     off_y,
                                  GLsizei   width,
                                  GLsizei   height,
                                  GLenum    format,
                                  GLenum    type,
                                  const GLvoid *data);

protected:
    texture_2d(const GLenum target, const GLenum binding);

    unsigned        _width;
    unsigned        _height;

private:

}; // class texture_2d

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TEXTURE_2D_H_INCLUDED
