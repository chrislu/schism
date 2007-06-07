
#ifndef TEXTURE_3D_H_INCLUDED
#define TEXTURE_3D_H_INCLUDED

#include <scm/ogl/textures/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture_3d : public texture
{
public:
    texture_3d();
    texture_3d(const gl::texture_3d& tex);
    virtual ~texture_3d();

    const gl::texture_3d& operator=(const gl::texture_3d& rhs);

    unsigned        get_width() const       { return (_width); }
    unsigned        get_height() const      { return (_height); }
    unsigned        get_depth() const       { return (_depth); }

    bool            tex_image(GLint     mip_level,
                              GLint     internal_format,
                              GLsizei   width,
                              GLsizei   height,
                              GLsizei   depth,
                              GLenum    format,
                              GLenum    type,
                              const GLvoid *data);

protected:
    unsigned        _width;
    unsigned        _height;
    unsigned        _depth;

private:

}; // class texture_3d

} // namespace gl
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // TEXTURE_3D_H_INCLUDED
