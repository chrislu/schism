
#ifndef TEXTURE_1D_H_INCLUDED
#define TEXTURE_1D_H_INCLUDED

#include <scm/ogl/textures/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture_1d : public texture
{
public:
    texture_1d();
    virtual ~texture_1d();

    const gl::texture_1d& operator=(const gl::texture_1d& rhs);

    unsigned        get_width() const { return (_width); }

    bool            tex_image(GLint     mip_level,
                              GLint     internal_format,
                              GLsizei   width,
                              GLenum    format,
                              GLenum    type,
                              const GLvoid *data);

protected:
    unsigned        _width;

private:

}; // class texture_1d

} // namespace gl
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // TEXTURE_1D_H_INCLUDED
