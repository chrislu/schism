
#ifndef TEXTURE_1D_H_INCLUDED
#define TEXTURE_1D_H_INCLUDED

#include <scm/gl/textures/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture_1d : public texture
{
public:
    texture_1d();
    virtual ~texture_1d();

    unsigned        width() const { return (_width); }

    bool            image_data(GLint     mip_level,
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

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TEXTURE_1D_H_INCLUDED
