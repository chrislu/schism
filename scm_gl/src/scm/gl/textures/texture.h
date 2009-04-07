
#ifndef TEXTURE_H_INCLUDED
#define TEXTURE_H_INCLUDED

#include <scm/gl/opengl.h>

#include <boost/shared_ptr.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture
{
public:
    texture(const gl::texture& tex);
    virtual ~texture();

    const gl::texture&              operator=(const gl::texture& rhs);

    void                            bind(int /*texunit*/ = -1) const;
    void                            unbind() const;

    void                            tex_parameteri(GLenum pname, GLint param);

    int                             get_texture_target() const;
    unsigned                        get_texture_id() const;
    int                             get_last_error() const;

protected:
    texture(const GLenum target);

private:
    void                                checked_lazy_generate_texture_id() const ;

protected:
    GLenum                              _last_error;

private:
    GLenum                              _texture_target;
    mutable boost::shared_ptr<GLuint>   _texture_id;

    mutable int                         _occupied_texture_unit;
}; // class texture

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TEXTURE_H_INCLUDED
