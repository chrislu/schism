
#ifndef TEXTURE_H_INCLUDED
#define TEXTURE_H_INCLUDED

#include <scm/ogl/gl.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture
{
public:
    texture(const gl::texture& tex);
    virtual ~texture();

    const gl::texture& operator=(const gl::texture& rhs);

    void                bind(int /*texunit*/ = -1) const;
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

    mutable int         _occupied_texture_unit;

}; // class texture

} // namespace gl
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // TEXTURE_H_INCLUDED
