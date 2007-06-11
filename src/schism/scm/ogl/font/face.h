
#ifndef GL_FONT_FACE_H_INCLUDED
#define GL_FONT_FACE_H_INCLUDED

#include <cstddef>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <scm/core/font/face.h>

#include <scm/ogl/textures/texture_2d_rect.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) face : public scm::font::face
{
protected:
    typedef boost::shared_ptr<texture_2d_rect>      texture_ptr;
    typedef std::map<font::face::style_type,
                     texture_ptr>                   style_textur_container;

public:
    face();
    virtual ~face();

    const texture_2d_rect&          get_glyph_texture(font::face::style_type /*style*/ = regular) const;

protected:
    void                            cleanup_textures();

    style_textur_container          _style_textures;

private:
    friend class face_loader;

}; // class gl_font_base

typedef boost::shared_ptr<scm::gl::face>    face_ptr;
typedef boost::shared_ptr<const scm::gl::face>    face_const_ptr;

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // GL_FONT_FACE_H_INCLUDED
