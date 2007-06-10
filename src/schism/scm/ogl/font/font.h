
#ifndef GL_FONT_H_INCLUDED
#define GL_FONT_H_INCLUDED

#include <cstddef>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <scm/core/font/face.h>
#include <scm/core/resource/resource.h>
#include <scm/core/resource/resource_pointer.h>
#include <scm/core/resource/resource_manager.h>

#include <scm/ogl/textures/texture_2d_rect.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class font_resource_loader;

class __scm_export(ogl) font_descriptor
{
public:
    std::size_t             hash_value() const;

    std::string             _name;
    unsigned                _size;

}; // class gl_font_descroptor

class __scm_export(ogl) font_face_resource : public font::face,
                                             public res::resource<font_descriptor>
{
protected:
    typedef boost::shared_ptr<texture_2d_rect>      texture_ptr;
    typedef std::map<font::face::style_type,
                     texture_ptr>                   style_textur_container;

public:
    virtual ~font_face_resource();

    const texture_2d_rect&          get_glyph_texture(font::face::style_type /*style*/ = regular) const;

protected:
    font_face_resource(const font_descriptor& /*desc*/);

    void                            cleanup_textures();

    style_textur_container          _style_textures;

private:
    friend class res::resource_manager<font_face_resource>;
    friend class font_resource_loader;

}; // class gl_font_base

typedef res::resource_pointer<font_face_resource>   font_face;

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // GL_FONT_H_INCLUDED
