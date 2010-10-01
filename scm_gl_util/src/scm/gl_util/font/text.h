
#ifndef SCM_GL_UTIL_TEXT_H_INCLUDED
#define SCM_GL_UTIL_TEXT_H_INCLUDED

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) text
{
public:
    text(const render_device_ptr&    device,
         const font_face_cptr&       font,
         const font_face::style_type stl,
         const std::string&          str);
    virtual ~text();

    const font_face_cptr&       font() const;
    const font_face::style_type text_style() const;
    const std::string&          text_string() const;
    void                        text_string(const std::string& str);
    void                        text_string(const std::string& str,
                                            const font_face::style_type stl);
    bool                        text_kerning() const;
    void                        text_kerning(bool k);

protected:
    void                        update();

protected:
    font_face_cptr              _font;
    font_face::style_type       _text_style;
    std::string                 _text_string;
    bool                        _text_kerning;

    int                         _glyph_capacity;
    buffer_ptr                  _vertex_buffer;
    buffer_ptr                  _index_buffer;
    int                         _indices_count;
    primitive_topology          _topology;

    vertex_array_ptr            _vertex_array;

    render_device_wptr          _render_device;
    render_context_wptr         _render_context;

    friend class text_renderer;
}; // class text

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXT_H_INCLUDED
