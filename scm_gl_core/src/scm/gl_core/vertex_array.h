
#ifndef SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED
#define SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED

#include <string>
#include <vector>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/program.h>
#include <scm/gl_core/render_device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) vertex_format : public render_device_child
{
public:
    struct element {
        element(unsigned s, const std::string& n,     data_type t, int o = 0, bool n = false) : _buffer_slot(s), _attrib_name(n), _attrib_location(-1), _type(t), _stride(o), _normalize(n) {}
        element(unsigned s, program::location_type l, data_type t, int o = 0, bool n = false) : _buffer_slot(s), _attrib_location(l), _type(t), _stride(o), _normalize(n) {}

        bool operator==(const element& e) const;
        bool operator!=(const element& e) const;

        unsigned                _buffer_slot;
        std::string             _attrib_name;
        program::location_type  _attrib_location;
        data_type               _type;
        int                     _stride;
        bool                    _normalize;
    }; // struct element
    typedef std::vector<element>        element_container;

public:
    virtual ~vertex_format();

protected:
    vertex_format(render_device& ren_dev,
        );

protected:
    element_container       _elements;

}; // class vertex_format

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_VERTEX_ARRAY_H_INCLUDED
