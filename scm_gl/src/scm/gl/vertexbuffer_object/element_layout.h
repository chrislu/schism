
#ifndef SCM_OGL_ELEMENT_LAYOUT_H_INCLUDED
#define SCM_OGL_ELEMENT_LAYOUT_H_INCLUDED

#include <cstddef>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) element_layout
{
public:
    typedef enum {
        triangles,
        triangle_strip,
        triangle_fan,
        quads,
        quad_strip,
        lines,
        line_strip,
        line_loop,
        points
    } primitive_type;

    typedef enum {
        dt_ubyte,
        dt_ushort,
        dt_uint
    } data_type;

public:
    element_layout();
    element_layout(primitive_type       /*pt*/,
                   data_type            /*dt*/);
    element_layout(const element_layout& rhs);
    virtual ~element_layout();

    const element_layout&       operator=(const element_layout& /*rhs*/);
    void                        swap(element_layout& /*rhs*/);

    std::size_t                 _size;
    unsigned                    _type;
    unsigned                    _primitive_type;

private:
    static std::size_t          data_size(data_type /*dt*/);
    static unsigned             data_elem_type(data_type /*dt*/);
    static unsigned             prim_type(primitive_type /*pt*/);

}; // class element_layout

} // namespace gl
} // namespace scm

namespace std {

void __scm_export(ogl) swap(scm::gl::element_layout& /*lhs*/,
                            scm::gl::element_layout& /*rhs*/);

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_ELEMENT_LAYOUT_H_INCLUDED
