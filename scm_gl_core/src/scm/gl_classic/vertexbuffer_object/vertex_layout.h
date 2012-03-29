
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_OGL_VERTEX_LAYOUT_H_INCLUDED
#define SCM_OGL_VERTEX_LAYOUT_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) vertex_element
{
public:
    typedef enum {
        position,
        normal,
        tex_coord,
        color,
        vertex_attrib
    } binding_type;

    typedef enum {
        dt_float,
        dt_vec2f,
        dt_vec3f,
        dt_vec4f
    } data_type;

public:
    vertex_element(binding_type         /*bind_type*/,
                   data_type            /*dta_type*/,
                   unsigned             /*num*/  = 0,
                   const std::string&   /*name*/ = std::string(""));
    vertex_element(const vertex_element& rhs);
    virtual ~vertex_element();

    const vertex_element&       operator=(const vertex_element& /*rhs*/);
    void                        swap(vertex_element& /*rhs*/);

    binding_type                _binding;
    data_type                   _data_type;
    unsigned                    _num;
    std::string                 _name;

    std::size_t                 _data_size;
    std::size_t                 _data_num_components;
    unsigned                    _data_elem_type;

private:
    static std::size_t          data_size(data_type /*dt*/);
    static std::size_t          data_num_components(data_type /*dt*/);
    static unsigned             data_elem_type(data_type /*dt*/);

}; // class vertex_element

class __scm_export(gl_core) vertex_layout
{
public:
    typedef std::vector<vertex_element> element_container;

public:
    vertex_layout();
    vertex_layout(const element_container& /*elem*/);
    vertex_layout(const vertex_layout& /*rhs*/);
    virtual ~vertex_layout();

    const vertex_layout&        operator=(const vertex_layout& /*rhs*/);
    void                        swap(vertex_layout& /*rhs*/);

    std::size_t                 num_elements() const;
    std::size_t                 size() const;
    std::size_t                 stride() const;
    std::size_t                 offset(unsigned /*elem_idx*/) const;

    const vertex_element&       element(unsigned /*elem_idx*/) const;

protected:
    std::vector<vertex_element> _elements;

}; // class vertex_layout

} // namespace gl_classic
} // namespace scm

namespace std {

void __scm_export(gl_core) swap(scm::gl_classic::vertex_element& /*lhs*/,
                            scm::gl_classic::vertex_element& /*rhs*/);

void __scm_export(gl_core) swap(scm::gl_classic::vertex_layout& /*lhs*/,
                            scm::gl_classic::vertex_layout& /*rhs*/);

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_VERTEX_LAYOUT_H_INCLUDED
