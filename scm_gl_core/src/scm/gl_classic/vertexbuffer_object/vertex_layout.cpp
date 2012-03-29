
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "vertex_layout.h"

#include <cassert>

#include <scm/core/math/math.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_classic/opengl.h>

namespace std {

void swap(scm::gl_classic::vertex_element& lhs,
          scm::gl_classic::vertex_element& rhs)
{
    lhs.swap(rhs);
}

void swap(scm::gl_classic::vertex_layout& lhs,
          scm::gl_classic::vertex_layout& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

namespace scm {
namespace gl_classic {

// vertex_element
vertex_element::vertex_element(vertex_element::binding_type bind_type,
                               vertex_element::data_type    dta_type,
                               unsigned                     num,
                               const std::string&           name)
  : _binding(bind_type),
    _data_type(dta_type),
    _num(num),
    _name(name),
    _data_size(data_size(dta_type)),
    _data_num_components(data_num_components(dta_type)),
    _data_elem_type(data_elem_type(dta_type))
{
}

vertex_element::vertex_element(const vertex_element& rhs)
  : _binding(rhs._binding),
    _data_type(rhs._data_type),
    _num(rhs._num),
    _name(rhs._name),
    _data_size(rhs._data_size),
    _data_num_components(rhs._data_num_components),
    _data_elem_type(rhs._data_elem_type)
{
}

vertex_element::~vertex_element()
{
}

const vertex_element& vertex_element::operator=(const vertex_element& rhs)
{
    _binding                = rhs._binding;
    _data_type              = rhs._data_type;
    _num                    = rhs._num;
    _name                   = rhs._name;
    _data_size              = rhs._data_size;
    _data_num_components    = rhs._data_num_components;
    _data_elem_type         = rhs._data_elem_type;

    return (*this);
}

void vertex_element::swap(vertex_element& rhs)
{
    std::swap(_binding                , rhs._binding);
    std::swap(_data_type              , rhs._data_type);
    std::swap(_num                    , rhs._num);
    std::swap(_name                   , rhs._name);
    std::swap(_data_size              , rhs._data_size);
    std::swap(_data_num_components    , rhs._data_num_components);
    std::swap(_data_elem_type         , rhs._data_elem_type);
}

std::size_t vertex_element::data_size(data_type dt)
{
    switch (dt) {
        case dt_float:return (sizeof(float));
        case dt_vec2f:return (sizeof(scm::math::vec2f));
        case dt_vec3f:return (sizeof(scm::math::vec3f));
        case dt_vec4f:return (sizeof(scm::math::vec4f));
        default: return (0);
    }
}

std::size_t vertex_element::data_num_components(data_type dt)
{
    switch (dt) {
        case dt_float:return (1);
        case dt_vec2f:return (2);
        case dt_vec3f:return (3);
        case dt_vec4f:return (4);
        default: return (0);
    }
}

unsigned vertex_element::data_elem_type(data_type dt)
{
    switch (dt) {
        case dt_float:return (GL_FLOAT);
        case dt_vec2f:return (GL_FLOAT);
        case dt_vec3f:return (GL_FLOAT);
        case dt_vec4f:return (GL_FLOAT);
        default: return (GL_NONE);
    }
}

// vertex_layout
vertex_layout::vertex_layout()
{
}

vertex_layout::vertex_layout(const element_container& elem)
  : _elements(elem)
{
}

vertex_layout::vertex_layout(const vertex_layout& rhs)
  : _elements(rhs._elements)
{
}

vertex_layout::~vertex_layout()
{
}

const vertex_layout& vertex_layout::operator=(const vertex_layout& rhs)
{
    _elements.assign(rhs._elements.begin(),
                     rhs._elements.end());

    return (*this);
}

void vertex_layout::swap(vertex_layout& rhs)
{
    std::swap(_elements, rhs._elements);
}

std::size_t vertex_layout::num_elements() const
{
    return (_elements.size());
}

std::size_t vertex_layout::size() const
{
    return (stride());
}

std::size_t vertex_layout::stride() const
{
    assert(!_elements.empty());

    std::size_t     stride = 0;

    foreach (const vertex_element& elem, _elements) {
        stride += elem._data_size;
    }

    return (stride);
}

std::size_t vertex_layout::offset(unsigned elem_idx) const
{
    assert(elem_idx < num_elements());

    std::size_t     offset = 0;

    for (unsigned i = 0; i < elem_idx; ++i) {
        offset += _elements.at(i)._data_size;
    }

    return (offset);
}

const vertex_element& vertex_layout::element(unsigned elem_idx) const
{
    assert(elem_idx < num_elements());

    return (_elements.at(elem_idx));
}

} // namespace gl_classic
} // namespace scm
