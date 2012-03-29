
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "element_layout.h"

#include <algorithm>

#include <scm/gl_classic/opengl.h>

namespace std {

void swap(scm::gl_classic::element_layout& lhs,
          scm::gl_classic::element_layout& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

namespace scm {
namespace gl_classic {

element_layout::element_layout()
  : _size(0),
    _type(GL_NONE),
    _primitive_type(GL_NONE)
{
}

element_layout::element_layout(element_layout::primitive_type pt,
                               element_layout::data_type      dt)
  : _size(data_size(dt)),
    _type(data_elem_type(dt)),
    _primitive_type(prim_type(pt))
{
}

element_layout::element_layout(const element_layout& rhs)
  : _size(rhs._size),
    _type(rhs._type),
    _primitive_type(rhs._primitive_type)
{
}

element_layout::~element_layout()
{
}

const element_layout& element_layout::operator=(const element_layout& rhs)
{
    _size           = rhs._size;
    _type           = rhs._type;
    _primitive_type = rhs._primitive_type;

    return (*this);
}

void element_layout::swap(element_layout& rhs)
{
    std::swap(_size           , rhs._size);
    std::swap(_type           , rhs._type);
    std::swap(_primitive_type , rhs._primitive_type);
}

std::size_t element_layout::data_size(element_layout::data_type dt)
{
    switch (dt) {
        case dt_ubyte:  return (sizeof(unsigned char));
        case dt_ushort: return (sizeof(unsigned short));
        case dt_uint:   return (sizeof(unsigned int));
        default:        return (0);
    }
}

unsigned element_layout::data_elem_type(element_layout::data_type dt)
{
    switch (dt) {
        case dt_ubyte:  return (GL_UNSIGNED_BYTE);
        case dt_ushort: return (GL_UNSIGNED_SHORT);
        case dt_uint:   return (GL_UNSIGNED_INT);
        default:        return (GL_NONE);
    }
}

unsigned element_layout::prim_type(element_layout::primitive_type pt)
{
    switch (pt) {
        case triangles:         return (GL_TRIANGLES);
        case triangle_strip:    return (GL_TRIANGLE_STRIP);
        case triangle_fan:      return (GL_TRIANGLE_FAN);
        case quads:             return (GL_QUADS);
        case quad_strip:        return (GL_QUAD_STRIP);
        case lines:             return (GL_LINES);
        case line_strip:        return (GL_LINE_STRIP);
        case line_loop:         return (GL_LINE_LOOP);
        case points:            return (GL_POINTS);
        default:                return (GL_NONE);
    }
}

} // namespace gl_classic
} // namespace scm
