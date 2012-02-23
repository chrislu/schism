
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "viewport.h"

namespace scm {
namespace gl {

viewport::viewport(const math::vec2ui& in_position,
                   const math::vec2ui& in_dimensions,
                   const math::vec2f&  in_depth_range)
  : _position(in_position),
    _dimensions(in_dimensions),
    _depth_range(in_depth_range)
{
}

viewport::viewport(const math::vec2f& in_position,
                   const math::vec2f& in_dimensions,
                   const math::vec2f& in_depth_range)
  : _position(in_position),
    _dimensions(in_dimensions),
    _depth_range(in_depth_range)
{
}

viewport_array
viewport::operator()(const math::vec2f& in_position,
                     const math::vec2f& in_dimensions,
                     const math::vec2f& in_depth_range)
{
    viewport_array ret(*this);

    return (ret(in_position, in_dimensions, in_depth_range));
}

bool
viewport::operator==(const viewport& rhs) const
{
    return (   (_position    == rhs._position)
            && (_dimensions  == rhs._dimensions)
            && (_depth_range == rhs._depth_range));
}

bool
viewport::operator!=(const viewport& rhs) const
{
    return (   (_position    != rhs._position)
            || (_dimensions  != rhs._dimensions)
            || (_depth_range != rhs._depth_range));
}


viewport_array::viewport_array(const viewport& in_viewport)
  : _array(1, in_viewport)
{
}

viewport_array::viewport_array(const math::vec2f& in_position,
                               const math::vec2f& in_dimensions,
                               const math::vec2f& in_depth_range)
  : _array(1, viewport(in_position, in_dimensions, in_depth_range))
{
}

viewport_array&
viewport_array::operator()(const viewport& in_viewport)
{
    _array.push_back(in_viewport);
    return (*this);
}

viewport_array&
viewport_array::operator()(const math::vec2f& in_position,
                           const math::vec2f& in_dimensions,
                           const math::vec2f& in_depth_range)
{
    _array.push_back(viewport(in_position, in_dimensions, in_depth_range));
    return (*this);
}

size_t
viewport_array::size() const
{
    return (_array.size());
}

const viewport_array::viewport_vector&
viewport_array::viewports() const
{
    return (_array);
}

bool
viewport_array::operator==(const viewport_array& rhs) const
{
    return (_array == rhs._array);
}

bool
viewport_array::operator!=(const viewport_array& rhs) const
{
    return (_array != rhs._array);
}

} // namespace gl
} // namespace scm
