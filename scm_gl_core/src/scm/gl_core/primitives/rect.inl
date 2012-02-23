
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

#include <scm/gl_core/primitives/ray.h>

namespace scm {
namespace gl {

template<typename s>
rect_impl<s>::rect_impl(const vec2_type& min_vert,
                        const vec2_type& max_vert)
  : _min_vertex(min_vert, typename vec3_type::value_type(0)),
    _max_vertex(max_vert, typename vec3_type::value_type(0)),
    _poly_plane(vec3_type(min_vert, typename vec3_type::value_type(0)),
                vec3_type(max_vert.x, min_vert.y, typename vec3_type::value_type(0)),
                vec3_type(max_vert, typename vec3_type::value_type(0)))
{
}

template<typename s>
rect_impl<s>::rect_impl(const rect_impl<s>& b)
  : _min_vertex(b._min_vertex),
    _max_vertex(b._max_vertex),
    _poly_plane(b._poly_plane)
{
}

template<typename s>
rect_impl<s>::~rect_impl()
{
}

template<typename s>
rect_impl<s>&
rect_impl<s>::operator=(const rect_impl<s>& b)
{
    rect_impl<s> tmp(b);
    swap(tmp);
    return (*this);
}

template<typename s>
void
rect_impl<s>::swap(rect_impl<s>& b)
{
    std::swap(_min_vertex, b._min_vertex);
    std::swap(_max_vertex, b._max_vertex);
    std::swap(_poly_plane, b._poly_plane);
}

template<typename s>
const typename rect_impl<s>::vec3_type&
rect_impl<s>::min_vertex() const
{
    return (_min_vertex);
}

template<typename s>
const typename rect_impl<s>::vec3_type&
rect_impl<s>::max_vertex() const
{
    return (_max_vertex);
}

template<typename s>
const typename rect_impl<s>::vec3_type
rect_impl<s>::center() const
{
    return ((_max_vertex + _min_vertex) * 0.5f);
}

template<typename s>
const typename rect_impl<s>::vec3_type
rect_impl<s>::corner(unsigned ind) const
{
    return (vec3_type(ind & 1 ? _max_vertex.x : _min_vertex.x,
                      ind & 2 ? _max_vertex.y : _min_vertex.y,
                      ind & 4 ? _max_vertex.z : _min_vertex.z));
}

template<typename s>
void
rect_impl<s>::min_vertex(const vec2_type& vert)
{
    _min_vertex = vec3_type(vert, vec2_type::value_type(0));
}
template<typename s>
const typename rect_impl<s>::plane_type&
rect_impl<s>::poly_plane() const
{
    return (_poly_plane);
}

template<typename s>
void
rect_impl<s>::max_vertex(const vec2_type& vert)
{
    _max_vertex = vec3_type(vert, vec2_type::value_type(0));
}

template<typename s>
typename rect_impl<s>::classification_result
rect_impl<s>::classify(const typename rect_impl<s>::vec3_type& p) const
{
    // precondition: point on plane

    // check point inside bounds
    for (unsigned i = 0; i < 2; ++i) {
        if (   _min_vertex[i] > p[i]
            || _max_vertex[i] < p[i]) {
            return (outside);
        }
    }

    return (inside);
}

template<typename s>
typename rect_impl<s>::classification_result
rect_impl<s>::classify(const rect_impl<s>& a) const
{
    for (unsigned i = 0; i < 2; ++i) {
        if (   _min_vertex[i] > a._max_vertex[i]
            || _max_vertex[i] < a._min_vertex[i]) {
            return (outside);
        }
    }

    return (overlaping);
}

template<typename s>
bool
rect_impl<s>::intersect(const ray_type& r,
                        vec3_type&      hit) const
{
    if (!_poly_plane.intersect(r, hit)) {
        return (false);
    }
    if (classify(hit) == outside) {
        return (false);
    }
    return (true);
}

} // namespace gl
} // namespace scm
