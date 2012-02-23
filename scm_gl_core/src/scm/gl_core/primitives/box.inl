
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

#include <scm/gl_core/primitives/ray.h>

namespace scm {
namespace gl {

template<typename s>
box_impl<s>::box_impl(const typename box_impl<s>::vec3_type& min_vert,
                      const typename box_impl<s>::vec3_type& max_vert)
  : _min_vertex(min_vert),
    _max_vertex(max_vert)
{
}

template<typename s>
box_impl<s>::box_impl(const box_impl<s>& b)
  : _min_vertex(b._min_vertex),
    _max_vertex(b._max_vertex)
{
}

template<typename s>
box_impl<s>::~box_impl()
{
}

template<typename s>
box_impl<s>&
box_impl<s>::operator=(const box_impl<s>& b)
{
    box_impl<s> tmp(b);
    swap(tmp);
    return (*this);
}

template<typename s>
void
box_impl<s>::swap(box_impl<s>& b)
{
    std::swap(_min_vertex, b._min_vertex);
    std::swap(_max_vertex, b._max_vertex);
}

template<typename s>
const typename box_impl<s>::vec3_type&
box_impl<s>::min_vertex() const
{
    return (_min_vertex);
}

template<typename s>
const typename box_impl<s>::vec3_type&
box_impl<s>::max_vertex() const
{
    return (_max_vertex);
}

template<typename s>
const typename box_impl<s>::vec3_type
box_impl<s>::center() const
{
    return ((_max_vertex + _min_vertex) * 0.5f);
}

template<typename s>
const typename box_impl<s>::vec3_type
box_impl<s>::corner(unsigned ind) const
{
    return (vec3_type(ind & 1 ? _max_vertex.x : _min_vertex.x,
                      ind & 2 ? _max_vertex.y : _min_vertex.y,
                      ind & 4 ? _max_vertex.z : _min_vertex.z));
}

template<typename s>
void
box_impl<s>::min_vertex(const vec3_type& vert)
{
    _min_vertex = vert;
}

template<typename s>
void
box_impl<s>::max_vertex(const vec3_type& vert)
{
    _max_vertex = vert;
}

template<typename s>
typename box_impl<s>::classification_result
box_impl<s>::classify(const vec3_type& p) const
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   _min_vertex[i] > p[i]
            || _max_vertex[i] < p[i]) {
            return (outside);
        }
    }
    return (inside);
}

template<typename s>
typename box_impl<s>::classification_result
box_impl<s>::classify(const box_impl<s>& a) const
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   _min_vertex[i] > a._max_vertex[i]
            || _max_vertex[i] < a._min_vertex[i]) {
            return (outside);
        }
    }

    return (overlaping);
}

template<typename s>
bool
box_impl<s>::intersect(const typename box_impl<s>::ray_type& r,
                       vec3_type& entry,
                       vec3_type& exit) const
{
    typename vec3_type::value_type tmin, tmax, tymin, tymax, tzmin, tzmax;

    const vec3_type& org = r.origin();
    const vec3_type& dir = r.direction();

    const vec3_type& b_min = _min_vertex;
    const vec3_type& b_max = _max_vertex;

    vec3_type rec_dir(vec3_type(typename vec3_type::value_type(1)) / dir);

    if (dir.x >= typename vec3_type::value_type(0)) {
        tmin = (b_min.x - org.x) * rec_dir.x;
        tmax = (b_max.x - org.x) * rec_dir.x;
    }
    else {
        tmin = (b_max.x - org.x) * rec_dir.x;
        tmax = (b_min.x - org.x) * rec_dir.x;
    }

    if (dir.y >= typename vec3_type::value_type(0)) {
        tymin = (b_min.y - org.y) * rec_dir.y;
        tymax = (b_max.y - org.y) * rec_dir.y;
    }
    else {
        tymin = (b_max.y - org.y) * rec_dir.y;
        tymax = (b_min.y - org.y) * rec_dir.y;
    }

    if ((tmin > tymax) || (tymin > tmax)) {
        return (false);
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    if (dir.z >= typename vec3_type::value_type(0)) {
        tzmin = (b_min.z - org.z) * rec_dir.z;
        tzmax = (b_max.z - org.z) * rec_dir.z;
    }
    else {
        tzmin = (b_max.z - org.z) * rec_dir.z;
        tzmax = (b_min.z - org.z) * rec_dir.z;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return (false);
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    // calculate intersection points
    entry = org + (tmin * r.direction());
    exit  = org + (tmax * r.direction());

    //return ((tmin < t1) && (tmax > t0));
    
    return (tmin > typename vec3_type::value_type(0));
}

} // namespace gl
} // namespace scm

