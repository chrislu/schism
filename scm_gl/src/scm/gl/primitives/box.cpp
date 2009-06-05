
#include "box.h"

#include <algorithm>

#include <scm/gl/primitives/ray.h>

namespace scm {
namespace gl {

box::box(const box::vec3_type& min_vert,
         const box::vec3_type& max_vert)
  : _min_vertex(min_vert),
    _max_vertex(max_vert)
{
}

box::box(const box& b)
  : _min_vertex(b._min_vertex),
    _max_vertex(b._max_vertex)
{
}

box::~box()
{
}

box&
box::operator=(const box& b)
{
    box tmp(b);
    swap(tmp);
    return (*this);
}

void
box::swap(box& b)
{
    std::swap(_min_vertex, b._min_vertex);
    std::swap(_max_vertex, b._max_vertex);
}

const box::vec3_type&
box::min_vertex() const
{
    return (_min_vertex);
}

const box::vec3_type&
box::max_vertex() const
{
    return (_max_vertex);
}

const box::vec3_type
box::center() const
{
    return ((_max_vertex + _min_vertex) * 0.5f);
}

const box::vec3_type
box::corner(unsigned ind) const
{
    return (vec3_type(ind & 1 ? _max_vertex.x : _min_vertex.x,
                      ind & 2 ? _max_vertex.y : _min_vertex.y,
                      ind & 4 ? _max_vertex.z : _min_vertex.z));
}

void
box::min_vertex(const box::vec3_type& vert)
{
    _min_vertex = vert;
}

void
box::max_vertex(const box::vec3_type& vert)
{
    _max_vertex = vert;
}

box::classification_result
box::classify(const box::vec3_type& p) const
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   _min_vertex[i] > p[i]
            || _max_vertex[i] < p[i]) {
            return (outside);
        }
    }
    return (inside);
}

box::classification_result
box::classify(const box& a) const
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   _min_vertex[i] > a._max_vertex[i]
            || _max_vertex[i] < a._min_vertex[i]) {
            return (outside);
        }
    }

    return (overlap);
}

bool
box::intersect(const ray& r,
               box::vec3_type& entry,
               box::vec3_type& exit) const
{
    vec3_type::value_type tmin, tmax, tymin, tymax, tzmin, tzmax;

    const vec3_type& org = r.origin();
    const vec3_type& dir = r.direction();

    const vec3_type& b_min = _min_vertex;
    const vec3_type& b_max = _max_vertex;

    vec3_type rec_dir(vec3_type(vec3_type::value_type(1)) / dir);

    if (dir.x >= vec3_type::value_type(0)) {
        tmin = (b_min.x - org.x) * rec_dir.x;
        tmax = (b_max.x - org.x) * rec_dir.x;
    }
    else {
        tmin = (b_max.x - org.x) * rec_dir.x;
        tmax = (b_min.x - org.x) * rec_dir.x;
    }

    if (dir.y >= vec3_type::value_type(0)) {
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

    if (dir.z >= vec3_type::value_type(0)) {
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
    
    return (tmin > vec3_type::value_type(0));
}

} // namespace gl
} // namespace scm
