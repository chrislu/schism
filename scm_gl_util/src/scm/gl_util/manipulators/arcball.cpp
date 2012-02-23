
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "arcball.h"

#include <iostream>

#include <scm/gl_core/math/math.h>

namespace scm {
namespace gl {

arc_ball::arc_ball()
  : _radius(value_type(1.0)),
    _center(vec2_type(value_type(0.0))),
    _viewport_size(vec2_type(value_type(1.0))),
    _start(vec3_type(value_type(0))),
    _dragging(false),
    _translation_scale(vec2_type(value_type(1.0))),
    _aspect_ratio(value_type(1.0)),
    _mode(none),
    _scope(local)
{
    reset();
}

arc_ball::~arc_ball()
{
}

void
arc_ball::reset()
{
    _matrix             = mat4_type::identity();
    _rotation_matrix    = mat4_type::identity();

    // local
    _eye_to_obj_matrix  = mat4_type::identity();
    _translation_vector = vec3_type(value_type(0));
    _center_of_rotation = vec3_type(value_type(0));

    // global
    _distance           = value_type(0);
}

void
arc_ball::reset(const mat4_type& t,
                arc_ball::scope  s,
                const vec3_type& c,
                const mat4_type& eye_to_obj)
{
    _scope = s;

    if (_scope == local) {
        _matrix              = t;
        
        _rotation_matrix     = t;
        _rotation_matrix[12] = value_type(0.0);
        _rotation_matrix[13] = value_type(0.0);
        _rotation_matrix[14] = value_type(0.0);

        _translation_vector  = t.column(3);

        _center_of_rotation  = vec3_type(_rotation_matrix * vec4_type(c, value_type(1.0)));
        _eye_to_obj_matrix = eye_to_obj;
    }
    else if (_scope == global) {
        using namespace scm::math;
        using namespace scm::gl;

        vec3_type   eye     = t.column(3);
        value_type  dist    = length(eye);
        vec3_type   center  = eye - dist * vec3_type(t.column(2));
        vec3_type   up      = t.column(1);

        look_at_matrix_inv(_rotation_matrix, center, center + (center - eye), up);
        _distance           = length(center - eye);
    }
}

void
arc_ball::start_drag(arc_ball::mode m, value_type x, value_type y)
{
    using namespace scm::math;

    _start       = viewport_normalize(x, y);
    _dragging    = true;
    _mode        = m;
}

void
arc_ball::drag(value_type x, value_type y)
{
    if (_dragging && _mode != none) {
        using namespace scm::math;

        vec2_type end  = viewport_normalize(x, y);

        if (_mode == rotation) {
            vec2_type sn      = _start - _center;
            vec2_type en      = end    - _center;
            vec3_type start_a = normalize(vec3_type(sn, project_to_sphere(sn)));
            vec3_type end_a   = normalize(vec3_type(en, project_to_sphere(en)));

            // get rotation axis as cross product of the two vecs
            vec3_type   rot_axis(cross(start_a, end_a));

            // calculate rotaion angle
            value_type  rot_angl = acos(dot(start_a, end_a));

            if (_scope == local) {
                // bring rotation axis to the obj_space
                rot_axis = vec3_type(_eye_to_obj_matrix * vec4_type(rot_axis, value_type(0.0)));

                // rotate around center
                mat4_type r = mat4_type::identity();
                translate(r, _center_of_rotation);
                rotate(r, rad2deg(rot_angl), rot_axis);
                translate(r, -_center_of_rotation);
                _rotation_matrix = r * _rotation_matrix;
            }
            else if (_scope == global) {
                rotate(_rotation_matrix, -rad2deg(rot_angl), rot_axis);
            }
        }
        else if (_mode == panning) {
            vec2_type v     = (end - _start) * _translation_scale;

            if (_scope == local) {
                vec3_type t_obj = vec3_type(_eye_to_obj_matrix * vec4_type(v));
                _translation_vector += t_obj;
            }
            else if (_scope == global) {
                translate(_rotation_matrix, -vec3_type(v));
            }
        }
        else if (_mode == zooming) {
            value_type v  = end.y - _start.y;

            if (_scope == local) {
                vec3_type t_obj = vec3_type(_eye_to_obj_matrix * vec4_type(value_type(0.0), value_type(0.0), v, value_type(0.0)));
                _translation_vector -= t_obj;
            }
            else if (_scope == global) {
                _distance += v;
            }
        }

        if (_scope == local) {
            mat4_type t = mat4_type::identity();
            translate(t, _translation_vector);
            _matrix = t * _rotation_matrix;
        }
        else if (_scope == global) {
            _matrix = _rotation_matrix;
            translate(_matrix, vec3_type(value_type(0.0), value_type(0.0), _distance));
        }

        _start = end;
    }
}

void
arc_ball::end_drag()
{
    _dragging = false;
    _mode     = none;
}

void
arc_ball::viewport_size(const vec2_type& vp)
{
    _viewport_size = vp;
    _aspect_ratio  = _viewport_size.x / _viewport_size.y;
}

void
arc_ball::center(const vec2_type& c)
{
    _center = c;
}

void
arc_ball::radius(const value_type r)
{
    _radius = r;
}

void
arc_ball::translation_scale(const value_type r)
{
    _translation_scale = vec2_type(r, r / _aspect_ratio);
}

void
arc_ball::translation_scale(const vec2_type& r)
{
    _translation_scale = r;
}

const arc_ball::mat4_type&
arc_ball::matrix() const
{
    return (_matrix);
}

arc_ball::value_type
arc_ball::project_to_sphere(const vec2_type& p) const
{
    using namespace scm::math;

    value_type len_sqr = math::length_sqr(p);;
    value_type len     = math::sqrt(len_sqr);

    // if point lies inside sphere map it to the sphere, if it lies 
    // outside map it to hyperbola (at radius/sqrt(2) sphere and
    // hyperbola intersect and so this is the decission point)
    if (len < _radius / math::sqrt(value_type(2))) {
        return (math::sqrt(_radius * _radius - len_sqr));
    } else {
        // hyperbola z = r²/(2*d)
        return ((_radius*_radius) / (value_type(2) * len));
    }
}

arc_ball::vec2_type
arc_ball::viewport_normalize(value_type x, value_type y) const
{
    return (vec2_type(value_type(2) * (                   x - _viewport_size.x * value_type(0.5)) / _viewport_size.x,
                      value_type(2) * (_viewport_size.y - y - _viewport_size.y * value_type(0.5)) / _viewport_size.y));
}


} // namespace gl
} // namespace scm
