
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "camera.h"

#include <scm/gl_core/math.h>
#include <scm/gl_core/primitives/ray.h>

namespace scm {
namespace gl {

camera::camera()
  : _projection_matrix(math::mat4f::identity()),
    _view_matrix(math::mat4f::identity()),
    _view_projection_matrix(math::mat4f::identity()),
    _field_of_view(0.0f),
    _aspect_ratio(1.0f),
    _near_plane(0.1f),
    _far_plane(10.0f)
{
    projection_ortho_2d(0.0f, 1.0f, 0.0f, 1.0f);
}

camera::~camera()
{
}

void
camera::projection_perspective(float fovy, float aspect, float near_z, float far_z)
{
    _field_of_view = fovy;
    _aspect_ratio  = aspect;
    _near_plane    = near_z;
    _far_plane     = far_z;
    _type          = camera::perspective;
    _projection_matrix = math::make_perspective_matrix(fovy, aspect, near_z, far_z);
    update();
}

void
camera::projection_ortho(float left, float right, float bottom, float top, float near_z, float far_z)
{
    _field_of_view = 0.0f;
    _aspect_ratio  = (right - left) / (top - bottom);
    _near_plane    = near_z;
    _far_plane     = far_z;
    _type          = camera::ortho;
    _projection_matrix = math::make_ortho_matrix(left, right, bottom, top, near_z, far_z);
    update();
}

void
camera::projection_ortho_2d(float left, float right, float bottom, float top)
{
    _field_of_view = 0.0f;
    _aspect_ratio  = (right - left) / (top - bottom);
    _near_plane    = -1.0f;
    _far_plane     = 1.0f;
    _type          = camera::ortho;
    _projection_matrix = math::make_ortho_matrix(left, right, bottom, top, _near_plane, _far_plane);
    update();
}

void
camera::projection_frustum(float left, float right, float bottom, float top, float near_z, float far_z)
{
    float a = math::atan(bottom / near_z);
    float b = math::atan(top / near_z);
    _field_of_view = math::rad2deg(a) + math::rad2deg(b);
    _aspect_ratio  = (right - left) / (top - bottom);
    _near_plane    = near_z;
    _far_plane     = far_z;
    _type          = camera::perspective;
    _projection_matrix = math::make_frustum_matrix(left, right, bottom, top, _near_plane, _far_plane);
    update();
}

void
camera::view_matrix(const math::mat4f& v)
{
    _view_matrix = v;
    update();
}

const math::mat4f&
camera::projection_matrix() const
{
    return (_projection_matrix);
}

const math::mat4f&
camera::projection_matrix_inverse() const
{
    return (_projection_matrix_inverse);
}

const math::mat4f&
camera::view_matrix() const
{
    return (_view_matrix);
}

const math::mat4f&
camera::view_matrix_inverse() const
{
    return (_view_matrix_inverse);
}

const math::mat4f&
camera::view_matrix_inverse_transpose() const
{
    return (_view_matrix_inverse_transpose);
}

const math::mat4f&
camera::view_projection_matrix() const
{
    return (_view_projection_matrix);
}

const math::mat4f&
camera::view_projection_matrix_inverse() const
{
    return (_view_projection_matrix_inverse);
}

const math::vec4f
camera::position() const
{
    return (_view_matrix_inverse.column(3) / _view_matrix_inverse.column(3).w);
}

const frustumf&
camera::view_frustum() const
{
    return (_view_frustum);
}

float
camera::field_of_view() const
{
    return (_field_of_view);
}

void
camera::update()
{
    using namespace scm::math;

    //_projection_matrix;
    _projection_matrix_inverse      = inverse(_projection_matrix);

    // _view_matrix;
    _view_matrix_inverse            = inverse(_view_matrix);
    _view_matrix_inverse_transpose  = transpose(_view_matrix_inverse);

    _view_projection_matrix         = _projection_matrix * _view_matrix;
    _view_projection_matrix_inverse = inverse(_view_projection_matrix);

    _view_frustum.update(_view_projection_matrix);
}

float
camera::aspect_ratio() const
{
    return (_aspect_ratio);
}

float
camera::near_plane() const
{
    return (_near_plane);
}

float
camera::far_plane() const
{
    return (_far_plane);
}

ray
camera::generate_ray(const math::vec2f& nrm_coord) const
{
    using namespace scm::gl;
    using namespace scm::math;

    ray     pick_ray;

    if (type() == ortho) {
        vec4f p = view_projection_matrix_inverse() * vec4f(nrm_coord.x, nrm_coord.y, -1.0f, 1.0f);
        p /= p.w;

        vec4f o = p;
        o.z = -1.5f;
        o.w =  0.0f; 

        pick_ray = (ray(o, p - o));
    }
    else {
        vec4f p = inverse(_view_projection_matrix) * vec4f(nrm_coord.x, nrm_coord.y, -1.0f, 1.0f);
        p /= p.w;

        vec4f o = inverse(_view_matrix).column(3);

        pick_ray = ray(o, p - o);
    }

    return (pick_ray);
}

camera::projection_type
camera::type() const
{
    return (_type);
}

} // namespace gl
} // namespace scm
