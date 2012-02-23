
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_CAMERA_H_INCLUDED
#define SCM_GL_UTIL_CAMERA_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/frame_buffer_objects/frame_buffer_objects_fwd.h>
#include <scm/gl_core/primitives/frustum.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) camera
{
public:
    enum projection_type {
        perspective = 0x01,
        ortho,
        asymmetric
    }; // enum type

public:
    camera();
    virtual ~camera();

    void                    projection_perspective(float fovy, float aspect, float near_z, float far_z);
    void                    projection_ortho(float left, float right, float bottom, float top, float near_z, float far_z);
    void                    projection_ortho_2d(float left, float right, float bottom, float top);
    void                    projection_frustum(float left, float right, float bottom, float top, float near_z, float far_z);

    void                    view_matrix(const math::mat4f& v);

    const math::mat4f&      projection_matrix() const;
    const math::mat4f&      projection_matrix_inverse() const;
    const math::mat4f&      view_matrix() const;
    const math::mat4f&      view_matrix_inverse() const;
    const math::mat4f&      view_matrix_inverse_transpose() const;
    const math::mat4f&      view_projection_matrix() const;
    const math::mat4f&      view_projection_matrix_inverse() const;

    const math::vec4f       position() const;

    const frustumf&         view_frustum() const;

    float                   field_of_view() const;
    float                   aspect_ratio() const;
    float                   near_plane() const;
    float                   far_plane() const;

    ray                     generate_ray(const math::vec2f& nrm_coord) const;

    projection_type         type() const;

protected:
    void                    update();

protected:
    float                   _field_of_view;
    float                   _aspect_ratio;
    float                   _near_plane;
    float                   _far_plane;

    frustumf                _view_frustum; // world space

    projection_type         _type;

    math::mat4f             _projection_matrix;
    math::mat4f             _projection_matrix_inverse;

    math::mat4f             _view_matrix; // world to camera/eye space
    math::mat4f             _view_matrix_inverse;
    math::mat4f             _view_matrix_inverse_transpose;

    math::mat4f             _view_projection_matrix;
    math::mat4f             _view_projection_matrix_inverse;

}; // class camera



} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_CAMERA_H_INCLUDED
