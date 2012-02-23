
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_MANIP_ARCBALL_H_INCLUDED
#define SCM_GL_UTIL_MANIP_ARCBALL_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) arc_ball
{
public:
    typedef math::vec2f             vec2_type;
    typedef math::vec3f             vec3_type;
    typedef math::vec4f             vec4_type;
    typedef math::mat4f             mat4_type;
    typedef mat4_type::value_type   value_type;

    typedef enum {
        none        = 0x01,
        rotation,
        panning,
        zooming
    } mode;

    typedef enum {
        global,
        local
    } scope;

public:
    arc_ball();
    virtual ~arc_ball();

    void                reset();
    void                reset(const mat4_type& t,
                              scope s = local,
                              const vec3_type& c = vec3_type(value_type(0.0)),
                              const mat4_type& eye_to_obj = mat4_type::identity());

    void                start_drag(mode m, value_type x, value_type y);
    void                drag(value_type x, value_type y);
    void                end_drag();

    void                viewport_size(const vec2_type& vp);
    void                center(const vec2_type& c);
    void                radius(const value_type r);
    void                translation_scale(const value_type r);
    void                translation_scale(const vec2_type& r);

    const mat4_type&    matrix() const;

private:
    value_type          project_to_sphere(const vec2_type& p) const;
    vec2_type           viewport_normalize(value_type x, value_type y) const;

    mat4_type           _eye_to_obj_matrix;

    mat4_type           _matrix;
    mat4_type           _rotation_matrix;

    // local mode
    vec3_type           _translation_vector;
    vec3_type           _center_of_rotation;

    // global mode
    value_type          _distance;

    vec2_type           _viewport_size;
    vec2_type           _center;
    value_type          _radius;
    vec2_type           _translation_scale;
    value_type          _aspect_ratio;

    bool                _dragging;
    vec2_type           _start;
    mode                _mode;
    scope               _scope;

}; // class arc_ball

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_MANIP_ARCBALL_H_INCLUDED
