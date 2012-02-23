
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TRACKBALL_MANIPULATOR_H_INCLUDED
#define SCM_GL_UTIL_TRACKBALL_MANIPULATOR_H_INCLUDED

#include <scm/core/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

// trackball manipulator class
class __scm_export(gl_util) trackball_manipulator
{
public:
    typedef math::vec2f             vec2_type;
    typedef math::vec3f             vec3_type;
    typedef math::vec4f             vec4_type;
    typedef math::mat4f             mat4_type;
    typedef mat4_type::value_type   value_type;

public:
    trackball_manipulator();
    virtual ~trackball_manipulator();

    // all transform functions with the window bounds [-1..1,-1..1],
    // so the window center is assumed at [0,0]
    // x-axis in left to right direction
    // y-axis in bottom to up direction
    void rotation(float fx, float fy, float tx, float ty);
    void translation(float x, float y);
    void dolly(float y);


    const mat4_type&    transform_matrix() const;
    void                transform_matrix(const mat4_type& m);
    float               dolly() const;

private:
    float project_to_sphere(float x, float y) const;

    mat4_type           _matrix;
    float               _radius;
    float               _dolly;

}; // class trackball_manipulator

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TRACKBALL_MANIPULATOR_H_INCLUDED
