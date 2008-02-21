
#ifndef TRACKBALL_MANIPULATOR_H_INCLUDED
#define TRACKBALL_MANIPULATOR_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

// trackball manipulator class
class __scm_export(ogl) trackball_manipulator
{
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

    void apply_transform() const;

    const scm::math::mat4f  get_transform_matrix() const;
    float                   dolly() const;

private:
    float project_to_sphere(float x, float y) const;

    scm::math::mat4f    _matrix;
    float               _radius;
    float               _dolly;

}; // class trackball_manipulator

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TRACKBALL_MANIPULATOR_H_INCLUDED
