
#ifndef TRACKBALL_MANIPULATOR_H_INCLUDED
#define TRACKBALL_MANIPULATOR_H_INCLUDED

#include <scm_core/math/math.h>
#include <scm_core/math/math_gl.h>

namespace gl
{
    // trackball manipulator class
    class trackball_manipulator
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

        const math::mat4x4f_t get_transform_matrix() const;

    private:
        float project_to_sphere(float x, float y) const;

        math::mat_glf_t _matrix;
        float           _radius;
        float           _dolly;
    };
} // namespace gl

#endif // TRACKBALL_MANIPULATOR_H_INCLUDED
