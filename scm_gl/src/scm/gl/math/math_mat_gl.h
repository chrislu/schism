
#ifndef SCM_MATH_MAT_GL_INCLUDED
#define SCM_MATH_MAT_GL_INCLUDED

#include <scm_math/math_mat.h>

namespace math
{
    template<typename scm_scalar>
    class mat_gl : public mat<scm_scalar, 4, 4>
    {
    public:
        mat_gl() {}
        mat_gl(const mat<scm_scalar, 4, 4>& m) : mat<scm_scalar, 4, 4>(m) {}
        explicit mat_gl(const scm_scalar s) : mat<scm_scalar, 4, 4>(s) {}
        explicit mat_gl(const scm_scalar a00,
                        const scm_scalar a01,
                        const scm_scalar a02,
                        const scm_scalar a03,
                        const scm_scalar a04,
                        const scm_scalar a05,
                        const scm_scalar a06,
                        const scm_scalar a07,
                        const scm_scalar a08,
                        const scm_scalar a09,
                        const scm_scalar a10,
                        const scm_scalar a11,
                        const scm_scalar a12,
                        const scm_scalar a13,
                        const scm_scalar a14,
                        const scm_scalar a15)  : mat<scm_scalar, 4, 4>(a00, a01, a02, a03,
                                                                       a04, a05, a06, a07,
                                                                       a08, a09, a10, a11,
                                                                       a12, a13, a14, a15) {}
        explicit mat_gl(const vec<scm_scalar, 4>& c00,
                        const vec<scm_scalar, 4>& c01,
                        const vec<scm_scalar, 4>& c02,
                        const vec<scm_scalar, 4>& c03) : mat<scm_scalar, 4, 4>(c00, c01, c02, c03) {}

        void    translate(const scm_scalar x,
                          const scm_scalar y,
                          const scm_scalar z);

        void    scale(const scm_scalar x,
                      const scm_scalar y,
                      const scm_scalar z);

        void    rotate(const scm_scalar angl,
                       const scm_scalar axis_x,
                       const scm_scalar axis_y,
                       const scm_scalar axis_z);

    protected:
    private:
    }; // class mat_gl

} // namespace math

#include "math_mat_gl.inl"

#endif // SCM_MATH_MAT_GL_INCLUDED
