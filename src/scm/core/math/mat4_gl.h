
#ifndef MATH_MAT4_GL_H_INCLUDED
#define MATH_MAT4_GL_H_INCLUDED

#include <scm/core/math/mat.h>

namespace scm {
namespace math {

template<typename scal_type>
void translate(mat<scal_type, 4, 4>&    m,
               const vec<scal_type, 3>& t);

template<typename scal_type>
void translate(mat<scal_type, 4, 4>&    m,
               const scal_type          x,
               const scal_type          y,
               const scal_type          z);

template<typename scal_type>
void rotate(mat<scal_type, 4, 4>&       m,
            const scal_type             angl,
            const vec<scal_type, 3>&    axis);

template<typename scal_type>
void rotate(mat<scal_type, 4, 4>&       m,
            const scal_type             angl,
            const scal_type             axis_x,
            const scal_type             axis_y,
            const scal_type             axis_z);

template<typename scal_type>
void scale(mat<scal_type, 4, 4>&        m,
           const vec<scal_type, 3>&     s);

template<typename scal_type>
void scale(mat<scal_type, 4, 4>&        m,
           const scal_type              x,
           const scal_type              y,
           const scal_type              z);

void get_gl_matrix(const int type, mat<float, 4, 4>& m);
void get_gl_matrix(const int type, mat<double, 4, 4>& m);

} // namespace math
} // namespace scm

#include "mat4_gl.inl"

#endif // MATH_MAT4_GL_H_INCLUDED
