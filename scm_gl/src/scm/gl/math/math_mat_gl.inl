
#ifndef SCM_MATH_MAT_GL_INL_INCLUDED
#define SCM_MATH_MAT_GL_INL_INCLUDED

#include <scm_math/math_lib.h>
#include <scm_math/math_vec.h>
#include <scm_math/math_vec_lib.h>

namespace math
{
    template<typename scm_scalar>
    void mat_gl<scm_scalar>::translate(const scm_scalar x,
                                       const scm_scalar y,
                                       const scm_scalar z)
    {
#if 0
        mat_gl<scm_scalar> tmp_trans(1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, 0,
                                     x, y, z, 1);

        *this *= tmp_trans;
#else
        // more optimized
        m12 = m00 * x + m04 * y + m08 * z + m12;
        m13 = m01 * x + m05 * y + m09 * z + m13;
        m14 = m02 * x + m06 * y + m10 * z + m14;
        m15 = m03 * x + m07 * y + m11 * z + m15;
#endif
    }

    template<typename scm_scalar>
    void mat_gl<scm_scalar>::scale(const scm_scalar x,
                                   const scm_scalar y,
                                   const scm_scalar z)
    {
#if 0
        mat_gl<scm_scalar> tmp_scale(x, 0, 0, 0,
                                     0, y, 0, 0,
                                     0, 0, z, 0,
                                     0, 0, 0, 1);

        *this *= tmp_scale;
#else
        // more optimized
        m00 *= x;
        m01 *= x;
        m02 *= x;
        m03 *= x;

        m04 *= y;
        m05 *= y;
        m06 *= y;
        m07 *= y;

        m08 *= z;
        m09 *= z;
        m10 *= z;
        m11 *= z;
#endif
    }

    template<typename scm_scalar>
    void mat_gl<scm_scalar>::rotate(const scm_scalar angl,
                                    const scm_scalar axis_x,
                                    const scm_scalar axis_y,
                                    const scm_scalar axis_z)
    {
        // code after mesagl example

        vec<scm_scalar, 3> axis(axis_x, axis_y, axis_z);

        axis = math::normalize(axis);

        if (axis == vec<scm_scalar, 3>(scm_scalar(0))) {
            return;
        }

        mat_gl<scm_scalar> tmp_rot;
        scm_scalar         s = math::sin(math::deg2rad(angl));
        scm_scalar         c = math::cos(math::deg2rad(angl));
        scm_scalar         one_c = scm_scalar(1) - c;
        scm_scalar         xx, yy, zz, xy, yz, zx, xs, ys, zs;

        xx = axis.x * axis.x;
        yy = axis.y * axis.y;
        zz = axis.z * axis.z;
        xy = axis.x * axis.y;
        yz = axis.y * axis.z;
        zx = axis.z * axis.x;
        xs = axis.x * s;
        ys = axis.y * s;
        zs = axis.z * s;

        tmp_rot.m00 = (one_c * xx) + c;
        tmp_rot.m04 = (one_c * xy) - zs;
        tmp_rot.m08 = (one_c * zx) + ys;
        tmp_rot.m12 = scm_scalar(0);

        tmp_rot.m01 = (one_c * xy) + zs;
        tmp_rot.m05 = (one_c * yy) + c;
        tmp_rot.m09 = (one_c * yz) - xs;
        tmp_rot.m13 = scm_scalar(0);

        tmp_rot.m02 = (one_c * zx) - ys;
        tmp_rot.m06 = (one_c * yz) + xs;
        tmp_rot.m10 = (one_c * zz) + c;
        tmp_rot.m14 = scm_scalar(0);

        tmp_rot.m03 = scm_scalar(0);
        tmp_rot.m07 = scm_scalar(0);
        tmp_rot.m11 = scm_scalar(0);
        tmp_rot.m15 = scm_scalar(1);

        *this *= tmp_rot;
    }

} // namespace math

#endif // SCM_MATH_MAT_GL_INL_INCLUDED

#if 0
/**********************************************************************/
/** \name Matrix generation */
/*@{*/

/**
 * Generate a 4x4 transformation matrix from glRotate parameters, and
 * post-multiply the input matrix by it.
 *
 * \author
 * This function was contributed by Erich Boleyn (erich@uruk.org).
 * Optimizations contributed by Rudolf Opalla (rudi@khm.de).
 */
void
_math_matrix_rotate( GLmatrix *mat,
		     GLfloat angle, GLfloat x, GLfloat y, GLfloat z )
{
   GLfloat xx, yy, zz, xy, yz, zx, xs, ys, zs, one_c, s, c;
   GLfloat m[16];
   GLboolean optimized;

   s = (GLfloat) sin( angle * DEG2RAD );
   c = (GLfloat) cos( angle * DEG2RAD );

   MEMCPY(m, Identity, sizeof(GLfloat)*16);
   optimized = GL_FALSE;

#define M(row,col)  m[col*4+row]

   if (x == 0.0F) {
      if (y == 0.0F) {
         if (z != 0.0F) {
            optimized = GL_TRUE;
            /* rotate only around z-axis */
            M(0,0) = c;
            M(1,1) = c;
            if (z < 0.0F) {
               M(0,1) = s;
               M(1,0) = -s;
            }
            else {
               M(0,1) = -s;
               M(1,0) = s;
            }
         }
      }
      else if (z == 0.0F) {
         optimized = GL_TRUE;
         /* rotate only around y-axis */
         M(0,0) = c;
         M(2,2) = c;
         if (y < 0.0F) {
            M(0,2) = -s;
            M(2,0) = s;
         }
         else {
            M(0,2) = s;
            M(2,0) = -s;
         }
      }
   }
   else if (y == 0.0F) {
      if (z == 0.0F) {
         optimized = GL_TRUE;
         /* rotate only around x-axis */
         M(1,1) = c;
         M(2,2) = c;
         if (x < 0.0F) {
            M(1,2) = s;
            M(2,1) = -s;
         }
         else {
            M(1,2) = -s;
            M(2,1) = s;
         }
      }
   }

   if (!optimized) {
      const GLfloat mag = SQRTF(x * x + y * y + z * z);

      if (mag <= 1.0e-4) {
         /* no rotation, leave mat as-is */
         return;
      }

      x /= mag;
      y /= mag;
      z /= mag;


      /*
       *     Arbitrary axis rotation matrix.
       *
       *  This is composed of 5 matrices, Rz, Ry, T, Ry', Rz', multiplied
       *  like so:  Rz * Ry * T * Ry' * Rz'.  T is the final rotation
       *  (which is about the X-axis), and the two composite transforms
       *  Ry' * Rz' and Rz * Ry are (respectively) the rotations necessary
       *  from the arbitrary axis to the X-axis then back.  They are
       *  all elementary rotations.
       *
       *  Rz' is a rotation about the Z-axis, to bring the axis vector
       *  into the x-z plane.  Then Ry' is applied, rotating about the
       *  Y-axis to bring the axis vector parallel with the X-axis.  The
       *  rotation about the X-axis is then performed.  Ry and Rz are
       *  simply the respective inverse transforms to bring the arbitrary
       *  axis back to it's original orientation.  The first transforms
       *  Rz' and Ry' are considered inverses, since the data from the
       *  arbitrary axis gives you info on how to get to it, not how
       *  to get away from it, and an inverse must be applied.
       *
       *  The basic calculation used is to recognize that the arbitrary
       *  axis vector (x, y, z), since it is of unit length, actually
       *  represents the sines and cosines of the angles to rotate the
       *  X-axis to the same orientation, with theta being the angle about
       *  Z and phi the angle about Y (in the order described above)
       *  as follows:
       *
       *  cos ( theta ) = x / sqrt ( 1 - z^2 )
       *  sin ( theta ) = y / sqrt ( 1 - z^2 )
       *
       *  cos ( phi ) = sqrt ( 1 - z^2 )
       *  sin ( phi ) = z
       *
       *  Note that cos ( phi ) can further be inserted to the above
       *  formulas:
       *
       *  cos ( theta ) = x / cos ( phi )
       *  sin ( theta ) = y / sin ( phi )
       *
       *  ...etc.  Because of those relations and the standard trigonometric
       *  relations, it is pssible to reduce the transforms down to what
       *  is used below.  It may be that any primary axis chosen will give the
       *  same results (modulo a sign convention) using thie method.
       *
       *  Particularly nice is to notice that all divisions that might
       *  have caused trouble when parallel to certain planes or
       *  axis go away with care paid to reducing the expressions.
       *  After checking, it does perform correctly under all cases, since
       *  in all the cases of division where the denominator would have
       *  been zero, the numerator would have been zero as well, giving
       *  the expected result.
       */

      xx = x * x;
      yy = y * y;
      zz = z * z;
      xy = x * y;
      yz = y * z;
      zx = z * x;
      xs = x * s;
      ys = y * s;
      zs = z * s;
      one_c = 1.0F - c;

      /* We already hold the identity-matrix so we can skip some statements */
      M(0,0) = (one_c * xx) + c;
      M(0,1) = (one_c * xy) - zs;
      M(0,2) = (one_c * zx) + ys;
/*    M(0,3) = 0.0F; */

      M(1,0) = (one_c * xy) + zs;
      M(1,1) = (one_c * yy) + c;
      M(1,2) = (one_c * yz) - xs;
/*    M(1,3) = 0.0F; */

      M(2,0) = (one_c * zx) - ys;
      M(2,1) = (one_c * yz) + xs;
      M(2,2) = (one_c * zz) + c;
/*    M(2,3) = 0.0F; */

/*
      M(3,0) = 0.0F;
      M(3,1) = 0.0F;
      M(3,2) = 0.0F;
      M(3,3) = 1.0F;
*/
   }
#undef M

   matrix_multf( mat, m, MAT_FLAG_ROTATION );
}

#endif
