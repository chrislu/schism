
#include <scm/core/math/common.h>
#include <scm/ogl/gl.h>

namespace scm {
namespace math {

template<typename scal_type>
inline void translate(mat<scal_type, 4, 4>&    m,
                      const vec<scal_type, 3>& t)
{
    m.m12 = m.m00 * t.x + m.m04 * t.y + m.m08 * t.z + m.m12;
    m.m13 = m.m01 * t.x + m.m05 * t.y + m.m09 * t.z + m.m13;
    m.m14 = m.m02 * t.x + m.m06 * t.y + m.m10 * t.z + m.m14;
    m.m15 = m.m03 * t.x + m.m07 * t.y + m.m11 * t.z + m.m15;
}

template<typename scal_type>
inline void translate(mat<scal_type, 4, 4>&    m,
                      const scal_type          x,
                      const scal_type          y,
                      const scal_type          z)
{
    translate(m, vec<scal_type, 3>(x, y, z));
}

template<typename scal_type>
inline void rotate(mat<scal_type, 4, 4>&       m,
                   const scal_type             angl,
                   const vec<scal_type, 3>&    axis)
{
    // code after mesagl example
    if (scm::math::length(axis) == scal_type(0)) {
        return;
    }

    mat<scal_type, 4, 4>    tmp_rot;
    scal_type               s = scm::math::sin(scm::math::deg2rad(angl));
    scal_type               c = scm::math::cos(scm::math::deg2rad(angl));
    scal_type               one_c = scal_type(1) - c;
    scal_type               xx, yy, zz, xy, yz, zx, xs, ys, zs;

    math::vec<scal_type, 3> axis_n = scm::math::normalize(axis);

    xx = axis_n.x * axis_n.x;
    yy = axis_n.y * axis_n.y;
    zz = axis_n.z * axis_n.z;
    xy = axis_n.x * axis_n.y;
    yz = axis_n.y * axis_n.z;
    zx = axis_n.z * axis_n.x;
    xs = axis_n.x * s;
    ys = axis_n.y * s;
    zs = axis_n.z * s;

    tmp_rot.m00 = (one_c * xx) + c;
    tmp_rot.m04 = (one_c * xy) - zs;
    tmp_rot.m08 = (one_c * zx) + ys;
    tmp_rot.m12 = scal_type(0);

    tmp_rot.m01 = (one_c * xy) + zs;
    tmp_rot.m05 = (one_c * yy) + c;
    tmp_rot.m09 = (one_c * yz) - xs;
    tmp_rot.m13 = scal_type(0);

    tmp_rot.m02 = (one_c * zx) - ys;
    tmp_rot.m06 = (one_c * yz) + xs;
    tmp_rot.m10 = (one_c * zz) + c;
    tmp_rot.m14 = scal_type(0);

    tmp_rot.m03 = scal_type(0);
    tmp_rot.m07 = scal_type(0);
    tmp_rot.m11 = scal_type(0);
    tmp_rot.m15 = scal_type(1);

    m *= tmp_rot;
}

template<typename scal_type>
inline void rotate(mat<scal_type, 4, 4>&       m,
                   const scal_type             angl,
                   const scal_type             axis_x,
                   const scal_type             axis_y,
                   const scal_type             axis_z)
{
    rotate(m, angl, vec<scal_type, 3>(axis_x, axis_y, axis_z));
}

template<typename scal_type>
inline void scale(mat<scal_type, 4, 4>&        m,
                  const vec<scal_type, 3>&     s)
{
    m.m00 *= s.x;
    m.m05 *= s.y;
    m.m10 *= s.z;
}

template<typename scal_type>
inline void scale(mat<scal_type, 4, 4>&        m,
                  const scal_type              x,
                  const scal_type              y,
                  const scal_type              z)
{
    scale(m, vec<scal_type, 3>(x, y, z));
}

inline void get_gl_matrix(const int type, mat<float, 4, 4>& m)
{
    glGetFloatv(type, m.data_array);
}

inline void get_gl_matrix(const int type, mat<double, 4, 4>& m)
{
    glGetDoublev(type, m.data_array);
}

// TBD
#if 0
inline mat4_t gl_ortho_matrix(  const float left,
                                const float right,
                                const float bottom,
                                const float top,
                                const float near_z,
                                const float far_z)
{
    float A,B,C,D,E,F;

    A=-(right+left)/(right-left);
    B=-(top+bottom)/(top-bottom);
    C=-(far_z+near_z)/(far_z-near_z);

    D=-2.0f/(far_z-near_z);
    E=2.0f/(top-bottom);
    F=2.0f/(right-left);

    return (mat4_t( F, 0, 0, 0,
                    0, E, 0, 0,
                    0, 0, D, 0,
                    A, B, C, 1));
}

inline mat4_t gl_frustum_matrix(const float left,
                                const float right,
                                const float bottom,
                                const float top,
                                const float near_z,
                                const float far_z)
{
    float A,B,C,D,E,F;

    A=(right+left)/(right-left);
    B=(top+bottom)/(top-bottom);
    C=-(far_z+near_z)/(far_z-near_z);
    D=-(2.0f*far_z*near_z)/(far_z-near_z);
    E=2.0f*near_z/(top-bottom);
    F=2.0f*near_z/(right-left);

    return (mat4_t(	F, 0, 0, 0,
                    0, E, 0, 0,
                    A, B, C,-1,
                    0, 0, D, 0));
}

inline mat4_t gl_perspective_matrix( const float fovy   = 45,      // = 45, 
                                     const float aspect = 4.0f/3.0f,
                                     const float near_z = 0.1f,
                                     const float far_z  = 100)
{
    float maxy = tanf( deg2rad(fovy*.5f)) * near_z;
    float maxx = maxy*aspect;

    return (gl_frustum_matrix(-maxx, maxx, -maxy, maxy, near_z, far_z));
}
#endif


} // namespace math
} // namespace scm
