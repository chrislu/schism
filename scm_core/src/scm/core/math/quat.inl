
#include <algorithm> // std::swap

#include <scm/core/math/common.h>

namespace scm {
namespace math {

template<typename scal_type>
inline
quat<scal_type>::quat()
{
}

template<typename scal_type>
inline
quat<scal_type>::quat(const value_type r, const value_type ii, const value_type ij, const value_type ik)
  : w(r)
  , i(ii)
  , j(ij)
  , k(ik)
{
}

template<typename scal_type>
inline
quat<scal_type>::quat(const value_type r, const vec<value_type, 3>& img)
  : w(r)
  , i(img.x)
  , j(img.y)
  , k(img.z)
{
}

template<typename scal_type>
inline
quat<scal_type>::quat(const quat<value_type>& rhs)
  : w(rhs.w)
  , i(rhs.i)
  , j(rhs.j)
  , k(rhs.k)
{
}
    
template<typename scal_type>
template<typename rhs_t>
inline
quat<scal_type>::quat(const quat<rhs_t>& rhs)
  : w(static_cast<value_type>(rhs.w))
  , i(static_cast<value_type>(rhs.i))
  , j(static_cast<value_type>(rhs.j))
  , k(static_cast<value_type>(rhs.k))
{
}

template<typename scal_type>
inline
void
quat<scal_type>::swap(quat<value_type>& rhs)
{
    std::swap(w, rhs.w);
    std::swap(i, rhs.i);
    std::swap(j, rhs.j);
    std::swap(k, rhs.k);
}

template<typename scal_type>
inline
quat<scal_type>&
quat<scal_type>::operator=(const quat<value_type>& rhs)
{
    w = rhs.w;
    i = rhs.i;
    j = rhs.j;
    k = rhs.k;

    return (*this);
}

template<typename scal_type>
template<typename rhs_t>
inline
quat<scal_type>&
quat<scal_type>::operator=(const quat<rhs_t>& rhs)
{
    // no static cast here, so we get warned if we convert types
    w = rhs.w;
    i = rhs.i;
    j = rhs.j;
    k = rhs.k;

    return (*this);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_axis(const value_type angle, const vec<value_type, 3>& axis)
{
    const value_type a = angle / value_type(2);
    const value_type s = sin(deg2rad(a)) / length(axis);
    const value_type c = cos(deg2rad(a));

    return quat<value_type>(axis.x * s,
                            axis.y * s,
                            axis.z * s,
                            c);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_euler(const value_type pitch, const value_type yaw, const value_type roll)
{
    // Basically we create 3 Quaternions, one for pitch, one for yaw, one for roll
    // and multiply those together.
    // the calculation below does the same, just shorter
 
    value_type p = deg2rad(pitch) / value_type(2);
    value_type y = deg2rad(yaw)   / value_type(2);
    value_type r = deg2rad(roll)  / value_type(2);
 
    value_type sinp = sin(p);
    value_type siny = sin(y);
    value_type sinr = sin(r);
    value_type cosp = cos(p);
    value_type cosy = cos(y);
    value_type cosr = cos(r);
 
    quat<value_type>    ret =
        quat<value_type>(sinr * cosp * cosy - cosr * sinp * siny,
                         cosr * sinp * cosy + sinr * cosp * siny,
                         cosr * cosp * siny - sinr * sinp * cosy,
                         cosr * cosp * cosy + sinr * sinp * siny);

    return normalize(ret);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_matrix(const mat<value_type, 3, 3>& m)
{
    const vec<scal_type, 3> ex = vec<scal_type, 3>(m.m00, m.m03, m.m06);
    const vec<scal_type, 3> ey = vec<scal_type, 3>(m.m01, m.m04, m.m07);
    const vec<scal_type, 3> ez = vec<scal_type, 3>(m.m02, m.m05, m.m08);

    quat<value_type>    ret;

    const value_type trace = ex.x + ey.y + ez.z;
    value_type       scale;
    if(trace > value_type(0)) {
        scale   = sqrt(value_type(1) + trace);
        ret.w   = scale / value_type(2);

        scale   = value_type(0.5) / scale;
        ret.x   = (ey.z - ez.y) * scale;
        ret.y   = (ez.x - ex.z) * scale;
        ret.z   = (ex.y - ey.x) * scale;
    }
    else if(ex.x > ey.y && ex.x > ez.z) {
        scale   = value_type(2) * sqrt(value_type(1) + ex.x - ey.y - ez.z);
        ret.x   = scale / value_type(4);
        ret.y   = (ex.y + ey.x) / scale;
        ret.z   = (ex.z + ez.x) / scale;
        ret.w   = (ey.z - ez.y) / scale;
    }
    else if(ey.y > ez.z) {
        scale   = value_type(2) * sqrt(value_type(1) + ey.y - ex.x - ez.z);
        ret.x   = (ex.y + ey.x) / scale;
        ret.y   = scale / value_type(4);
        ret.z   = (ey.z + ez.y) / scale;
        ret.w   = (ez.x - ex.z) / scale;
    }
    else {
        scale   = value_type(2) * sqrt(value_type(1) + ez.z - ex.x - ey.y);
        ret.x   = (ex.z + ez.x) / scale;
        ret.y   = (ey.z + ez.y) / scale;
        ret.z   = scale / value_type(4);
        ret.w   = (ex.y - ey.x) / scale;
    }

    return (ret);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_matrix(const mat<value_type, 4, 4>& m)
{
    const vec<scal_type, 3> ex = vec<scal_type, 3>(m.m00, m.m04, m.m08);
    const vec<scal_type, 3> ey = vec<scal_type, 3>(m.m01, m.m05, m.m09);
    const vec<scal_type, 3> ez = vec<scal_type, 3>(m.m02, m.m06, m.m10);

    quat<value_type>    ret;

    const value_type trace = ex.x + ey.y + ez.z;
    value_type       scale;
    if(trace > value_type(0)) {
        scale   = sqrt(value_type(1) + trace);
        ret.w   = scale / value_type(2);

        scale   = value_type(0.5) / scale;
        ret.x   = (ey.z - ez.y) * scale;
        ret.y   = (ez.x - ex.z) * scale;
        ret.z   = (ex.y - ey.x) * scale;
    }
    else if(ex.x > ey.y && ex.x > ez.z) {
        scale   = value_type(2) * sqrt(value_type(1) + ex.x - ey.y - ez.z);
        ret.x   = scale / value_type(4);
        ret.y   = (ex.y + ey.x) / scale;
        ret.z   = (ex.z + ez.x) / scale;
        ret.w   = (ey.z - ez.y) / scale;
    }
    else if(ey.y > ez.z) {
        scale   = value_type(2) * sqrt(value_type(1) + ey.y - ex.x - ez.z);
        ret.x   = (ex.y + ey.x) / scale;
        ret.y   = scale / value_type(4);
        ret.z   = (ey.z + ez.y) / scale;
        ret.w   = (ez.x - ex.z) / scale;
    }
    else {
        scale   = value_type(2) * sqrt(value_type(1) + ez.z - ex.x - ey.y);
        ret.x   = (ex.z + ez.x) / scale;
        ret.y   = (ey.z + ez.y) / scale;
        ret.z   = scale / value_type(4);
        ret.w   = (ex.y - ey.x) / scale;
    }

    return (ret);
}

// library functions //////////////////////////////////////////////////////////////////////////////
template<typename scal_type>
inline 
const quat<scal_type>
normalize(const quat<scal_type>& lhs)
{
    scal_type    t =   lhs.w * lhs.w
                     + lhs.i * lhs.i
                     + lhs.j * lhs.j
                     + lhs.k * lhs.k;
    scal_type    f;
    quat<value_type>    ret(lhs);

    if (t > scal_type(0)) {
        f = scal_type(1) / sqrt(t);
        ret.w *= f;
        ret.i *= f;
        ret.j *= f;
        ret.k *= f;
    }

    return ret;
}

template<typename scal_type>
inline 
const quat<scal_type>
conjugate(const quat<scal_type>& lhs)
{
    return quat<value_type>( lhs.w,
                            -lhs.i,
                            -lhs.j,
                            -lhs.k);
}

template<typename scal_type>
inline 
const scal_type
magnitude(const quat<scal_type>& lhs)
{
    return sqrt(  lhs.w * lhs.w
                + lhs.i * lhs.i
                + lhs.j * lhs.j
                + lhs.k * lhs.k);
}

} // namespace math
} // namespace scm
