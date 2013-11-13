
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm> // std::swap

#include <scm/core/math/common.h>

namespace std {

template<typename scal_type>
inline void swap(scm::math::quat<scal_type>& lhs,
                 scm::math::quat<scal_type>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

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

    return *this;
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

    return *this;
}

template<typename scal_type>
inline
quat<scal_type>&
quat<scal_type>::operator*=(const quat<value_type>& a)
{
    *this = *this * a;

    return *this;
}

template<typename scal_type>
template<typename rhs_t>
inline
quat<scal_type>&
quat<scal_type>::operator*=(const quat<rhs_t>& a)
{
    *this = *this * a;

    return *this;
}

template<typename scal_type>
inline
const mat<scal_type, 4, 4>
quat<scal_type>::to_matrix() const
{
    mat<scal_type, 4, 4> ret = mat<scal_type, 4, 4>::zero();

    ret.data_array[0 * 4 + 0] = scal_type(1) - scal_type(2) * y * y - scal_type(2) * z * z;
    ret.data_array[1 * 4 + 0] = scal_type(2) * x * y - scal_type(2) * w * z;
    ret.data_array[2 * 4 + 0] = scal_type(2) * x * z + scal_type(2) * w * y;

    ret.data_array[0 * 4 + 1] = scal_type(2) * x * y + scal_type(2) * w * z;
    ret.data_array[1 * 4 + 1] = scal_type(1) - scal_type(2) * x * x - scal_type(2) * z * z;
    ret.data_array[2 * 4 + 1] = scal_type(2) * y * z - scal_type(2) * w * x;
    
    ret.data_array[0 * 4 + 2] = scal_type(2) * x * z - scal_type(2) * w * y;
    ret.data_array[1 * 4 + 2] = scal_type(2) * y * z + scal_type(2) * w * x;
    ret.data_array[2 * 4 + 2] = scal_type(1) - scal_type(2) * x * x - scal_type(2) * y * y;

    ret.data_array[3 * 4 + 3] = scal_type(1);

    return ret;
}

template<typename scal_type>
inline
void
quat<scal_type>::retrieve_axis_angle(value_type& angle, vec<value_type, 3>& axis) const
{
    const value_type n = sqrt(i * i + j * j + k * k);

    if (n > value_type(0)) {
        axis.x = i / n;
        axis.y = j / n;
        axis.z = k / n;
        angle  = rad2deg(value_type(2.0) * acos(w));
    }
    else {
        axis.x = value_type(1.0);
        axis.y = value_type(0.0);
        axis.z = value_type(0.0);
        angle  = value_type(0.0);
    }
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::identity()
{
    return quat<value_type>(value_type(1), value_type(0), value_type(0), value_type(0));
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_arc(const vec<value_type, 3>& a, const vec<value_type, 3>& b)
{
    const vec<value_type, 3> na = normalize(a);
    const vec<value_type, 3> nb = normalize(b);

    quat<value_type>    ret = quat<value_type>(dot(na, nb) + value_type(1), cross(na, nb));

    return normalize(ret);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_axis(const value_type angle, const vec<value_type, 3>& axis)
{
    const value_type a = angle / value_type(2);
    const value_type s = sin(deg2rad(a)) / length(axis);
    const value_type c = cos(deg2rad(a));

    quat<value_type> ret;
        
    ret.w = c;
    ret.x = axis.x * s;
    ret.y = axis.y * s;
    ret.z = axis.z * s;

    return normalize(ret);
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
 
    quat<value_type>    ret;
    ret.x = sinr * cosp * cosy - cosr * sinp * siny;
    ret.y = cosr * sinp * cosy + sinr * cosp * siny;
    ret.z = cosr * cosp * siny - sinr * sinp * cosy;
    ret.w = cosr * cosp * cosy + sinr * sinp * siny;;

    return normalize(ret);
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_matrix(const mat<value_type, 3, 3>& m)
{
    const vec<scal_type, 3> ex = vec<scal_type, 3>(m.column(0));
    const vec<scal_type, 3> ey = vec<scal_type, 3>(m.column(1));
    const vec<scal_type, 3> ez = vec<scal_type, 3>(m.column(2));

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

    return ret;
}

template<typename scal_type>
inline
quat<scal_type>
quat<scal_type>::from_matrix(const mat<value_type, 4, 4>& m)
{
    const vec<scal_type, 3> ex = vec<scal_type, 3>(m.column(0));
    const vec<scal_type, 3> ey = vec<scal_type, 3>(m.column(1));
    const vec<scal_type, 3> ez = vec<scal_type, 3>(m.column(2));

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

    return ret;
}

// operators //////////////////////////////////////////////////////////////////////////////////////
template<typename scal_type>
inline
const quat<scal_type>
operator*(const quat<scal_type>& lhs, const quat<scal_type>& rhs)
{
    quat<scal_type> ret;

    ret.x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y;
    ret.y = lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z;
    ret.z = lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x;
    ret.w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z;

    return ret;
}

template<typename scal_l, typename scal_r>
inline
const quat<scal_l>
operator*(const quat<scal_l>& lhs, const quat<scal_r>& rhs)
{
    // no static cast here, so we get warned if we convert types
    quat<scal_l> ret;

    ret.x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y;
    ret.y = lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z;
    ret.z = lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x;
    ret.w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z;

    return ret;
}

template<typename scal_type>
inline
const vec<scal_type, 3>
operator*(const quat<scal_type>& lhs, const vec<scal_type, 3>& rhs)
{
    const quat<scal_type> vq(0.0, rhs);
    const quat<scal_type> res = lhs * vq * conjugate(lhs);

    return vec<scal_type, 3>(res.x, res.y, res.z);
}

template<typename scal_type>
inline
bool                               
operator==(const quat<scal_type>& lhs, const quat<scal_type>& rhs)
{
  return lhs.w == rhs.w &&
         lhs.i == rhs.i &&
         lhs.j == rhs.j &&
         lhs.i == rhs.k;
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
    quat<scal_type>    ret(lhs);

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
    return quat<scal_type>( lhs.w,
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
template<typename scal_type>
inline 
const quat<scal_type>
lerp(const quat<scal_type>& a, const quat<scal_type>& b, const scal_type u)
{
    quat<scal_type> ret;
        
    ret.w = (scal_type(1) - u) * a.w + u * b.w;
    ret.x = (scal_type(1) - u) * a.x + u * b.x;
    ret.y = (scal_type(1) - u) * a.y + u * b.y;
    ret.z = (scal_type(1) - u) * a.z + u * b.z;

    return normalize(ret);
}

template<typename scal_type>
inline 
const quat<scal_type>
slerp(const quat<scal_type>& a, const quat<scal_type>& b, const scal_type u)
{
    scal_type cos_theta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    scal_type sgn_theta = cos_theta < scal_type(0) ? scal_type(-1) : scal_type(1);

    cos_theta *= sgn_theta;

    scal_type alpha = u;
    scal_type beta  = scal_type(1) - u;

    if (scal_type(1) - cos_theta > scal_type(0)) {
        scal_type theta     = acos(cos_theta);
        scal_type sin_theta = sin(theta);
        
        beta  = sin(theta - u * theta) / sin_theta;
        alpha = sgn_theta * sin(u * theta) / sin_theta;
    }

    quat<scal_type> ret;

    ret.w = beta * a.w + alpha * b.w;
    ret.x = beta * a.x + alpha * b.x;
    ret.y = beta * a.y + alpha * b.y;
    ret.z = beta * a.z + alpha * b.z;

    return normalize(ret);
}

} // namespace math
} // namespace scm
