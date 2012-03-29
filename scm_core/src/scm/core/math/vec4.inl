
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

#include <scm/core/math/common.h>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline
vec<scal_type, 4>::vec()
{
}

template<typename scal_type>
inline
vec<scal_type, 4>::vec(const vec<scal_type, 4>& v)
  : x(v.x), y(v.y), z(v.z), w(v.w)
{
//    std::copy(v.data_array, v.data_array + 4, data_array);
}

//template<typename scal_type>
//inline vec<scal_type, 4>::vec(const scal_type a[4])
//{
//    std::copy(a, a + 4, data_array);
//}

template<typename scal_type>
inline
vec<scal_type, 4>::vec(const vec<scal_type, 2>& v,
                       const scal_type          z,
                       const scal_type          w)
  : x(v.x), y(v.y), z(z), w(w)
{
}

template<typename scal_type>
inline
vec<scal_type, 4>::vec(const vec<scal_type, 3>& v,
                       const scal_type          w)
  : x(v.x), y(v.y), z(v.z), w(w)
{
}

template<typename scal_type>
inline
vec<scal_type, 4>::vec(const scal_type s)
  : x(s), y(s), z(s), w(s) 
{
}

template<typename scal_type>
inline
vec<scal_type, 4>::vec(const scal_type _x,
                       const scal_type _y,
                       const scal_type _z,
                       const scal_type _w)
  : x(_x), y(_y), z(_z), w(_w)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>::vec(const vec<rhs_scal_t, 4>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y)),
    z(static_cast<scal_type>(v.z)),
    w(static_cast<scal_type>(v.w))
{
}

// dtor
//template<typename scal_type>
//inline vec<scal_type, 4>::~vec()
//{
//}

// constants
template<typename scal_type>
inline
const vec<scal_type, 4>&
vec<scal_type, 4>::zero()
{
    static vec<scal_type, 4> zero_(scal_type(0), scal_type(0), scal_type(0), scal_type(0));
    return (zero_);
}

template<typename scal_type>
inline
const vec<scal_type, 4>&
vec<scal_type, 4>::one()
{
    static vec<scal_type, 4> one_(scal_type(1), scal_type(1), scal_type(1), scal_type(1));
    return (one_);
}

// swap
template<typename scal_type>
inline
void vec<scal_type, 4>::swap(vec<scal_type, 4>& rhs)
{
    std::swap_ranges(data_array, data_array + 4, rhs.data_array);
}

// assign
template<typename scal_type>
inline
vec<scal_type, 4>& vec<scal_type, 4>::operator=(const vec<scal_type, 4>& rhs)
{
    // performancewise very bad!
    //vec<scal_type, 4> tmp(rhs);
    //swap(tmp);

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator=(const vec<rhs_scal_t, 4>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;

    return (*this);
}

// unary operators
template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator+=(const scal_type s)
{
    x += s;
    y += s;
    z += s;
    w += s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator+=(const vec<scal_type, 4>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;
    z -= s;
    w -= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator-=(const vec<scal_type, 4>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;
    z *= s;
    w *= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator*=(const vec<scal_type, 4>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;
    z /= s;
    w /= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator/=(const vec<scal_type, 4>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;

    return (*this);
}

// unary operators
template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator+=(const rhs_scal_t s)
{
    x += s;
    y += s;
    z += s;
    w += s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator+=(const vec<rhs_scal_t, 4>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator-=(const rhs_scal_t s)
{
    x -= s;
    y -= s;
    z -= s;
    w -= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator-=(const vec<rhs_scal_t, 4>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator*=(const rhs_scal_t s)
{
    x *= s;
    y *= s;
    z *= s;
    w *= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator*=(const vec<rhs_scal_t, 4>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator/=(const rhs_scal_t s)
{
    x /= s;
    y /= s;
    z /= s;
    w /= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 4>&
vec<scal_type, 4>::operator/=(const vec<rhs_scal_t, 4>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;

    return (*this);
}

template<typename scal_type>
inline
bool
vec<scal_type, 4>::operator==(const vec<scal_type, 4>& v) const
{
    return ((x == v.x) && (y == v.y) && (z == v.z) && (w == v.w));
}

template<typename scal_type>
inline
bool
vec<scal_type, 4>::operator!=(const vec<scal_type, 4>& v) const
{
    return ((x != v.x) || (y != v.y) || (z != v.z) || (w != v.w));
}

// common functions
template<typename scal_type>
inline
scal_type
dot(const vec<scal_type, 4>& lhs,
    const vec<scal_type, 4>& rhs)
{
    return (  lhs.x * rhs.x
            + lhs.y * rhs.y
            + lhs.z * rhs.z
            + lhs.w * rhs.w);
}

template<typename scal_type>
inline
const vec<scal_type, 4>
cross(const vec<scal_type, 4>& lhs,
      const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs.y * rhs.z - lhs.z * rhs.y,
                              lhs.z * rhs.x - lhs.x * rhs.z,
                              lhs.x * rhs.y - lhs.y * rhs.x,
                              scal_type(0)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
clamp(const vec<scal_type, 4>& val,
      const vec<scal_type, 4>& min,
      const vec<scal_type, 4>& max)
{
    return (vec<scal_type, 4>(clamp(val.x, min.x, max.x),
                              clamp(val.y, min.y, max.y),
                              clamp(val.z, min.z, max.z),
                              clamp(val.w, min.w, max.w)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
pow(const vec<scal_type, 4>& val,
    const scal_type          exp)
{
    return (vec<scal_type, 4>(std::pow(val.x, exp),
                              std::pow(val.y, exp),
                              std::pow(val.z, exp),
                              std::pow(val.w, exp)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
min(const vec<scal_type, 4>& a,
    const vec<scal_type, 4>& b)
{
    return (vec<scal_type, 4>(min(a.x, b.x),
                              min(a.y, b.y),
                              min(a.z, b.z),
                              min(a.w, b.w)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
max(const vec<scal_type, 4>& a,
    const vec<scal_type, 4>& b)
{
    return (vec<scal_type, 4>(max(a.x, b.x),
                              max(a.y, b.y),
                              max(a.z, b.z),
                              max(a.w, b.w)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
abs(const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(std::abs(rhs.x),
                              std::abs(rhs.y),
                              std::abs(rhs.z),
                              std::abs(rhs.w)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
floor(const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(std::floor(rhs.x),
                              std::floor(rhs.y),
                              std::floor(rhs.z),
                              std::floor(rhs.w)));
}

template<typename scal_type>
inline
const vec<scal_type, 4>
ceil(const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(std::ceil(rhs.x),
                              std::ceil(rhs.y),
                              std::ceil(rhs.z),
                              std::ceil(rhs.w)));
}

template<typename scal_type> 
inline
const vec<scal_type, 4>
fract(const vec<scal_type, 4>& rhs)
{ 
    return (rhs - floor(rhs));
}

} // namespace math
} // namespace scm
