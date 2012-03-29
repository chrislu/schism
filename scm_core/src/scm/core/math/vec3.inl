
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline
vec<scal_type, 3>::vec()
{
}

template<typename scal_type>
inline
vec<scal_type, 3>::vec(const vec<scal_type, 3>& v)
  : x(v.x), y(v.y), z(v.z)
{
}

template<typename scal_type>
inline
vec<scal_type, 3>::vec(const vec<scal_type, 2>& v,
                       const scal_type          z)
  : x(v.x), y(v.y), z(z)
{
}

template<typename scal_type>
inline
vec<scal_type, 3>::vec(const vec<scal_type, 4>& v)
  : x(v.x), y(v.y), z(v.z)
{
}

//template<typename scal_type>
//inline vec<scal_type, 3>::vec(const scal_type a[3])
//{
//    std::copy(a, a + 3, data_array);
//}

template<typename scal_type>
inline
vec<scal_type, 3>::vec(const scal_type s)
  : x(s), y(s), z(s) 
{
}

template<typename scal_type>
inline
vec<scal_type, 3>::vec(const scal_type _x,
                       const scal_type _y,
                       const scal_type _z)
  : x(_x), y(_y), z(_z)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>::vec(const vec<rhs_scal_t, 3>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y)),
    z(static_cast<scal_type>(v.z))
{
}

// dtor
//template<typename scal_type>
//inline vec<scal_type, 3>::~vec()
//{
//}

// constants
template<typename scal_type>
inline
const vec<scal_type, 3>&
vec<scal_type, 3>::zero()
{
    static vec<scal_type, 3> zero_(scal_type(0), scal_type(0), scal_type(0));
    return (zero_);
}

template<typename scal_type>
inline
const vec<scal_type, 3>&
vec<scal_type, 3>::one()
{
    static vec<scal_type, 3> one_(scal_type(1), scal_type(1), scal_type(1));
    return (one_);
}

// swap
template<typename scal_type>
inline
void vec<scal_type, 3>::swap(vec<scal_type, 3>& rhs)
{
    std::swap_ranges(data_array, data_array + 3, rhs.data_array);
}

// assign
template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator=(const vec<scal_type, 3>& rhs)
{
    // performance wise very bad!
    //vec<scal_type, 3> tmp(rhs);
    //swap(tmp);

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator=(const vec<rhs_scal_t, 3>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;

    return (*this);
}

// unary operators
template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator+=(const scal_type s)
{
    x += s;
    y += s;
    z += s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator+=(const vec<scal_type, 3>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;
    z -= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator-=(const vec<scal_type, 3>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;
    z *= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator*=(const vec<scal_type, 3>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;
    z /= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator/=(const vec<scal_type, 3>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;

    return (*this);
}

// unary operators
template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator+=(const rhs_scal_t s)
{
    x += s;
    y += s;
    z += s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator+=(const vec<rhs_scal_t, 3>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator-=(const rhs_scal_t s)
{
    x -= s;
    y -= s;
    z -= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator-=(const vec<rhs_scal_t, 3>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator*=(const rhs_scal_t s)
{
    x *= s;
    y *= s;
    z *= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator*=(const vec<rhs_scal_t, 3>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator/=(const rhs_scal_t s)
{
    x /= s;
    y /= s;
    z /= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 3>&
vec<scal_type, 3>::operator/=(const vec<rhs_scal_t, 3>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;

    return (*this);
}

template<typename scal_type>
inline
bool
vec<scal_type, 3>::operator==(const vec<scal_type, 3>& v) const
{
    return ((x == v.x) && (y == v.y) && (z == v.z));
}

template<typename scal_type>
inline
bool
vec<scal_type, 3>::operator!=(const vec<scal_type, 3>& v) const
{
    return ((x != v.x) || (y != v.y) || (z != v.z));
}

// common functions
template<typename scal_type>
inline
scal_type
dot(const vec<scal_type, 3>& lhs,
    const vec<scal_type, 3>& rhs)
{
    return (  lhs.x * rhs.x
            + lhs.y * rhs.y
            + lhs.z * rhs.z);
}

template<typename scal_type>
inline
const vec<scal_type, 3>
cross(const vec<scal_type, 3>& lhs,
      const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs.y * rhs.z - lhs.z * rhs.y,
                              lhs.z * rhs.x - lhs.x * rhs.z,
                              lhs.x * rhs.y - lhs.y * rhs.x));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
clamp(const vec<scal_type, 3>& val,
      const vec<scal_type, 3>& min,
      const vec<scal_type, 3>& max)
{
    return (vec<scal_type, 3>(clamp(val.x, min.x, max.x),
                              clamp(val.y, min.y, max.y),
                              clamp(val.z, min.z, max.z)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
pow(const vec<scal_type, 3>& val,
    const scal_type          exp)
{
    return (vec<scal_type, 3>(std::pow(val.x, exp),
                              std::pow(val.y, exp),
                              std::pow(val.z, exp)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
min(const vec<scal_type, 3>& a,
    const vec<scal_type, 3>& b)
{
    return (vec<scal_type, 3>(min(a.x, b.x),
                              min(a.y, b.y),
                              min(a.z, b.z)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
max(const vec<scal_type, 3>& a,
    const vec<scal_type, 3>& b)
{
    return (vec<scal_type, 3>(max(a.x, b.x),
                              max(a.y, b.y),
                              max(a.z, b.z)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
abs(const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(std::abs(rhs.x),
                              std::abs(rhs.y),
                              std::abs(rhs.z)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
floor(const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(std::floor(rhs.x),
                              std::floor(rhs.y),
                              std::floor(rhs.z)));
}

template<typename scal_type>
inline
const vec<scal_type, 3>
ceil(const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(std::ceil(rhs.x),
                              std::ceil(rhs.y),
                              std::ceil(rhs.z)));
}

template<typename scal_type> 
inline
const vec<scal_type, 3>
fract(const vec<scal_type, 3>& rhs)
{ 
    return (rhs - floor(rhs));
}

} // namespace math
} // namespace scm
