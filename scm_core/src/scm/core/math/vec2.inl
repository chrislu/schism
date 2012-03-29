
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline
vec<scal_type, 2>::vec()
{
}

template<typename scal_type>
inline
vec<scal_type, 2>::vec(const vec<scal_type, 2>& v)
  : x(v.x), y(v.y)
{
    //std::copy(v.data_array, v.data_array + 2, data_array);
}

template<typename scal_type>
inline
vec<scal_type, 2>::vec(const vec<scal_type, 3>& v)
  : x(v.x), y(v.y)
{
}

template<typename scal_type>
inline
vec<scal_type, 2>::vec(const vec<scal_type, 4>& v)
  : x(v.x), y(v.y)
{
}

//template<typename scal_type>
//inline vec<scal_type, 2>::vec(const scal_type a[2])
//{
//    std::copy(a, a + 2, data_array);
//}

template<typename scal_type>
inline
vec<scal_type, 2>::vec(const scal_type s)
  : x(s), y(s) 
{
}

template<typename scal_type>
inline
vec<scal_type, 2>::vec(const scal_type _x,
                       const scal_type _y)
  : x(_x), y(_y)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>::vec(const vec<rhs_scal_t, 2>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y))
{
}

// dtor
//template<typename scal_type>
//inline vec<scal_type, 2>::~vec()
//{
//}

// constants
template<typename scal_type>
inline
const vec<scal_type, 2>&
vec<scal_type, 2>::zero()
{
    static vec<scal_type, 2> zero_(scal_type(0), scal_type(0));
    return (zero_);
}

template<typename scal_type>
inline
const vec<scal_type, 2>&
vec<scal_type, 2>::one()
{
    static vec<scal_type, 2> one_(scal_type(1), scal_type(1));
    return (one_);
}

// swap
template<typename scal_type>
inline
void
vec<scal_type, 2>::swap(vec<scal_type, 2>& rhs)
{
    std::swap_ranges(data_array, data_array + 2, rhs.data_array);
}

// assign
template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator=(const vec<scal_type, 2>& rhs)
{
    // performance wise very bad!
    //vec<scal_type, 2> tmp(rhs);
    //swap(tmp);

    x = rhs.x;
    y = rhs.y;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator=(const vec<rhs_scal_t, 2>& rhs)
{
    x = rhs.x;
    y = rhs.y;

    return (*this);
}

// unary operators
template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator+=(const scal_type s)
{
    x += s;
    y += s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator+=(const vec<scal_type, 2>& v)
{
    x += v.x;
    y += v.y;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator-=(const vec<scal_type, 2>& v)
{
    x -= v.x;
    y -= v.y;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator*=(const vec<scal_type, 2>& v)
{
    x *= v.x;
    y *= v.y;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator/=(const vec<scal_type, 2>& v)
{
    x /= v.x;
    y /= v.y;

    return (*this);
}

// unary operators
template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator+=(const rhs_scal_t s)
{
    x += s;
    y += s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator+=(const vec<rhs_scal_t, 2>& v)
{
    x += v.x;
    y += v.y;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator-=(const rhs_scal_t s)
{
    x -= s;
    y -= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator-=(const vec<rhs_scal_t, 2>& v)
{
    x -= v.x;
    y -= v.y;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator*=(const rhs_scal_t s)
{
    x *= s;
    y *= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator*=(const vec<rhs_scal_t, 2>& v)
{
    x *= v.x;
    y *= v.y;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator/=(const rhs_scal_t s)
{
    x /= s;
    y /= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 2>&
vec<scal_type, 2>::operator/=(const vec<rhs_scal_t, 2>& v)
{
    x /= v.x;
    y /= v.y;

    return (*this);
}

template<typename scal_type>
inline bool vec<scal_type, 2>::operator==(const vec<scal_type, 2>& v) const
{
    return ((x == v.x) && (y == v.y));
}

template<typename scal_type>
inline bool vec<scal_type, 2>::operator!=(const vec<scal_type, 2>& v) const
{
    return ((x != v.x) || (y != v.y));
}

// common functions
template<typename scal_type>
inline
scal_type dot(const vec<scal_type, 2>& lhs,
              const vec<scal_type, 2>& rhs)
{
    return (  lhs.x * rhs.x
            + lhs.y * rhs.y);
}

template<typename scal_type>
inline
const vec<scal_type, 2>
cross(const vec<scal_type, 2>& lhs,
      const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(lhs.y * rhs.z - lhs.z * rhs.y,
                              lhs.z * rhs.x - lhs.x * rhs.z));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
clamp(const vec<scal_type, 2>& val,
      const vec<scal_type, 2>& min,
      const vec<scal_type, 2>& max)
{
    return (vec<scal_type, 2>(clamp(val.x, min.x, max.x),
                              clamp(val.y, min.y, max.y)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
pow(const vec<scal_type, 2>& val,
    const scal_type          exp)
{
    return (vec<scal_type, 2>(std::pow(val.x, exp),
                              std::pow(val.y, exp)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
min(const vec<scal_type, 2>& a,
    const vec<scal_type, 2>& b)
{
    return (vec<scal_type, 2>(min(a.x, b.x),
                              min(a.y, b.y)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
max(const vec<scal_type, 2>& a,
    const vec<scal_type, 2>& b)
{
    return (vec<scal_type, 2>(max(a.x, b.x),
                              max(a.y, b.y)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
abs(const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(std::abs(rhs.x),
                              std::abs(rhs.y)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
floor(const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(std::floor(rhs.x),
                              std::floor(rhs.y)));
}

template<typename scal_type>
inline
const vec<scal_type, 2>
ceil(const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(std::ceil(rhs.x),
                              std::ceil(rhs.y)));
}

template<typename scal_type> 
inline
const vec<scal_type, 2>
fract(const vec<scal_type, 2>& rhs)
{ 
    return (rhs - floor(rhs));
}

} // namespace math
} // namespace scm
