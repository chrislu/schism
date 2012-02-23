
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline
vec<scal_type, 1>::vec()
{
}

template<typename scal_type>
inline
vec<scal_type, 1>::vec(const vec<scal_type, 1>& v)
  : x(v.x)
{
    //std::copy(v.data_array, v.data_array + 2, data_array);
}

template<typename scal_type>
inline
vec<scal_type, 1>::vec(const vec<scal_type, 2>& v)
  : x(v.x)
{
}

template<typename scal_type>
inline
vec<scal_type, 1>::vec(const vec<scal_type, 3>& v)
  : x(v.x)
{
}

template<typename scal_type>
inline
vec<scal_type, 1>::vec(const vec<scal_type, 4>& v)
  : x(v.x)
{
}

//template<typename scal_type>
//inline vec<scal_type, 1>::vec(const scal_type a[2])
//{
//    std::copy(a, a + 2, data_array);
//}

template<typename scal_type>
inline
vec<scal_type, 1>::vec(const scal_type s)
  : x(s)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>::vec(const vec<rhs_scal_t, 1>& v)
  : x(static_cast<scal_type>(v.x))
{
}

// dtor
//template<typename scal_type>
//inline vec<scal_type, 1>::~vec()
//{
//}

// constants
template<typename scal_type>
inline
const vec<scal_type, 1>&
vec<scal_type, 1>::zero()
{
    static vec<scal_type, 1> zero_(scal_type(0));
    return (zero_);
}

template<typename scal_type>
inline
const vec<scal_type, 1>&
vec<scal_type, 1>::one()
{
    static vec<scal_type, 1> one_(scal_type(1));
    return (one_);
}

// swap
template<typename scal_type>
inline
void
vec<scal_type, 1>::swap(vec<scal_type, 1>& rhs)
{
    std::swap_ranges(data_array, data_array + 1, rhs.data_array);
}

// assign
template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator=(const vec<scal_type, 1>& rhs)
{
    // performance wise very bad!
    //vec<scal_type, 1> tmp(rhs);
    //swap(tmp);

    x = rhs.x;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator=(const vec<rhs_scal_t, 1>& rhs)
{
    x = rhs.x;

    return (*this);
}

// unary operators
template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator+=(const scal_type s)
{
    x += s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator+=(const vec<scal_type, 1>& v)
{
    x += v.x;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator-=(const scal_type s)
{
    x -= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator-=(const vec<scal_type, 1>& v)
{
    x -= v.x;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator*=(const scal_type s)
{
    x *= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator*=(const vec<scal_type, 1>& v)
{
    x *= v.x;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator/=(const scal_type s)
{
    x /= s;

    return (*this);
}

template<typename scal_type>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator/=(const vec<scal_type, 1>& v)
{
    x /= v.x;

    return (*this);
}

// unary operators
template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator+=(const rhs_scal_t s)
{
    x += s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator+=(const vec<rhs_scal_t, 1>& v)
{
    x += v.x;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator-=(const rhs_scal_t s)
{
    x -= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator-=(const vec<rhs_scal_t, 1>& v)
{
    x -= v.x;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator*=(const rhs_scal_t s)
{
    x *= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator*=(const vec<rhs_scal_t, 1>& v)
{
    x *= v.x;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator/=(const rhs_scal_t s)
{
    x /= s;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline
vec<scal_type, 1>&
vec<scal_type, 1>::operator/=(const vec<rhs_scal_t, 1>& v)
{
    x /= v.x;

    return (*this);
}

template<typename scal_type>
inline bool vec<scal_type, 1>::operator==(const vec<scal_type, 1>& v) const
{
    return ((x == v.x));
}

template<typename scal_type>
inline bool vec<scal_type, 1>::operator!=(const vec<scal_type, 1>& v) const
{
    return ((x != v.x));
}

// common functions
template<typename scal_type>
inline
scal_type dot(const vec<scal_type, 1>& lhs,
              const vec<scal_type, 1>& rhs)
{
    return (  lhs.x * rhs.x);
}

template<typename scal_type>
inline
const vec<scal_type, 1>
clamp(const vec<scal_type, 1>& val,
      const vec<scal_type, 1>& min,
      const vec<scal_type, 1>& max)
{
    return (vec<scal_type, 1>(clamp(val.x, min.x, max.x)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
pow(const vec<scal_type, 1>& val,
    const scal_type          exp)
{
    return (vec<scal_type, 1>(std::pow(val.x, exp)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
min(const vec<scal_type, 1>& a,
    const vec<scal_type, 1>& b)
{
    return (vec<scal_type, 1>(min(a.x, b.x)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
max(const vec<scal_type, 1>& a,
    const vec<scal_type, 1>& b)
{
    return (vec<scal_type, 1>(max(a.x, b.x)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
abs(const vec<scal_type, 1>& rhs)
{
    return (vec<scal_type, 1>(std::abs(rhs.x)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
floor(const vec<scal_type, 1>& rhs)
{
    return (vec<scal_type, 1>(std::floor(rhs.x)));
}

template<typename scal_type>
inline
const vec<scal_type, 1>
ceil(const vec<scal_type, 1>& rhs)
{
    return (vec<scal_type, 1>(std::ceil(rhs.x)));
}

template<typename scal_type> 
inline
const vec<scal_type, 1>
fract(const vec<scal_type, 1>& rhs)
{ 
    return (rhs - floor(rhs));
}

} // namespace math
} // namespace scm
