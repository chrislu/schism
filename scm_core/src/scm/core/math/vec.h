
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_VEC_H_INLCUDED
#define MATH_VEC_H_INLCUDED

#include <algorithm>

namespace scm {
namespace math {

template<typename scal_type, const unsigned dim>
class vec
{
    typedef scal_type   value_type;
}; // class vec<scm_scalar, dim>

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator++(vec<scal_type, dim>& v, int);
template<typename scal_type, const unsigned dim> vec<scal_type, dim>&           operator++(vec<scal_type, dim>& v);

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator--(vec<scal_type, dim>& v, int);
template<typename scal_type, const unsigned dim> vec<scal_type, dim>&           operator--(vec<scal_type, dim>& v);

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator-(const vec<scal_type, dim>& rhs);

// binary operators
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator+(const vec<scal_type, dim>& lhs, const vec<scal_type, dim>& rhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator+(const vec<scal_type, dim>& lhs, const scal_type            rhs);

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator-(const vec<scal_type, dim>& lhs, const vec<scal_type, dim>& rhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator-(const vec<scal_type, dim>& lhs, const scal_type            rhs);

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator*(const vec<scal_type, dim>& lhs, const vec<scal_type, dim>& rhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator*(const scal_type            lhs, const vec<scal_type, dim>& rhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator*(const vec<scal_type, dim>& lhs, const scal_type            rhs);

template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator/(const vec<scal_type, dim>& lhs, const vec<scal_type, dim>& rhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      operator/(const vec<scal_type, dim>& lhs, const scal_type            rhs);;

template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator+(const vec<scal_type, dim>& lhs, const vec<rhs_scal_t, dim>& rhs);
template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator+(const vec<scal_type, dim>& lhs, const rhs_scal_t            rhs);

template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator-(const vec<scal_type, dim>& lhs, const vec<rhs_scal_t, dim>& rhs);
template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator-(const vec<scal_type, dim>& lhs, const rhs_scal_t            rhs);

template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator*(const vec<scal_type, dim>& lhs, const vec<rhs_scal_t, dim>& rhs);
template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator*(const rhs_scal_t           lhs, const vec<scal_type, dim>&  rhs);
template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator*(const vec<scal_type, dim>& lhs, const rhs_scal_t            rhs);

template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator/(const vec<scal_type, dim>& lhs,  const vec<rhs_scal_t, dim>& rhs);
template<typename scal_type, typename rhs_scal_t, const unsigned dim> const vec<scal_type, dim>     operator/(const vec<scal_type, dim>& lhs,  const rhs_scal_t            rhs);

template<typename scal_type, const unsigned dim> vec<scal_type, dim>            lerp(const vec<scal_type, dim>& min, const vec<scal_type, dim>& max, const scal_type a);
template<typename scal_type, const unsigned dim> scal_type                      length_sqr(const vec<scal_type, dim>& lhs);
template<typename scal_type, const unsigned dim> scal_type                      length(const vec<scal_type, dim>& lhs);
template<typename scal_type, const unsigned dim> const vec<scal_type, dim>      normalize(const vec<scal_type, dim>& lhs);
template<typename scal_type, const unsigned dim> scal_type                      distance(const vec<scal_type, dim>& lhs, const vec<scal_type, dim>& rhs);

} // namespace math
} // namespace scm

namespace std {

template<typename scal_type, const unsigned dim>
void swap(scm::math::vec<scal_type, dim>& lhs,
          scm::math::vec<scal_type, dim>& rhs);

} // namespace std

#include "vec.inl"

#endif // MATH_VEC_H_INLCUDED
