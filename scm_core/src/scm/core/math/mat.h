
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_MAT_H_INCLUDED
#define MATH_MAT_H_INCLUDED

namespace scm {
namespace math {

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
class mat
{
public:
    typedef scal_type   value_type;

public:
    // data definition
    scal_type  data_array[row_dim * col_dim];

}; // class mat<scal_type, row_dim, col_dim>


// savety for some recursive library functions
template<typename scal_type>
class mat<scal_type, 1, 1>
{
public:
    typedef scal_type   value_type;

public:
    // data definition
    union
    {
        scal_type  m00;
        scal_type  data_array[1];
    };

}; // class mat<scal_type, 1, 1>

template<typename scal_type, const unsigned row_dim, const unsigned col_dim> mat<scal_type, row_dim, col_dim>&      operator+=(mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> mat<scal_type, row_dim, col_dim>&      operator-=(mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);
template<typename scal_type, const unsigned order>                           mat<scal_type, order, order>&          operator*=(mat<scal_type, order, order>&     lhs, const mat<scal_type, order, order>&     rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> mat<scal_type, row_dim, col_dim>&      operator*=(mat<scal_type, row_dim, col_dim>& lhs, const scal_type                         rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> mat<scal_type, row_dim, col_dim>&      operator/=(mat<scal_type, row_dim, col_dim>& lhs, const scal_type                         rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> bool                                   operator==(const mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> bool                                   operator!=(const mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);

// binary operators
template<typename scal_type, const unsigned order>                           const mat<scal_type, order, order>     operator*(const mat<scal_type, order, order>& lhs, const mat<scal_type, order, order>& rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> const mat<scal_type, row_dim, col_dim> operator*(const mat<scal_type, row_dim, col_dim>& lhs, const scal_type                 rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> const mat<scal_type, row_dim, col_dim> operator/(const mat<scal_type, row_dim, col_dim>& lhs, const scal_type                         rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> const mat<scal_type, row_dim, col_dim> operator+(const mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> const mat<scal_type, row_dim, col_dim> operator-(const mat<scal_type, row_dim, col_dim>& lhs, const mat<scal_type, row_dim, col_dim>& rhs);
template<typename scal_type, const unsigned order>                           const vec<scal_type, order>            operator*(const mat<scal_type, order, order>& lhs, const vec<scal_type, order>&        rhs);
template<typename scal_type, const unsigned order>                           const vec<scal_type, order>            operator*(const mat<scal_type, order, order>& lhs, const vec<scal_type, order - 1>&    rhs);
template<typename scal_type, const unsigned order>                           const vec<scal_type, order>            operator*(const vec<scal_type, order>&        lhs, const mat<scal_type, order, order>& rhs);

// common functions
template<typename scal_type, const unsigned order>                           void                                   set_identity(mat<scal_type, order, order>& m);
template<typename scal_type, const unsigned row_dim, const unsigned col_dim> const mat<scal_type, row_dim, col_dim> transpose(const mat<scal_type, row_dim, col_dim>& lhs);
template<typename scal_type, const unsigned order>                           const mat<scal_type, order - 1, order - 1> minor_mat(const mat<scal_type, order, order>& lhs, const unsigned row, const unsigned col);
template<typename scal_type>                                                 scal_type                              determinant(const mat<scal_type, 1, 1>& lhs);
template<typename scal_type>                                                 scal_type                              determinant(const mat<scal_type, 2, 2>& lhs);
template<typename scal_type, const unsigned order>                           scal_type                              determinant(const mat<scal_type, order, order>& lhs);
template<typename scal_type, const unsigned order>                           const mat<scal_type, order, order>     inverse(const mat<scal_type, order, order>& lhs);

} // namespace math
} // namespace scm

namespace std {

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
void swap(scm::math::mat<scal_type, row_dim, col_dim>& lhs,
          scm::math::mat<scal_type, row_dim, col_dim>& rhs);

} // namespace std

#include "mat.inl"

#endif // MATH_MAT_H_INCLUDED
