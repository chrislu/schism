
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
