
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/core/math/common.h>
#include <scm/core/math/vec_fwd.h>

namespace std {

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline void swap(scm::math::mat<scal_type, row_dim, col_dim>& lhs,
                 scm::math::mat<scal_type, row_dim, col_dim>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

namespace scm {
namespace math {

// default operators
// unary
template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
mat<scal_type, row_dim, col_dim>& 
operator+=(      mat<scal_type, row_dim, col_dim>& lhs,
           const mat<scal_type, row_dim, col_dim>& rhs)
{
    for (unsigned i = 0; i < (row_dim * col_dim); ++i) {
        lhs.data_array[i] += rhs.data_array[i];
    }
    return (lhs);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
mat<scal_type, row_dim, col_dim>&
operator-=(      mat<scal_type, row_dim, col_dim>& lhs,
           const mat<scal_type, row_dim, col_dim>& rhs)
{
    for (unsigned i = 0; i < (row_dim * col_dim); ++i) {
        lhs.data_array[i] -= rhs.data_array[i];
    }
    return (lhs);
}

template<typename scal_type,
         const unsigned order>
inline
mat<scal_type, order, order>&
operator*=(      mat<scal_type, order, order>& lhs,
           const mat<scal_type, order, order>& rhs)
{
    mat<scal_type, order, order> tmp_ret;

    unsigned    dst_off;
    unsigned    row_off;
    unsigned    col_off;

    scal_type   tmp_dp;

    for (unsigned c = 0; c < order; ++c) {
        for (unsigned r = 0; r < order; ++r) {
            dst_off = r + order * c;
            tmp_dp = scal_type(0);

            for (unsigned d = 0; d < order; ++d) {
                row_off = r + d * order;
                col_off = d + c * order;
                tmp_dp += lhs.data_array[row_off] * rhs.data_array[col_off];
            }

            tmp_ret.data_array[dst_off] = tmp_dp;
        }
    }

    lhs = tmp_ret;

    return (lhs);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
mat<scal_type, row_dim, col_dim>&
operator*=(mat<scal_type, row_dim, col_dim>&   lhs,
           const scal_type                     rhs)
{
    for (unsigned i = 0; i < (row_dim * col_dim); ++i) {
        lhs.data_array[i] *= rhs;
    }
    return (lhs);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
mat<scal_type, row_dim, col_dim>&
operator/=(mat<scal_type, row_dim, col_dim>& lhs,
           const scal_type rhs)
{
    for (unsigned i = 0; i < (row_dim * col_dim); ++i) {
        lhs.data_array[i] /= rhs;
    }
    return (lhs);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
bool
operator==(const mat<scal_type, row_dim, col_dim>& lhs,
           const mat<scal_type, row_dim, col_dim>& rhs)
{
    bool return_value = true;

    for (unsigned i = 0; (i < (row_dim * col_dim)) && return_value; ++i) {
        return_value = (return_value && (lhs.data_array[i] == rhs.data_array[i]));
    }

    return (return_value);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
bool
operator!=(const mat<scal_type, row_dim, col_dim>& lhs,
           const mat<scal_type, row_dim, col_dim>& rhs)
{
    bool return_value = false;

    for (unsigned i = 0; (i < (row_dim * col_dim)) && !return_value; ++i) {
        return_value = (return_value || (lhs.data_array[i] != rhs.data_array[i]));
    }

    return (return_value);
}

// binary operators
template<typename scal_type,
         const unsigned order>
inline
const mat<scal_type, order, order>
operator*(const mat<scal_type, order, order>& lhs,
          const mat<scal_type, order, order>& rhs)
{
    mat<scal_type, order, order> tmp(lhs);

    tmp *= rhs;

    return (tmp);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
const mat<scal_type, row_dim, col_dim>
operator*(const mat<scal_type, row_dim, col_dim>& lhs,
          const scal_type                         rhs)
{
    mat<scal_type, row_dim, col_dim> tmp(lhs);

    tmp *= rhs;

    return (tmp);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
const mat<scal_type, row_dim, col_dim>
operator/(const mat<scal_type, row_dim, col_dim>& lhs,
          const scal_type                         rhs)
{
    mat<scal_type, row_dim, col_dim> tmp(lhs);

    tmp /= rhs;

    return (tmp);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
const mat<scal_type, row_dim, col_dim>
operator+(const mat<scal_type, row_dim, col_dim>& lhs,
          const mat<scal_type, row_dim, col_dim>& rhs)
{
    mat<scal_type, row_dim, col_dim> tmp(lhs);

    tmp += rhs;

    return (tmp);
}

template<typename scal_type,
         const unsigned row_dim,
         const unsigned col_dim>
inline
const mat<scal_type, row_dim, col_dim>
operator-(const mat<scal_type, row_dim, col_dim>& lhs,
          const mat<scal_type, row_dim, col_dim>& rhs)
{
    mat<scal_type, row_dim, col_dim> tmp(lhs);

    tmp -= rhs;

    return (tmp);
}

template<typename scal_type,
         const unsigned order>
inline
const vec<scal_type, order>
operator*(const mat<scal_type, order, order>& lhs,
          const vec<scal_type, order>&        rhs)
{
    vec<scal_type, order> tmp_ret(scal_type(0));

    unsigned    row_off;

    scal_type   tmp_dp;

    for (unsigned r = 0; r < order; ++r) {
        tmp_dp = scal_type(0);

        for (unsigned c = 0; c < order; ++c) {
            row_off = r + c * order;
            tmp_dp += lhs.data_array[row_off] * rhs.data_array[c];
        }

        tmp_ret.data_array[r] = tmp_dp;
    }

    return (tmp_ret);
}

template<typename scal_type,
         const unsigned order>
inline
const vec<scal_type, order>
operator*(const mat<scal_type, order, order>& lhs,
          const vec<scal_type, order - 1>&    rhs)
{
    vec<scal_type, order - 1> tmp_ret(scal_type(0));

    unsigned    row_off;

    scal_type   tmp_dp;

    for (unsigned r = 0; r < order - 1; ++r) {
        tmp_dp = scal_type(0);

        for (unsigned c = 0; c < order - 1; ++c) {
            row_off = r + c * order;
            tmp_dp += lhs.data_array[row_off] * rhs.data_array[c];
        }

        // w == 1
        tmp_ret.data_array[r] = tmp_dp + lhs.data_array[r + (order - 1) * order];
    }

    return (tmp_ret);
}

template<typename scal_type,
         const unsigned order>
inline
const vec<scal_type, order>
operator*(const vec<scal_type, order>&        lhs,
          const mat<scal_type, order, order>& rhs)
{
    vec<scal_type, order> tmp_ret(scal_type(0));

    unsigned    row_off;

    scal_type   tmp_dp;

    for (unsigned r = 0; r < order; ++r) {
        tmp_dp = scal_type(0);

        for (unsigned c = 0; c < order; ++c) {
            row_off = r * order + c;
            tmp_dp += rhs.data_array[row_off] * lhs.data_array[c];
        }

        tmp_ret.data_array[r] = tmp_dp;
    }

    return (tmp_ret);
}

// common functions
template<typename scal_type,
         const unsigned order>
inline
void
set_identity(mat<scal_type, order, order>& m)
{
    for (unsigned i = 0; i < (order * order); ++i) {
        m.data_array[i] = (i % (order + 1)) == 0 ? scal_type(1) : scal_type(0);
    }
}

template<typename scal_type, 
         const unsigned row_dim,
         const unsigned col_dim>
inline
const mat<scal_type, row_dim, col_dim>
transpose(const mat<scal_type, row_dim, col_dim>& lhs)
{
    mat<scal_type, col_dim, row_dim> tmp_ret;

    unsigned src_off;
    unsigned dst_off;

    for (unsigned c = 0; c < col_dim; ++c) {
        for (unsigned r = 0; r < row_dim; ++r) {
            src_off = r + c * row_dim;
            dst_off = c + r * col_dim;

            tmp_ret.data_array[dst_off] = lhs.data_array[src_off];
        }
    }

    return (tmp_ret);
}

template<typename scal_type,
         const unsigned order>
inline
const mat<scal_type, order - 1, order - 1>
minor_mat(const mat<scal_type, order, order>& lhs,
          const unsigned row,
          const unsigned col)
{
    mat<scal_type, order - 1, order - 1>   tmp_minor;

    unsigned min_off;
    unsigned src_off;

    unsigned min_row = 0;
    unsigned min_col = 0;

    for (unsigned r = 0; r < order; ++r) {
        if (r != row) {
            min_col = 0;
            for (unsigned c = 0; c < order; ++c) {
                if (c != col) {
                    src_off = r + c * order;
                    min_off = min_row + min_col * (order - 1);

                    tmp_minor.data_array[min_off] = lhs.data_array[src_off];
                    ++min_col;
                }
            }
            ++min_row;
        }
    }

    return (tmp_minor);
}

template<typename scal_type>
inline
scal_type
determinant(const mat<scal_type, 1, 1>& lhs)
{
    return (lhs.data_array[0]);
}

template<typename scal_type>
inline
scal_type
determinant(const mat<scal_type, 2, 2>& lhs)
{
    return (lhs.data_array[0] * lhs.data_array[3] - lhs.data_array[1] * lhs.data_array[2]);
}

template<typename scal_type, const unsigned order>
inline
scal_type
determinant(const mat<scal_type, order, order>& lhs)
{
    scal_type tmp_ret = scal_type(0);

    // determinat development after first column
    for (unsigned r = 0; r < order; ++r) {
        tmp_ret +=  lhs.data_array[r] * sign(-int(r % 2)) * determinant(minor_mat(lhs, r, 0));
    }

    return (tmp_ret);
}

template<typename scal_type, const unsigned order>
inline
const mat<scal_type, order, order>
inverse(const mat<scal_type, order, order>& lhs)
{
    mat<scal_type, order, order> tmp_ret(mat<scal_type, order, order>::zero());
    scal_type                    tmp_det = determinant(lhs);

    unsigned dst_off;

    // ATTENTION!!!! float equal test
    if (tmp_det != scal_type(0)) {
        for (unsigned r = 0; r < order; ++r) {
            for (unsigned c = 0; c < order; ++c) {
                dst_off = c + r * order;
                tmp_ret.data_array[dst_off] = (scal_type(1) / tmp_det) * sign(-int((r+c) % 2)) * determinant(minor_mat(lhs, r, c));
            }
        }
    }

    return (tmp_ret);
}

} // namespace math
} // namespace scm
