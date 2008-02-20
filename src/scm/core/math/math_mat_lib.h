
#ifndef SCM_MATH_MAT_LIB_H_INCLUDED
#define SCM_MATH_MAT_LIB_H_INCLUDED

#include "math_lib.h"

namespace math
{
    template<typename scm_scalar, unsigned order>
    inline void set_identity(mat<scm_scalar, order, order>& m)
    {
        for (unsigned i = 0; i < (order * order); ++i) {
            m.mat_array[i] = (i % (order + 1)) == 0 ? scm_scalar(1) : scm_scalar(0);
        }

    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> transpose(const mat<scm_scalar, row_dim, col_dim>& lhs)
    {
        mat<scm_scalar, col_dim, row_dim> tmp_ret;

        unsigned src_off;
        unsigned dst_off;

        for (unsigned c = 0; c < col_dim; c++) {
            for (unsigned r = 0; r < row_dim; r++) {
                src_off = r + c * row_dim;
                dst_off = c + r * col_dim;

                tmp_ret.mat_array[dst_off] = lhs.mat_array[src_off];
            }
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned order>
    inline const mat<scm_scalar, order - 1, order - 1> minor__(const mat<scm_scalar, order, order>& lhs, unsigned row, unsigned col)
    {
        mat<scm_scalar, order - 1, order - 1>   tmp_minor;

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

                        tmp_minor.mat_array[min_off] = lhs.mat_array[src_off];
                        ++min_col;
                    }
                }
                ++min_row;
            }
        }

        return (tmp_minor);
    }

    template<typename scm_scalar>
    inline scm_scalar determinant(const mat<scm_scalar, 1, 1>& lhs)
    {
        return (lhs.m00);
    }

    template<typename scm_scalar>
    inline scm_scalar determinant(const mat<scm_scalar, 2, 2>& lhs)
    {
        return (lhs.m00 * lhs.m03 - lhs.m01 * lhs.m02);
    }

    template<typename scm_scalar, unsigned order>
    inline scm_scalar determinant(const mat<scm_scalar, order, order>& lhs)
    {
        scm_scalar tmp_ret = scm_scalar(0);

        // determinat development after first column
        for (unsigned r = 0; r < order; ++r) {
            tmp_ret +=  lhs.mat_array[r] * sgn(-int(r % 2)) * determinant(minor__(lhs, r, 0));
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned order>
    inline const mat<scm_scalar, order, order> inverse(const mat<scm_scalar, order, order>& lhs)
    {
        mat<scm_scalar, order, order> tmp_ret(scm_scalar(0));
        scm_scalar                    tmp_det = determinant(lhs);

        unsigned dst_off;

        if (tmp_det != scm_scalar(0)) {
            for (unsigned r = 0; r < order; ++r) {
                for (unsigned c = 0; c < order; ++c) {
                    dst_off = c + r * order;
                    tmp_ret.mat_array[dst_off] = (scm_scalar(1) / tmp_det) * sgn(-int((r+c) % 2)) * determinant(minor__(lhs, r, c));
                }
            }
        }

        return (tmp_ret);
    }


} // namespace math

#endif // SCM_MATH_MAT_LIB_H_INCLUDED




