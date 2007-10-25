
#ifndef SCM_MATH_MAT_INL_INCLUDED
#define SCM_MATH_MAT_INL_INCLUDED

namespace math
{
    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline bool operator==(const mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        bool tmp_ret = true;

        for (unsigned i = 0; i < (row_dim * col_dim) && tmp_ret; i++) {
            tmp_ret = (lhs.mat_array[i] == rhs.mat_array[i]); // TODO something like epsilon compare
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline bool operator!=(const mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        bool tmp_ret = false;

        for (unsigned i = 0; i < (row_dim * col_dim) && !tmp_ret; i++) {
            tmp_ret = (lhs.mat_array[i] != rhs.mat_array[i]); // TODO something like epsilon compare
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const scm_scalar*const operator&(const mat<scm_scalar, row_dim, col_dim>& m)
    {
        return (m.mat_array);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator-(const mat<scm_scalar, row_dim, col_dim>& lhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = -lhs.mat_array[i];
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator+(const mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = lhs.mat_array[i] + rhs.mat_array[i];
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator-(const mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = lhs.mat_array[i] - rhs.mat_array[i];
        }

        return (tmp_ret);
    }

    // yet only for square matrices!
    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator*(const mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        unsigned    dst_off;
        unsigned    row_off;
        unsigned    col_off;

        scm_scalar  tmp_dp;

        for (unsigned c = 0; c < col_dim; c++) {
            for (unsigned r = 0; r < row_dim; r++) {
                dst_off = r + row_dim * c;
                tmp_dp = scm_scalar(0);

                for (unsigned d = 0; d < row_dim; d++) {
                    row_off = r + d * row_dim;
                    col_off = d + c * col_dim;
                    tmp_dp += lhs.mat_array[row_off] * rhs.mat_array[col_off];
                }

                tmp_ret.mat_array[dst_off] = tmp_dp;
            }
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const vec<scm_scalar, row_dim> operator*(const mat<scm_scalar, row_dim, col_dim>& lhs, const vec<scm_scalar, col_dim>& rhs)
    {
        vec<scm_scalar, row_dim> tmp_ret;

        unsigned    row_off;

        scm_scalar  tmp_dp;

        for (unsigned r = 0; r < row_dim; r++) {
            tmp_dp = scm_scalar(0);

            for (unsigned c = 0; c < col_dim; c++) {
                row_off = r + c * row_dim;
                tmp_dp += lhs.mat_array[row_off] * rhs.vec_array[c];
            }

            tmp_ret.vec_array[r] = tmp_dp;
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator*(const mat<scm_scalar, row_dim, col_dim>& lhs, const scm_scalar rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = lhs.mat_array[i] * rhs;
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator*(const scm_scalar lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = lhs * rhs.mat_array[i];
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline const mat<scm_scalar, row_dim, col_dim> operator/(const mat<scm_scalar, row_dim, col_dim>& lhs, const scm_scalar rhs)
    {
        mat<scm_scalar, row_dim, col_dim> tmp_ret;

        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            tmp_ret.mat_array[i] = lhs.mat_array[i] / rhs;
        }

        return (tmp_ret);
    }
    
    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline mat<scm_scalar, row_dim, col_dim>& operator+=(mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            lhs.mat_array[i] += rhs.mat_array[i];
        }
        return (lhs);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline mat<scm_scalar, row_dim, col_dim>& operator-=(mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            lhs.mat_array[i] -= rhs.mat_array[i];
        }
        return (lhs);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline mat<scm_scalar, row_dim, col_dim>& operator*=(mat<scm_scalar, row_dim, col_dim>& lhs, const mat<scm_scalar, row_dim, col_dim>& rhs)
    {
        lhs = lhs * rhs;

        return (lhs);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline mat<scm_scalar, row_dim, col_dim>& operator*=(mat<scm_scalar, row_dim, col_dim>& lhs, const scm_scalar rhs)
    {
        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            lhs.mat_array[i] *= rhs;
        }
        return (lhs);
    }

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    inline mat<scm_scalar, row_dim, col_dim>& operator/=(mat<scm_scalar, row_dim, col_dim>& lhs, const scm_scalar rhs)
    {
        for (unsigned i = 0; i < (row_dim * col_dim); i++) {
            lhs.mat_array[i] /= rhs;
        }
        return (lhs);
    }

} // namespace math

#endif // SCM_MATH_MAT_INL_INCLUDED

