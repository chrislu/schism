
#ifndef SCM_MATH_SCAL_INL_INCLUDED
#define SCM_MATH_SCAL_INL_INCLUDED

#include <cmath>

namespace math
{
    template<typename scm_scalar>
    inline bool operator==(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs._val == rhs._val);
    }

    template<typename scm_scalar>
    inline bool operator!=(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs._val != rhs._val);
    }

    template<typename scm_scalar>
    inline bool operator==(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        return (lhs._val == rhs);
    }

    template<typename scm_scalar>
    inline bool operator!=(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        return (lhs._val != rhs);
    }

    template<typename scm_scalar>
    inline bool operator==(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs == rhs._val);
    }

    template<typename scm_scalar>
    inline bool operator!=(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs != rhs._val);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator+=(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs = lhs._val + rhs._val;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator-=(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs = lhs._val - rhs._val;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator*=(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs = lhs._val * rhs._val;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator/=(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs = lhs._val / rhs._val;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator+=(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        lhs = lhs._val + rhs;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator-=(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        lhs = lhs._val - rhs;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator*=(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        lhs = lhs._val * rhs;

        return (lhs);
    }

    template<typename scm_scalar>
    inline clamp_scal<scm_scalar>& operator/=(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        lhs = lhs._val / rhs;

        return (lhs);
    }

    template<typename scm_scalar>
    inline scm_scalar& operator+=(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs += rhs._val;
        return (lhs);
    }

    template<typename scm_scalar>
    inline scm_scalar& operator-=(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs -= rhs._val;
        return (lhs);
    }

    template<typename scm_scalar>
    inline scm_scalar& operator*=(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs *= rhs._val;
        return (lhs);
    }

    template<typename scm_scalar>
    inline scm_scalar& operator/=(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        lhs /= rhs._val;
        return (lhs);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator+(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val + rhs._val;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator-(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val - rhs._val;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator*(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val * rhs._val;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator/(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val / rhs._val;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator+(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val + rhs;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator-(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val - rhs;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator*(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val * rhs;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const clamp_scal<scm_scalar> operator/(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs)
    {
        clamp_scal<scm_scalar> tmp_ret(lhs._min, lhs._max);

        tmp_ret = lhs._val / rhs;

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const scm_scalar operator+(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs + rhs._val);
    }

    template<typename scm_scalar>
    inline const scm_scalar operator-(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs - rhs._val);
    }

    template<typename scm_scalar>
    inline const scm_scalar operator*(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs * rhs._val);
    }

    template<typename scm_scalar>
    inline const scm_scalar operator/(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs)
    {
        return (lhs / rhs._val);
    }
} // namespace math

#endif // SCM_MATH_SCAL_INL_INCLUDED



