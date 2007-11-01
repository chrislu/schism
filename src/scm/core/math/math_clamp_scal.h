
#ifndef SCM_MATH_CLAMP_SCAL_H_INCLUDED
#define SCM_MATH_CLAMP_SCAL_H_INCLUDED

#include <cassert>
#include <limits>

#include <scm/core/math/math_lib.h>

#pragma warning (push)
#pragma warning (disable : 4146)

namespace math
{
    template<typename scm_scalar>
    class clamp_scal
    {
    public:
        clamp_scal() : _min(std::numeric_limits<scm_scalar>::is_integer ? (std::numeric_limits<scm_scalar>::min)() : -(std::numeric_limits<scm_scalar>::max)()),
                       _max((std::numeric_limits<scm_scalar>::max)()) {}
        clamp_scal(scm_scalar s) : _min(std::numeric_limits<scm_scalar>::is_integer ? (std::numeric_limits<scm_scalar>::min)() : -(std::numeric_limits<scm_scalar>::max)()),
                                   _max((std::numeric_limits<scm_scalar>::max)()),
                                   _val(s) {}
        clamp_scal(const clamp_scal<scm_scalar>& s) : _min(s._min), _max(s._max), _val(s._val) {}
        explicit clamp_scal(scm_scalar v, scm_scalar min, scm_scalar max) : _min(min), _max(max) { _val = math::clamp(v, min, max); }
        explicit clamp_scal(scm_scalar min, scm_scalar max) : _min(min), _max(max), _val(min) {}

        clamp_scal<scm_scalar>& operator = (const clamp_scal<scm_scalar>& rhs) { _val = math::clamp(rhs._val, _min, _max); return (*this); }
        clamp_scal<scm_scalar>& operator = (scm_scalar rhs) { _val = math::clamp(rhs, _min, _max); return (*this); }

        operator scm_scalar() {return _val;}
        operator const scm_scalar() const {return _val;}

        const scm_scalar  _min;
        const scm_scalar  _max;

    private:
        scm_scalar        _val;

        friend bool operator==<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend bool operator!=<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);

        friend bool operator==<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend bool operator!=<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);

        friend bool operator==<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);
        friend bool operator!=<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);

        friend clamp_scal<scm_scalar>& operator+=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend clamp_scal<scm_scalar>& operator-=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend clamp_scal<scm_scalar>& operator*=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend clamp_scal<scm_scalar>& operator/=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);

        friend clamp_scal<scm_scalar>& operator+=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend clamp_scal<scm_scalar>& operator-=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend clamp_scal<scm_scalar>& operator*=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend clamp_scal<scm_scalar>& operator/=<scm_scalar>(clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);

        friend scm_scalar& operator+=<scm_scalar>(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs);
        friend scm_scalar& operator-=<scm_scalar>(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs);
        friend scm_scalar& operator*=<scm_scalar>(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs);
        friend scm_scalar& operator/=<scm_scalar>(scm_scalar& lhs, const clamp_scal<scm_scalar>& rhs);

        friend const clamp_scal<scm_scalar> operator+<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend const clamp_scal<scm_scalar> operator-<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend const clamp_scal<scm_scalar> operator*<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);
        friend const clamp_scal<scm_scalar> operator/<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const clamp_scal<scm_scalar>& rhs);

        friend const clamp_scal<scm_scalar> operator+<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend const clamp_scal<scm_scalar> operator-<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend const clamp_scal<scm_scalar> operator*<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);
        friend const clamp_scal<scm_scalar> operator/<scm_scalar>(const clamp_scal<scm_scalar>& lhs, const scm_scalar rhs);

        friend const scm_scalar operator+<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);
        friend const scm_scalar operator-<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);
        friend const scm_scalar operator*<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);
        friend const scm_scalar operator/<scm_scalar>(const scm_scalar lhs, const clamp_scal<scm_scalar>& rhs);

    }; // class clamp_scal

} // namespace math

#pragma warning (pop)

#include "math_clamp_scal.inl"

#endif // SCM_MATH_CLAMP_SCAL_H_INCLUDED 


