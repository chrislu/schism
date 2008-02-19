
#ifndef SCM_MATH_VEC_LIB_H_INCLUDED
#define SCM_MATH_VEC_LIB_H_INCLUDED

namespace math
{
    template<typename scm_scalar, unsigned dim>
    inline scm_scalar dot(const vec<scm_scalar, dim>& lhs, const vec<scm_scalar, dim>& rhs)
    {
        scm_scalar tmp_ret = scm_scalar(0);

        for (unsigned i = 0; i < dim; i++) {
            tmp_ret += lhs.vec_array[i] * rhs.vec_array[i];
        }

        return (tmp_ret);
    }

    template<typename scm_scalar>
    inline const vec<scm_scalar, 3> cross(const vec<scm_scalar, 3>& lhs, const vec<scm_scalar, 3>& rhs)
    {
        vec<scm_scalar, 3> tmp_ret;

        tmp_ret.x = lhs.y * rhs.z - lhs.z * rhs.y;
        tmp_ret.y = lhs.z * rhs.x - lhs.x * rhs.z;
        tmp_ret.z = lhs.x * rhs.y - lhs.y * rhs.x;

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned dim>
    inline scm_scalar length_sqr(const vec<scm_scalar, dim>& lhs)
    {
        return (dot(lhs, lhs));
    }

    template<unsigned dim>
    inline float length(const vec<float, dim>& lhs)
    {
        return (std::sqrt(length_sqr(lhs)));
    }

    template<unsigned dim>
    inline double length(const vec<double, dim>& lhs)
    {
        return (std::sqrt(length_sqr(lhs)));
    }

    template<typename scm_scalar, unsigned dim>
    inline const vec<scm_scalar, dim> normalize(const vec<scm_scalar, dim>& lhs)
    {
        scm_scalar nrm = length(lhs);

        return ((nrm > scm_scalar(0)) ? (lhs / nrm) : vec<scm_scalar, dim>(scm_scalar(0))); // use limits!
    }

    template<typename scm_scalar, unsigned dim>
    const vec<scm_scalar, dim> clamp(const vec<scm_scalar, dim>& val, const vec<scm_scalar, dim>& min, const vec<scm_scalar, dim>& max)
    {
        vec<scm_scalar, dim> tmp_ret;

        for (unsigned i = 0; i < dim; i++) {
            tmp_ret[i] = ((val.vec_array[i] > max.vec_array[i]) ? max.vec_array[i] : (val.vec_array[i] < min.vec_array[i]) ? min.vec_array[i] : val.vec_array[i]);
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned dim>
    const vec<scm_scalar, dim> pow(const vec<scm_scalar, dim>& val, scm_scalar exponent)
    {
        vec<scm_scalar, dim> tmp_ret;

        for (unsigned i = 0; i < dim; i++) {
            tmp_ret[i] = math::pow(val.vec_array[i], exponent);
        }

        return (tmp_ret);
    }
} // namespace math

#endif // SCM_MATH_VEC_LIB_H_INCLUDED
