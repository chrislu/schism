
#ifndef SCM_MATH_LIB_H_INCLUDED
#define SCM_MATH_LIB_H_INCLUDED

#include <cmath>

namespace math
{
    // trigonometiy
    inline const float sin(const float rad)
    {
        return (std::sin(rad));
    }

    inline const double sin(const double rad)
    {
        return (std::sin(rad));
    }

    inline const float asin(const float rad)
    {
        return (std::asin(rad));
    }

    inline const double asin(const double rad)
    {
        return (std::asin(rad));
    }

    inline const float cos(const float rad)
    {
        return (std::cos(rad));
    }

    inline const double cos(const double rad)
    {
        return (std::cos(rad));
    }

    inline const float acos(const float rad)
    {
        return (std::acos(rad));
    }

    inline const double acos(const double rad)
    {
        return (std::acos(rad));
    }

    inline const float tan(const float rad)
    {
        return (std::tan(rad));
    }

    inline const double tan(const double rad)
    {
        return (std::tan(rad));
    }

    inline const float atan(const float rad)
    {
        return (std::atan(rad));
    }

    inline const double atan(const double rad)
    {
        return (std::atan(rad));
    }

    inline const float sqrt(const float x)
    {
        return (std::sqrt(x));
    }

    inline const double sqrt(const double x)
    {
        return (std::sqrt(x));
    }

    inline const float log(const float x)
    {
        return (std::log(x));
    }

    inline const double log(const double x)
    {
        return (std::log(x));
    }

    inline const float log10(const float x)
    {
        return (std::log10(x));
    }

    inline const double log10(const double x)
    {
        return (std::log10(x));
    }

    // functions
    template<typename scal>
    inline int sgn(const scal val)
    {
        return ((val < scal(0)) ? -1 : 1);
    }

    template<typename T> 
    inline T max(const T a, const T b)
    { 
        return ( (a > b) ? a : b );
    }

    template<typename T>
    inline T min(const T a, const T b)
    {
        return ( (a < b) ? a : b );
    }

    template<typename T>
    inline const T abs(const T a)
    {
        return ( (a < T(0)) ? -a : a );
    }
    
    inline const float abs(const float x)
    {
        return (std::fabs(x));
    }

    inline const double abs(const double x)
    {
        return (std::fabs(x));
    }

    inline const float floor(const float x)
    {
        return (std::floor(x));
    }

    inline const double floor(const double x)
    {
        return (std::floor(x));
    }

    inline const float ceil(const float x)
    {
        return (std::ceil(x));
    }

    inline const double ceil(const double x)
    {
        return (std::ceil(x));
    }

    inline const float pow(const float x, const float y)
    {
        return (std::pow(x, y));
    }

    inline const double pow(const double x, const double y)
    {
        return (std::pow(x, y));
    }

    template<typename T> 
    inline T fract(const T a)
    { 
        return (a - floor(a));
    }

    template<typename T> 
    inline T round(const T a)
    { 
        return ((fract(a) < T(0.5)) ? floor(a) : ceil(a) );
    }

    template<typename T>
    inline bool in_range(const T x, const T a, const T b)
    {
        return ( (x >= a)  && (x <= b) );
    }

    template<typename T>
    inline T sqr(const T& a)
    {
        return ( a*a );
    }

    template<typename T>
    const T clamp(const T val, const T min, const T max) {
        return ((val > max) ? max : (val < min) ? min : val);
    }

    template<typename T>
    inline T lerp(const T min, const T max, const float a)
    {
        return (T(a * max + (1.0f - a) * min));
    }

    template<typename T>
    inline float shoothstep(const T min, const T max, const T x)
    {
        float s = clamp(float(x-min)/float(max-min), 0.0f, 1.0f);
        s = (s*s*(3.0f-2.0f*s));
        return (s);
    }

    template<typename scm_scalar>
    inline const scm_scalar rad2deg(const scm_scalar rad)
    {
	    return (rad * scm_scalar(57.295779513082320876798154814105));
    }

    template<typename scm_scalar>
    inline scm_scalar deg2rad(const scm_scalar deg)
    {
	    return (deg * scm_scalar(0.017453292519943295769236907684886));
    }

    inline bool is_pow_of_2(unsigned x)
    {
	    return (! ( x & (x-1)));
    }
} // namespace math

#endif // SCM_MATH_LIB_H_INCLUDED

