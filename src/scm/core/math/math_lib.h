
#ifndef SCM_MATH_LIB_H_INCLUDED
#define SCM_MATH_LIB_H_INCLUDED

#include <cassert>
#include <cmath>

namespace math
{
    // trigonometiy
    using std::sin;
    using std::asin;
    using std::cos;
    using std::acos;
    using std::tan;
    using std::atan;
    using std::sqrt;
    using std::log;
    using std::log10;

    using std::abs;
    using std::fabs;
    using std::floor;
    using std::ceil;
    using std::pow;

    // functions
    template<typename scal>
    inline int sgn(const scal val)
    {
        return ((val < scal(0)) ? -1 : 1);
    }

    template<typename T> 
    inline T max(const T a, const T b)
    { 
        return ((a > b) ? a : b );
    }

    template<typename T>
    inline T min(const T a, const T b)
    {
        return ((a < b) ? a : b );
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

    inline bool is_power_of_two(unsigned x)
    {
	    return (! ( x & (x-1)));
    }

    inline unsigned next_power_of_two(unsigned x)
    {
        assert(sizeof(x) == 4);

        --x;

        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;

        return (++x);
    }

} // namespace math

#endif // SCM_MATH_LIB_H_INCLUDED

