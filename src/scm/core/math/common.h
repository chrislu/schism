
#ifndef MATH_COMMON_H_INCLUDED
#define MATH_COMMON_H_INCLUDED

#include <cassert>
#include <cmath>
#include <limits>

namespace scm {
namespace math {

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

template<typename scal_type>
inline int sign(const scal_type val)
{
    return ((val < scal_type(0)) ? -1 : 1);
}

template<typename T> 
inline T max(const T a,
             const T b)
{ 
    return ((a > b) ? a : b);
}

template<typename T>
inline T min(const T a,
             const T b)
{
    return ((a < b) ? a : b);
}

template<typename T> 
inline T fract(const T a)
{ 
    return (a - std::floor(a));
}

template<typename T> 
inline T round(const T a)
{ 
    return ((fract(a) < T(0.5)) ? std::floor(a) : std::ceil(a));
}

template<typename T>
inline bool in_range(const T x,
                     const T a,
                     const T b)
{
    return ((x >= a) && (x <= b));
}

template<typename T>
inline T sqr(const T& a)
{
    return (a * a);
}

template<typename T>
const T clamp(const T val,
              const T min,
              const T max)
{
    return ((val > max) ? max : (val < min) ? min : val);
}

template<typename T, typename scal_type>
inline T lerp(const T min,
              const T max,
              const scal_type a)
{
    return (T(a * max + (scal_type(1) - a) * min));
}

template<typename scal_type>
inline scal_type shoothstep(const scal_type min,
                            const scal_type max,
                            const scal_type x)
{
    scal_type s = clamp((x - min) / (max-min), 0.0f, 1.0f);
    s = (s*s*(3.0f-2.0f*s));

    return (s);
}

template<typename scal_type>
inline const scal_type rad2deg(const scal_type rad)
{
    return (rad * scal_type(57.295779513082320876798154814105));
}

template<typename scal_type>
inline scal_type deg2rad(const scal_type deg)
{
    return (deg * scal_type(0.017453292519943295769236907684886));
}

template<typename int_type>
inline bool is_power_of_two(int_type x)
{
    assert(    std::numeric_limits<int_type>::is_integer()
           && !std::numeric_limits<int_type>::is_signed());

    return (! ( x & (x-1)));
}

template<typename int_type>
inline unsigned next_power_of_two(int_type x)
{
    assert(    std::numeric_limits<int_type>::is_integer()
           && !std::numeric_limits<int_type>::is_signed()
           &&  sizeof(int_type) == 4);

    --x;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return (++x);
}

} // namespace math
} // namespace scm

#endif // MATH_COMMON_H_INCLUDED
