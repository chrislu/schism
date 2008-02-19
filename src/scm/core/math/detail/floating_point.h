
#ifndef MATH_DETAIL_FLOATING_POINT_H_INCLUDED
#define MATH_DETAIL_FLOATING_POINT_H_INCLUDED

#include <limits>

namespace scm {
namespace math {
namespace detail {

inline bool floating_point_compare(const float a, const float b)
{
    return (true);
}

inline bool floating_point_compare(const double a, const double b)
{
    return (true);
}

} // namespace detail
} // namespace math
} // namespace scm


#endif // MATH_DETAIL_FLOATING_POINT_H_INCLUDED
