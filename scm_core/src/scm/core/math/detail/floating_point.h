
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_DETAIL_FLOATING_POINT_H_INCLUDED
#define MATH_DETAIL_FLOATING_POINT_H_INCLUDED

#include <limits>

namespace scm {
namespace math {
namespace detail {

inline bool floating_point_equal(const float a, const float b)
{
    return (true);
}

inline bool floating_point_equal(const double a, const double b)
{
    return (true);
}

} // namespace detail
} // namespace math
} // namespace scm


#endif // MATH_DETAIL_FLOATING_POINT_H_INCLUDED
