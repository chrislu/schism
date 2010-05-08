
#include "viewport.h"

namespace scm {
namespace gl {

viewport::viewport(const math::vec2ui& in_lower_left,
                   const math::vec2ui& in_dimensions,
                   const math::vec2f&  in_depth_range)
  : _lower_left(in_lower_left),
    _dimensions(in_dimensions),
    _depth_range(in_depth_range)
{
}

bool
viewport::operator==(const viewport& rhs) const
{
    return (   (_lower_left  == rhs._lower_left)
            && (_dimensions  == rhs._dimensions)
            && (_depth_range == rhs._depth_range));
}

bool
viewport::operator!=(const viewport& rhs) const
{
    return (   (_lower_left  != rhs._lower_left)
            || (_dimensions  != rhs._dimensions)
            || (_depth_range != rhs._depth_range));
}

} // namespace gl
} // namespace scm
