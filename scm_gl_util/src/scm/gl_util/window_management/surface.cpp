
#include "surface.h"

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>

namespace scm {
namespace gl {
namespace wm {

surface::surface(const display&           in_display,
                 const pixel_format_desc& in_pf)
  : _associated_display(in_display),
    _pixel_format(in_pf)
{
}

surface::~surface()
{
}

const display&
surface::associated_display() const
{
    return (_associated_display);
}

const pixel_format_desc&
surface::pixel_format() const
{
    return (_pixel_format);
}

} // namespace wm
} // namepspace gl
} // namepspace scm
