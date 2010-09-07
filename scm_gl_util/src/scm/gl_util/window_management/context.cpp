
#include "context.h"


#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/surface.h>

#include <scm/gl_util/window_management/wm_win32/context_impl_win32.h>
#include <scm/gl_util/window_management/wm_x/context_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

context::context(const surface&  in_surface,
                 const context&  in_share_ctx)
  : _associated_display(in_surface.associated_display()),
    _pixel_format(in_surface.pixel_format())
{
    try {
        _impl.reset(new context_impl(in_surface, in_share_ctx));
    }
    catch(const std::exception& e) {
        err() << log::fatal
              << "context::context(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw (e);
    }
}

context::~context()
{
    _impl.reset();
}

const display&
context::associated_display() const
{
    return (_associated_display);
}

const pixel_format_desc&
context::pixel_format() const
{
    return (_pixel_format);
}

} // namespace wm
} // namepspace gl
} // namepspace scm
