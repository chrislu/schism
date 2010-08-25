
#include "window.h"

#include <exception>
#include <stdexcept>

#include <scm/gl_util/window_management/wm_win32/window_impl_win32.h>
#include <scm/gl_util/window_management/wm_x/window_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

window::window()
{
    try {
        _impl.reset(new window_impl());
    }
    catch(const std::exception& /*e*/) {
    }
}

window::~window()
{
    _impl.reset();
}

} // namespace wm
} // namepspace gl
} // namepspace scm
