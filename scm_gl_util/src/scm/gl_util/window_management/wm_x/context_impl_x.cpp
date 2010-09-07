
#include "context_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>

namespace scm {
namespace gl {
namespace wm {

context::context_impl::context_impl(const surface&  in_surface,
                                    const context&  in_share_ctx)
{
    try {
    }
    catch(...) {
        cleanup();
        throw;
    }
}

context::context_impl::~context_impl()
{
    cleanup();
}

void
context::context_impl::cleanup()
{
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
