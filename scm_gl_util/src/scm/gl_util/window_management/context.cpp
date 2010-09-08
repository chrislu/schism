
#include "context.h"


#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_util/window_management/wm_win32/context_impl_win32.h>
#include <scm/gl_util/window_management/wm_x/context_impl_x.h>

namespace scm {
namespace gl {
namespace wm {


context::attribute_desc::attribute_desc(int version_major, int version_minor,
                                        bool compatibility, bool debug, bool forward)
  : _version_major(version_major),
    _version_minor(version_minor),
    _compatibility_profile(compatibility),
    _debug(debug),
    _forward_compatible(forward)
{
}

context::context(const surface_ptr&     in_surface,
                 const attribute_desc&  in_attributes,
                 const context_ptr&     in_share_ctx)
  : _associated_display(in_surface->associated_display()),
    _surface_format(in_surface->surface_format()),
    _attributes(in_attributes)
{
    try {
        _impl.reset(new context_impl(in_surface, in_attributes, in_share_ctx));
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
    //surface_ptr lock_cur_surface = _current_surface.lock();
    //if (lock_cur_surface) {
    //    _impl->make_current(lock_cur_surface, false);
    //}
    _impl.reset();
}

bool
context::make_current(const surface_ptr& in_surface, bool current)
{
    if (_impl->make_current(in_surface, current)) {
        _current_surface = in_surface;
        return (true);
    }
    else {
        err() << log::error
              << "context::make_current(): unable to make context current to surface." << log::end;
        return (false);
    }
}

const display_ptr&
context::associated_display() const
{
    return (_associated_display);
}

const surface::format_desc&
context::surface_format() const
{
    return (_surface_format);
}

const context::attribute_desc&
context::context_attributes() const
{
    return (_attributes);
}

/*static*/
const context::attribute_desc&
context::default_attributes()
{
    static attribute_desc   default_attrib(0, 0);
    return (default_attrib);
}

} // namespace wm
} // namepspace gl
} // namepspace scm
