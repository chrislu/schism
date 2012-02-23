
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "display.h"

#include <exception>
#include <stdexcept>

#include <scm/log.h>

#include <scm/gl_core/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_core/window_management/wm_x/display_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

display::display(const std::string& name)
{
    try {
        _impl.reset(new display_impl(name));
    }
    catch(const std::exception& e) {
        err() << log::fatal
              << "display::display(): "
              << log::indent << e.what() << log::outdent << log::end;
        throw (e);
    }
}

display::~display()
{
    _impl.reset();
}

} // namespace wm
} // namepspace gl
} // namepspace scm
