
#ifndef SCM_GL_UTIL_WM_WM_FWD_H_INCLUDED
#define SCM_GL_UTIL_WM_WM_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {
namespace wm {

class context;
class display;
class surface;
class window;
class headless_surface;

typedef shared_ptr<context>             context_ptr;
typedef shared_ptr<display>             display_ptr;
typedef shared_ptr<surface>             surface_ptr;
typedef shared_ptr<window>              window_ptr;
typedef shared_ptr<headless_surface>    headless_surface_ptr;

} // namespace wm
} // namepspace gl
} // namepspace scm


#endif // SCM_GL_UTIL_WM_WM_FWD_H_INCLUDED