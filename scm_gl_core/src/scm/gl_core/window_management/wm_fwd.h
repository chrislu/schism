
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WM_FWD_H_INCLUDED
#define SCM_GL_CORE_WM_WM_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {
namespace wm {

class context;
class display;
class surface;
class window;
class headless_surface;

typedef shared_ptr<context>                 context_ptr;
typedef shared_ptr<context const>           context_cptr;
typedef shared_ptr<display>                 display_ptr;
typedef shared_ptr<display const>           display_cptr;
typedef shared_ptr<surface>                 surface_ptr;
typedef shared_ptr<surface const>           surface_cptr;
typedef shared_ptr<window>                  window_ptr;
typedef shared_ptr<window const>            window_cptr;
typedef shared_ptr<headless_surface>        headless_surface_ptr;
typedef shared_ptr<headless_surface const>  headless_surface_cptr;

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_GL_CORE_WM_WM_FWD_H_INCLUDED
