
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_FONT_FWD_H_INCLUDED
#define SCM_GL_UTIL_FONT_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class font_face;
class text;
class text_renderer;

typedef shared_ptr<font_face>           font_face_ptr;
typedef shared_ptr<const font_face>     font_face_cptr;

typedef shared_ptr<text>                text_ptr;
typedef shared_ptr<const text>          text_cptr;

typedef shared_ptr<text_renderer>       text_renderer_ptr;
typedef shared_ptr<const text_renderer> text_renderer_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_FONT_FWD_H_INCLUDED
