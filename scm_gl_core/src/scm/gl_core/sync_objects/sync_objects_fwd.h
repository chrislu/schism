
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SYNC_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_SYNC_OBJECTS_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class sync;
class fence_sync;

typedef shared_ptr<sync>                 sync_ptr;
typedef shared_ptr<sync const>           sync_cptr;
typedef shared_ptr<fence_sync>           fence_sync_ptr;
typedef shared_ptr<fence_sync const>     fence_sync_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_SYNC_OBJECTS_FWD_H_INCLUDED
