
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_FENCE_SYNC_H_INCLUDED
#define SCM_GL_CORE_FENCE_SYNC_H_INCLUDED

#include <scm/gl_core/sync_objects/sync.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) fence_sync : public sync
{
public:
    virtual ~fence_sync();

protected:
    fence_sync(render_device& in_device);

protected:

private:
    friend class render_device;
    friend class render_context;

}; // class fence_sync

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_FENCE_SYNC_H_INCLUDED
