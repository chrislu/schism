
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DEVICE_RESOURCE_H_INCLUDED
#define SCM_GL_CORE_DEVICE_RESOURCE_H_INCLUDED

#include <iosfwd>

#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class render_device;

class __scm_export(gl_core) render_device_resource : public render_device_child
{
public:
    virtual ~render_device_resource();

    virtual void        print(std::ostream& os) const = 0;

protected:
    render_device_resource(render_device& dev);

}; // class render_device_resource

} // namespace scm
} // namespace gl

#endif // SCM_GL_CORE_DEVICE_RESOURCE_H_INCLUDED
