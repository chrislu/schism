
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DEVICE_CHILD_H_INCLUDED
#define SCM_GL_CORE_DEVICE_CHILD_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/gl_core/object_state.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class render_device;

class __scm_export(gl_core) render_device_child : boost::noncopyable
{
public:
    virtual ~render_device_child();

    render_device&          parent_device();
    const render_device&    parent_device() const;

    object_state&           state()                 { return _state; }
    const object_state&     state() const           { return _state; }

                            operator bool() const   { return _state.ok(); }
    bool                    operator!() const       { return _state.fail(); }
    bool                    ok() const              { return _state.ok(); }
    bool                    bad() const             { return _state.bad(); }
    bool                    fail() const            { return _state.fail(); }

private:
    render_device&          _parent_device;
    object_state            _state;

protected:
    render_device_child(render_device& dev);

}; // class render_device_child

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_DEVICE_CHILD_H_INCLUDED
