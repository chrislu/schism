
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "device_child.h"

#include <scm/gl_core/render_device/device.h>

namespace scm {
namespace gl {

render_device_child::render_device_child(render_device& dev)
  : _parent_device(dev)
{
}

render_device_child::~render_device_child()
{
}

render_device&
render_device_child::parent_device()
{
    return _parent_device;
}

const render_device&
render_device_child::parent_device() const
{
    return _parent_device;
}

} // namespace gl
} // namespace scm
