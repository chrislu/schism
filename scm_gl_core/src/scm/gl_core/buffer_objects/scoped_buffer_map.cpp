
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "scoped_buffer_map.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects.h>


namespace scm {
namespace gl {

scoped_buffer_map::scoped_buffer_map(const render_context_ptr& in_context,
                                     const buffer_ptr&         in_buffer,
                                     const access_mode         in_access)
  : _context(in_context)
  , _buffer(in_buffer)
{
    assert(_context);
    assert(_buffer);

    _data = reinterpret_cast<uint8*>(_context->map_buffer(_buffer, in_access));
}

scoped_buffer_map::scoped_buffer_map(const render_context_ptr& in_context,
                                     const buffer_ptr&         in_buffer,
                                           scm::size_t         in_offset,
                                           scm::size_t         in_size,
                                     const access_mode         in_access)
  : _context(in_context)
  , _buffer(in_buffer)
{
    assert(_context);
    assert(_buffer);

    _data = reinterpret_cast<uint8*>(_context->map_buffer_range(_buffer, in_offset, in_size, in_access));
}

scoped_buffer_map::~scoped_buffer_map()
{
    assert(_context);
    assert(_buffer);

    _context->unmap_buffer(_buffer);
}

scoped_buffer_map::operator bool() const
{
    return valid();
}

bool
scoped_buffer_map::valid() const
{
    return 0 != _data;
}

uint8*const
scoped_buffer_map::data_ptr() const
{
    return _data;
}

} // namespace gl
} // namespace scm
