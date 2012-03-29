
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "buffer_object.h"

#include <cassert>

#include <scm/gl_classic/opengl.h>

namespace scm {
namespace gl_classic {

buffer_object::binding_guard::binding_guard(unsigned target, unsigned binding)
  : _binding(binding),
    _save(0),
    _target(target)
{
    glGetIntegerv(_binding, &_save);
}

buffer_object::binding_guard::~binding_guard()
{
    glBindBuffer(_target, _save);
}

buffer_object::buffer_object(unsigned target, unsigned binding)
  : _target(target),
    _target_binding(binding),
    _id(0)
{
}

buffer_object::~buffer_object()
{
    delete_buffer();
}

void buffer_object::bind() const
{
    assert(_id != 0);
    glBindBuffer(_target, _id);
}

void buffer_object::unbind() const
{
    glBindBuffer(_target, 0);
}

bool buffer_object::reset()
{
    delete_buffer();

    return (generate_buffer());
}

void buffer_object::clear()
{
    delete_buffer();
}

bool buffer_object::buffer_data(std::size_t size, const void* data, unsigned usage)
{
    binding_guard guard(_target, _target_binding);
    bind();

    glBufferData(_target, size, data, usage);

    if (glGetError() != GL_NO_ERROR) {
        return (false);
    }
    else {
        return (true);
    }
}

bool buffer_object::generate_buffer()
{
    glGenBuffers(1, &_id);

    return (_id != 0);
}

void buffer_object::delete_buffer()
{
    if (_id != 0) {
        glDeleteBuffers(1, &_id);
        _id = 0;
    }
}

} // namespace gl_classic
} // namespace scm
