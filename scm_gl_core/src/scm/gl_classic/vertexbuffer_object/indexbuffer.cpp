
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "indexbuffer.h"

#include <cassert>

#include <scm/core/utilities/foreach.h>

namespace {

} // namespace

namespace scm {
namespace gl_classic {

indexbuffer::indexbuffer()
  : _num_indices(0)
{
}

indexbuffer::~indexbuffer()
{
    clear();
}

void indexbuffer::bind() const
{
    _indices.bind();
}

void indexbuffer::draw_elements() const
{
    glDrawElements(_element_layout._primitive_type,
                   static_cast<GLsizei>(_num_indices),
                   _element_layout._type,
                   NULL);
}

void indexbuffer::unbind() const
{
    _indices.unbind();
}

bool indexbuffer::element_data(std::size_t            num_indices,
                               const element_layout&  layout,
                               const void*const       data,
                               unsigned               usage_type)
{
    if (!_indices.reset()) {
        return (false);
    }

    if (!_indices.buffer_data(num_indices * sizeof(unsigned),//layout._size,
                              data,
                              usage_type)) {
        return (false);
    }

    _element_layout = layout;
    _num_indices    = num_indices;

    return (true);
}

void indexbuffer::clear()
{
    _indices.clear();

    _element_layout = element_layout();

    _num_indices    = 0;
}

std::size_t indexbuffer::num_indices() const
{
    return (_num_indices);
}

const element_layout& indexbuffer::get_element_layout() const
{
    return (_element_layout);
}

} // namespace gl_classic
} // namespace scm
