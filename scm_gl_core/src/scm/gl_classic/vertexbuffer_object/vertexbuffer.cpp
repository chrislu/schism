
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "vertexbuffer.h"

#include <cassert>

#include <scm/core/utilities/foreach.h>

namespace {

} // namespace

namespace scm {
namespace gl_classic {

vertexbuffer::vertexbuffer()
  : _num_vertices(0)
{
}

vertexbuffer::~vertexbuffer()
{
    clear();
}

void vertexbuffer::bind() const
{
    ////// ATTENTION no interlaced support yet!!!!!!!
    _vertices.bind();

    for (unsigned i = 0; i < _vertex_layout.num_elements(); ++i) {

        const vertex_element& elem = _vertex_layout.element(i);

        switch (elem._binding) {
            case vertex_element::position:
                glVertexPointer(static_cast<GLint>(elem._data_num_components),
                                elem._data_elem_type,
                                0,
                                (GLvoid*)(0 + _vertex_layout.offset(i) * _num_vertices));
                glEnableClientState(GL_VERTEX_ARRAY);
                break;
            case vertex_element::normal:
                assert(elem._data_num_components == 3);

                glNormalPointer(elem._data_elem_type,
                                0,
                                (GLvoid*)(0 + _vertex_layout.offset(i) * _num_vertices));
                glEnableClientState(GL_NORMAL_ARRAY);
                break;
            case vertex_element::tex_coord:
                if (elem._num != 0) assert (0);

                glTexCoordPointer(static_cast<GLint>(elem._data_num_components),
                                  elem._data_elem_type,
                                  0,
                                  (GLvoid*)(0 + _vertex_layout.offset(i) * _num_vertices));
                glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                break;
            case vertex_element::color:
                assert(0);
                break;
            case vertex_element::vertex_attrib:
                assert(0);
                break;
        }
    }
}

void vertexbuffer::unbind() const
{
    for (unsigned i = 0; i < _vertex_layout.num_elements(); ++i) {

        const vertex_element& elem = _vertex_layout.element(i);

        switch (elem._binding) {
            case vertex_element::position:
                glDisableClientState(GL_VERTEX_ARRAY);
                break;
            case vertex_element::normal:
                glDisableClientState(GL_NORMAL_ARRAY);
                break;
            case vertex_element::tex_coord:;break;
                if (elem._num != 0) assert (0);
                glDisableClientState(GL_TEXTURE_COORD_ARRAY);
                break;
            case vertex_element::color:
                assert(0);
                break;
            case vertex_element::vertex_attrib:
                assert(0);
                break;
        }
    }

    _vertices.unbind();
}

bool vertexbuffer::vertex_data(std::size_t           num_vertices,
                               const vertex_layout&  layout,
                               const void*const      data,
                               unsigned              usage_type)
{
    if (!_vertices.reset()) {
        return (false);
    }

    if (!_vertices.buffer_data(num_vertices * layout.size(),
                               data,
                               usage_type)) {
        return (false);
    }

    _vertex_layout  = layout;
    _num_vertices   = num_vertices;

    return (true);
}

void vertexbuffer::clear()
{
    _vertices.clear();

    _vertex_layout  = vertex_layout();

    _num_vertices   = 0;
}

std::size_t vertexbuffer::num_vertices() const
{
    return (_num_vertices);
}

const vertex_layout& vertexbuffer::get_vertex_layout() const
{
    return (_vertex_layout);
}

} // namespace gl_classic
} // namespace scm
