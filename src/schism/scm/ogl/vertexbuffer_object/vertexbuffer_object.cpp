
#include "vertexbuffer_object.h"

#include <cassert>

#include <scm/core/utilities/foreach.h>

namespace {

} // namespace

namespace scm {
namespace gl {

vertexbuffer_object::vertexbuffer_object()
  : _num_vertices(0),
    _num_indices(0)
{
}

vertexbuffer_object::~vertexbuffer_object()
{
    clear();
}

void vertexbuffer_object::bind() const
{
    ////// ATTENTION no interlaced support yet!!!!!!!
    _vertices.bind();
    _indices.bind();

    for (unsigned i = 0; i < _vertex_layout.num_elements(); ++i) {

        const vertex_element& elem = _vertex_layout.element(i);

        switch (elem._binding) {
            case vertex_element::position:
                glVertexPointer(static_cast<GLint>(elem._data_num_components),
                                elem._data_elem_type,
                                0,
                                (GLvoid*)(NULL + _vertex_layout.offset(i) * _num_vertices));
                glEnableClientState(GL_VERTEX_ARRAY);
                break;
            case vertex_element::normal:
                assert(elem._data_num_components == 3);

                glNormalPointer(elem._data_elem_type,
                                0,
                                (GLvoid*)(NULL + _vertex_layout.offset(i) * _num_vertices));
                glEnableClientState(GL_NORMAL_ARRAY);
                break;
            case vertex_element::tex_coord:;break;
                if (elem._num != 0) assert (0);

                glTexCoordPointer(static_cast<GLint>(elem._data_num_components),
                                  elem._data_elem_type,
                                  0,
                                  (GLvoid*)(NULL + _vertex_layout.offset(i) * _num_vertices));
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

void vertexbuffer_object::draw_elements() const
{
    glDrawElements(_element_layout._primitive_type,
                   static_cast<GLsizei>(_num_indices),
                   _element_layout._type,
                   NULL);
}

void vertexbuffer_object::unbind() const
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
    _indices.unbind();
}

bool vertexbuffer_object::vertex_data(std::size_t           num_vertices,
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

bool vertexbuffer_object::element_data(std::size_t            num_indices,
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

void vertexbuffer_object::clear()
{
    _vertices.clear();
    _indices.clear();

    _element_layout = element_layout();
    _vertex_layout  = vertex_layout();

    _num_vertices   = 0;
    _num_indices    = 0;
}

std::size_t vertexbuffer_object::num_vertices() const
{
    return (_num_vertices);
}

std::size_t vertexbuffer_object::num_indices() const
{
    return (_num_indices);
}

const vertex_layout& vertexbuffer_object::get_vertex_layout() const
{
    return (_vertex_layout);
}

const element_layout& vertexbuffer_object::get_element_layout() const
{
    return (_element_layout);
}

} // namespace gl
} // namespace scm
