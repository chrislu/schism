
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "wavefront_obj.h"

#include <cassert>
#include <iostream>

#include <boost/assign/list_of.hpp>

#include <scm/core/memory.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>

#include <scm/gl_util/primitives/util/wavefront_obj_file.h>
#include <scm/gl_util/primitives/util/wavefront_obj_loader.h>
#include <scm/gl_util/primitives/util/wavefront_obj_to_vertex_array.h>

namespace scm {
namespace gl {

wavefront_obj_geometry::wavefront_obj_geometry(const render_device_ptr& in_device,
                                               const std::string& in_obj_file)
  : geometry(in_device)
{
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;


    util::wavefront_model obj_f;

    if (!util::open_obj_file(in_obj_file, obj_f)) {
        std::cout << "failed to parse obj file: " << in_obj_file << std::endl;
    }
    else {
        std::cout << "done parsing obj file: " << in_obj_file << std::endl;
    }

    util::vertexbuffer_data obj_vbuf;

    if (!util::generate_vertex_buffer(obj_f, obj_vbuf, true)) {
        std::cout << "failed to generate vertex buffer for: " << in_obj_file << std::endl;
    }
    else {
        std::cout << "done generating vertex buffer data file: " << in_obj_file << std::endl;
    }


    // vertex_buffer
    // position
    unsigned v_size = sizeof(vec3f);
    // normal
    if (obj_vbuf._normals_offset) {
        v_size += sizeof(vec3f);
    }
    // texcoord
    if (obj_vbuf._texcoords_offset) {
        v_size += sizeof(vec2f);
    }

    vertex_format v_fmt = vertex_format(0, 0, TYPE_VEC3F, v_size);
    if (obj_vbuf._normals_offset) {
        v_fmt(0, 1, TYPE_VEC3F, v_size);
    }
    // texcoord
    if (obj_vbuf._texcoords_offset) {
        v_fmt(0, 2, TYPE_VEC2F, v_size);
    }
    scm::size_t vb_size = v_size * obj_vbuf._vert_array_count;
    
    _vertex_buffer = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, vb_size, obj_vbuf._vert_array.get());
    _vertex_array  = in_device->create_vertex_array(v_fmt, list_of(_vertex_buffer));


    scm::size_t next_start_index = 0;
    for (scm::size_t i = 0; i < obj_vbuf._index_array_counts.size(); ++i) {
        const util::wavefront_material& mat = obj_vbuf._materials[i];
        if (obj_vbuf._materials[i]._d < 0.99f) {
            _transparent_object_start_indices.push_back(static_cast<int>(next_start_index));
            _transparent_object_indices_count.push_back(static_cast<int>(obj_vbuf._index_array_counts[i]));
            _transparent_object_materials.push_back(material());
            material& cur_mat  = _transparent_object_materials.back();
            cur_mat._diffuse   = math::vec3f(mat._Kd);
            cur_mat._specular  = math::vec3f(mat._Ks);
            cur_mat._ambient   = math::vec3f(mat._Ka);
            cur_mat._opacity   = mat._d;
            cur_mat._shininess = mat._Ns;
        }
        else {
            _opaque_object_start_indices.push_back(static_cast<int>(next_start_index));
            _opaque_object_indices_count.push_back(static_cast<int>(obj_vbuf._index_array_counts[i]));
            _opaque_object_materials.push_back(material());
            material& cur_mat  = _opaque_object_materials.back();
            cur_mat._diffuse   = math::vec3f(mat._Kd);
            cur_mat._specular  = math::vec3f(mat._Ks);
            cur_mat._ambient   = math::vec3f(mat._Ka);
            cur_mat._opacity   = mat._d;
            cur_mat._shininess = mat._Ns;
        }
        next_start_index += obj_vbuf._index_array_counts[i];
    }
    //assert((_transparent_object_start_indices.size() + _opaque_object_start_indices.size()) == _object_indices_count.size());

    if ( obj_vbuf._vert_array_count < (1 << 16)) {
        _index_type = TYPE_USHORT;

        scoped_array<unsigned short>    ind(new unsigned short[next_start_index]);
        scm::size_t next_index = 0;
        for (scm::size_t a = 0; a < obj_vbuf._index_arrays.size(); ++a) {
            for (scm::size_t i = 0; i < obj_vbuf._index_array_counts[a]; ++i) {
                ind[next_index + i] = static_cast<unsigned short>(obj_vbuf._index_arrays[a][i]);
            }
            next_index += obj_vbuf._index_array_counts[a];
        }

        _index_buffer = in_device->create_buffer(BIND_INDEX_BUFFER, USAGE_STATIC_DRAW, next_start_index * sizeof(unsigned short), ind.get());
    }
    else {
        _index_type = TYPE_UINT;

        scoped_array<unsigned>    ind(new unsigned[next_start_index]);
        scm::size_t next_index = 0;
        for (scm::size_t a = 0; a < obj_vbuf._index_arrays.size(); ++a) {
            for (scm::size_t i = 0; i < obj_vbuf._index_array_counts[a]; ++i) {
                ind[next_index + i] = obj_vbuf._index_arrays[a][i];
            }
            next_index += obj_vbuf._index_array_counts[a];
        }

        _index_buffer = in_device->create_buffer(BIND_INDEX_BUFFER, USAGE_STATIC_DRAW, next_start_index * sizeof(unsigned), ind.get());
    }

    _no_blend_state = in_device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _alpha_blend    = in_device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);

    assert(_vertex_buffer->ok());
    assert(_index_buffer->ok());
    assert(_vertex_array->ok());
}

wavefront_obj_geometry::~wavefront_obj_geometry()
{
    _vertex_buffer.reset();
    _index_buffer.reset();
    _vertex_array.reset();
}

void
wavefront_obj_geometry::draw(const render_context_ptr& in_context,
                             const draw_mode in_draw_mode) const
{
    context_vertex_input_guard  cvg(in_context);
    context_state_objects_guard csg(in_context);

    in_context->bind_vertex_array(_vertex_array);

    if (in_draw_mode == MODE_SOLID) {

        in_context->bind_index_buffer(_index_buffer, PRIMITIVE_TRIANGLE_LIST, _index_type);

        in_context->set_blend_state(_no_blend_state);
        for (scm::size_t i = 0; i < _opaque_object_start_indices.size(); ++i) {
            const material& m = _opaque_object_materials[i];
            program_ptr p = in_context->current_program();
            p->uniform("material_diffuse",   m._diffuse);
            p->uniform("material_specular",  m._specular);
            p->uniform("material_ambient",   m._ambient);
            p->uniform("material_shininess", m._shininess);
            p->uniform("material_opacity",   m._opacity);
            in_context->apply();
            in_context->draw_elements(_opaque_object_indices_count[i], _opaque_object_start_indices[i]);
        }

        in_context->set_blend_state(_alpha_blend);

        for (scm::size_t i = 0; i < _transparent_object_start_indices.size(); ++i) {
            const material& m = _transparent_object_materials[i];
            program_ptr p = in_context->current_program();
            p->uniform("material_diffuse",   m._diffuse);
            p->uniform("material_specular",  m._specular);
            p->uniform("material_ambient",   m._ambient);
            p->uniform("material_shininess", m._shininess);
            p->uniform("material_opacity",   m._opacity);
            in_context->apply();
            in_context->draw_elements(_transparent_object_indices_count[i], _transparent_object_start_indices[i]);
        }
    }
}

void
wavefront_obj_geometry::draw_raw(const render_context_ptr& in_context,
                                 const draw_mode in_draw_mode) const
{
    context_vertex_input_guard  cvg(in_context);
    context_state_objects_guard csg(in_context);

    in_context->bind_vertex_array(_vertex_array);

    if (in_draw_mode == MODE_SOLID) {

        in_context->bind_index_buffer(_index_buffer, PRIMITIVE_TRIANGLE_LIST, _index_type);

        in_context->apply();
        for (scm::size_t i = 0; i < _opaque_object_start_indices.size(); ++i) {
            in_context->draw_elements(_opaque_object_indices_count[i], _opaque_object_start_indices[i]);
        }
        for (scm::size_t i = 0; i < _transparent_object_start_indices.size(); ++i) {
            in_context->draw_elements(_transparent_object_indices_count[i], _transparent_object_start_indices[i]);
        }
    }
}

const buffer_ptr&
wavefront_obj_geometry::vertex_buffer() const
{
    return _vertex_buffer;
}

const buffer_ptr&
wavefront_obj_geometry::index_buffer() const
{
    return _index_buffer;
}

const vertex_array_ptr&
wavefront_obj_geometry::vertex_array() const
{
    return _vertex_array;
}

} // namespace gl
} // namespace scm
