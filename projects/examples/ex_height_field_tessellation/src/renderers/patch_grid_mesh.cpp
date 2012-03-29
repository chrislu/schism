
#include "patch_grid_mesh.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>
#include <scm/core/memory.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>

namespace {

struct vertex {
    scm::math::vec3f pos;
    scm::math::vec2f tex_hf; // height map coord
    scm::math::vec2f tex_dm; // density map coord
}; //  struct vertex

} // namespace

namespace scm {
namespace data {

patch_grid_mesh::patch_grid_mesh(const gl::render_device_ptr& device,
                                 const math::vec2ui&          full_resolution,
                                 const math::vec2ui&          cell_size,
                                 const math::vec2f&           mesh_extends)
  : _full_resolution(full_resolution)
  , _mesh_extends(mesh_extends)
{
    using namespace scm::math;
    using namespace scm::gl;
    using boost::assign::list_of;

    // neighboring cells share texels, so we have to account for the 1 texel overlap
    _grid_resolution = vec2ui(floor(vec2f(_full_resolution) / vec2f(cell_size - 1)));
    //out() << "_full_resolution: " << _full_resolution << log::end;
    //out() << "cell_size: " << cell_size << log::end;
    //out() << "_grid_resolution: " << _grid_resolution << log::end;
    //out() << "_mesh_extends: " << _mesh_extends << log::end;

    size_t num_vertices     = (_grid_resolution.x + 1) * (_grid_resolution.y + 1);
    size_t num_quad_patches = _grid_resolution.x * _grid_resolution.y;
    size_t num_tri_patches  = _grid_resolution.x * _grid_resolution.y * 2;

    _vertices         = device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vertex), 0);
    _quad_indices     = device->create_buffer(BIND_INDEX_BUFFER,  USAGE_STATIC_DRAW, num_quad_patches * 4 * sizeof(unsigned), 0);
    _triangle_indices = device->create_buffer(BIND_INDEX_BUFFER,  USAGE_STATIC_DRAW, num_tri_patches * 3 * sizeof(unsigned), 0);

    if (   !_vertices
        || !_quad_indices) {
        throw (std::runtime_error("patch_grid_mesh::patch_grid_mesh(): unable to create vertex or index buffer."));
    }

    render_context_ptr ctx = device->main_context();
   
    { // initialize vertices
        vertex* data = static_cast<vertex*>(ctx->map_buffer(_vertices, ACCESS_WRITE_INVALIDATE_BUFFER));

        if (!data) {
            throw (std::runtime_error("patch_grid_mesh::patch_grid_mesh(): unable map vertex buffer."));
        }

        const vec2f dm_extends   = vec2f(_full_resolution) / (vec2f(_grid_resolution) * (cell_size - 1));

        const vec2f pos_delta    = (_mesh_extends * vec2f(cell_size - 1)) / _full_resolution;//_grid_dimension / _grid_resolution;
        const vec2f hf_tex_delta = (vec2f(1.0f)   * vec2f(cell_size - 1)) / _full_resolution;//vec2f(1.0, 1.0f) / _grid_resolution;
        const vec2f dm_tex_delta = (dm_extends    * vec2f(cell_size - 1)) / _full_resolution;//vec2f(1.0, 1.0f) / _grid_resolution;

        for(unsigned y = 0; y < (_grid_resolution.y + 1); ++y){
            for(unsigned x = 0; x < (_grid_resolution.x + 1); ++x){
                data[x + y * (_grid_resolution.x + 1)].pos    = vec3f(pos_delta.x * x,    pos_delta.y * y, 0.0f);
                data[x + y * (_grid_resolution.x + 1)].tex_hf = vec2f(hf_tex_delta.x * x, hf_tex_delta.y * y) + vec2f(0.5f) / _full_resolution;
                data[x + y * (_grid_resolution.x + 1)].tex_dm = vec2f(dm_tex_delta.x * x, dm_tex_delta.y * y) + vec2f(0.5f) / _full_resolution;
            }
        }
        ctx->unmap_buffer(_vertices);
    }

    _vertex_array = device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F, sizeof(vertex))  // vertex positions
                                                             (0, 2, TYPE_VEC2F, sizeof(vertex))  // texture coordinates height  map
                                                             (0, 3, TYPE_VEC2F, sizeof(vertex)), // texture coordinates density map
                                                list_of(_vertices));

    { // initialize quad list index buffer
        unsigned int* data = static_cast<unsigned int*>(ctx->map_buffer(_quad_indices, ACCESS_WRITE_INVALIDATE_BUFFER));

        if (!data) {
            throw (std::runtime_error("patch_grid_mesh::patch_grid_mesh(): unable map quad index buffer."));
        }

        for(unsigned y = 0; y < _grid_resolution.y; ++y){
            for(unsigned x = 0; x < _grid_resolution.x; ++x){
                unsigned i00 =  x      + y       * (_grid_resolution.x + 1);
                unsigned i10 = (x + 1) + y       * (_grid_resolution.x + 1);
                unsigned i11 = (x + 1) + (y + 1) * (_grid_resolution.x + 1);
                unsigned i01 =  x      + (y + 1) * (_grid_resolution.x + 1);

                size_t off = 4 * (x + y * _grid_resolution.x);

                data[off + 0] = i00;
                data[off + 1] = i10;
                data[off + 2] = i11;
                data[off + 3] = i01;
            }
        }
        ctx->unmap_buffer(_quad_indices);

        _quad_prim_topology = PRIMITIVE_PATCH_LIST_4_CONTROL_POINTS;
        _quad_vertex_count  = 4 * _grid_resolution.x * _grid_resolution.y;
    }
    { // initialize triangle list index buffer
        unsigned int* data = static_cast<unsigned int*>(ctx->map_buffer(_triangle_indices, ACCESS_WRITE_INVALIDATE_BUFFER));

        if (!data) {
            throw (std::runtime_error("patch_grid_mesh::patch_grid_mesh(): unable map triangle index buffer."));
        }

        for(unsigned y = 0; y < _grid_resolution.y; ++y){
            for(unsigned x = 0; x < _grid_resolution.x; ++x){
                unsigned i00 =  x      + y       * (_grid_resolution.x + 1);
                unsigned i10 = (x + 1) + y       * (_grid_resolution.x + 1);
                unsigned i11 = (x + 1) + (y + 1) * (_grid_resolution.x + 1);
                unsigned i01 =  x      + (y + 1) * (_grid_resolution.x + 1);

                size_t off = 6 * (x + y * _grid_resolution.x);

                data[off + 0] = i00;
                data[off + 1] = i01;
                data[off + 2] = i10;

                data[off + 3] = i10;
                data[off + 4] = i01;
                data[off + 5] = i11;
            }
        }
        ctx->unmap_buffer(_triangle_indices);

        _triangle_prim_topology = PRIMITIVE_PATCH_LIST_3_CONTROL_POINTS;
        _triangle_vertex_count  = 6 * _grid_resolution.x * _grid_resolution.y;
    }
}

patch_grid_mesh::~patch_grid_mesh()
{
    _vertices.reset();
    _quad_indices.reset();
    _quad_indices.reset();
    _triangle_indices.reset();
}

void
patch_grid_mesh::draw(const gl::render_context_ptr& context,
                      const mesh_mode               mode)
{
    using namespace scm::gl;
    context_vertex_input_guard vig(context);

    if (mode == MODE_QUAD_PATCHES) {
        context->bind_vertex_array(_vertex_array);
        context->bind_index_buffer(_quad_indices, _quad_prim_topology, TYPE_UINT);

        context->apply();
        context->draw_elements(_quad_vertex_count);
    }
    else if (mode == MODE_TRIANGLE_PATCHES) {
        context->bind_vertex_array(_vertex_array);
        context->bind_index_buffer(_triangle_indices, _triangle_prim_topology, TYPE_UINT);

        context->apply();
        context->draw_elements(_triangle_vertex_count);
    }
    else {
        err() << log::warning << "patch_grid_mesh::draw(): unknown mesh mode." << log::end;
    }
}

} // namespace data
} // namespace scm
