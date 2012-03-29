
#ifndef SCM_LDATA_PATCH_GRID_MESH_H_INCLUDED
#define SCM_LDATA_PATCH_GRID_MESH_H_INCLUDED

#include <scm/core/math.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>

#include <renderers/renderers_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class /*__scm_export(large_data)*/ patch_grid_mesh
{
public:
    enum mesh_mode {
        MODE_QUAD_PATCHES       = 0x00,
        MODE_TRIANGLE_PATCHES
    };
public:
    patch_grid_mesh(const gl::render_device_ptr& device,
                    const math::vec2ui&          full_resolution,
                    const math::vec2ui&          cell_size,
                    const math::vec2f&           mesh_extends);
    virtual ~patch_grid_mesh();

    void                    draw(const gl::render_context_ptr& context,
                                 const mesh_mode               mode);

protected:
    math::vec2ui            _grid_resolution;
    math::vec2ui            _full_resolution;
    math::vec2f             _mesh_extends;

    gl::buffer_ptr          _vertices;
    gl::vertex_array_ptr    _vertex_array;

    gl::buffer_ptr          _quad_indices;
    int                     _quad_vertex_count;
    gl::primitive_topology  _quad_prim_topology;

    gl::buffer_ptr          _triangle_indices;
    int                     _triangle_vertex_count;
    gl::primitive_topology  _triangle_prim_topology;

}; // patch_grid_mesh

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_LDATA_PATCH_GRID_MESH_H_INCLUDED
