
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_WAVEFRONT_OBJ_H_INCLUDED
#define SCM_GL_UTIL_WAVEFRONT_OBJ_H_INCLUDED

#include <string>
#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/state_objects.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/geometry.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) wavefront_obj_geometry : public geometry
{
    struct material {
        math::vec3f     _diffuse;
        math::vec3f     _specular;
        math::vec3f     _ambient;
        float           _opacity;
        float           _shininess;
    }; // struct material
public:
    wavefront_obj_geometry(const render_device_ptr& in_device, const std::string& in_obj_file);
    virtual ~wavefront_obj_geometry();

    void                draw(const render_context_ptr& in_context,
                             const draw_mode           in_draw_mode = MODE_SOLID) const;
    void                draw_raw(const render_context_ptr& in_context,
                                 const draw_mode           in_draw_mode = MODE_SOLID) const;

    const buffer_ptr&       vertex_buffer() const;
    const buffer_ptr&       index_buffer() const;
    const vertex_array_ptr& vertex_array() const;

protected:
    buffer_ptr              _vertex_buffer;
    buffer_ptr              _index_buffer;
    data_type               _index_type;
    std::vector<int>        _opaque_object_start_indices;
    std::vector<int>        _opaque_object_indices_count;
    std::vector<int>        _transparent_object_start_indices;
    std::vector<int>        _transparent_object_indices_count;
    std::vector<material>   _opaque_object_materials;
    std::vector<material>   _transparent_object_materials;
    vertex_array_ptr        _vertex_array;
    blend_state_ptr         _no_blend_state;
    blend_state_ptr         _alpha_blend;

}; // class box_geometry

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WAVEFRONT_OBJ_H_INCLUDED
