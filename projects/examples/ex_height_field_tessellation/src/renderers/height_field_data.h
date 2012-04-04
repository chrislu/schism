
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LDATA_HEIGHT_FIELD_DATA_H_INCLUDED
#define SCM_LDATA_HEIGHT_FIELD_DATA_H_INCLUDED

#include <string>

#include <scm/core/math.h>

#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>
#include <scm/gl_core/primitives/box.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/data/imaging/imaging_fwd.h>

#include <renderers/renderers_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class /*__scm_export(large_data*/ height_field_data
{
public:
    typedef piecewise_function_1d<float, math::vec3f>   color_transfer_type;

public:
    height_field_data(const gl::render_device_ptr& device,
                      const std::string&           file_name,
                      const math::vec3f&           height_field_extends);
    virtual ~height_field_data();

    const math::vec3f&                  extends() const;
    const math::mat4f&                  transform() const;
    void                                transform(const math::mat4f& m);

    const gl::box&                      bbox() const;
    const gl::box_geometry_ptr&         bbox_geometry() const;

    const gl::texture_2d_ptr&           height_map() const;
    const gl::texture_2d_ptr&           density_map() const;
    const gl::texture_1d_ptr&           color_map() const;
    const gl::texture_buffer_ptr&       quad_edge_density_buffer() const;
    const gl::texture_buffer_ptr&       triangle_edge_density_buffer() const;

    const data::patch_grid_mesh_ptr&    patch_mesh() const;


protected:
    gl::texture_image_data_ptr          generate_density_data(const gl::texture_image_data_ptr& src_image,
                                                              const math::vec2ui&               patch_size,
                                                              const math::vec3f&                height_field_extends) const;
    gl::texture_image_data_ptr          pad_to_patch_size(const gl::texture_image_data_ptr& src_image,
                                                          const math::vec2ui&               patch_size) const;
    gl::texture_1d_ptr                  create_color_map(gl::render_device& in_device,
                                                         unsigned in_size,
                                                         const color_transfer_type& in_color) const;

protected:
    math::vec3f                         _extends;
    math::mat4f                         _transform;

    color_transfer_type                 _color_transfer;

    gl::box                             _bbox;
    gl::box_geometry_ptr                _bbox_geometry;

    gl::texture_2d_ptr                  _height_map;
    gl::texture_2d_ptr                  _density_map;
    gl::texture_1d_ptr                  _color_map;
    gl::texture_buffer_ptr              _quad_edge_density_buffer;
    gl::texture_buffer_ptr              _triangle_edge_density_buffer;
    data::patch_grid_mesh_ptr           _patch_mesh;

}; // height_field_data

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_LDATA_HEIGHT_FIELD_DATA_H_INCLUDED
