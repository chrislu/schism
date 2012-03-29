
#ifndef SCM_LARGE_DATA_VOLUME_DATA_H_INCLUDED
#define SCM_LARGE_DATA_VOLUME_DATA_H_INCLUDED

#include <string>
#include <vector>

#include <scm/core/math.h>

#include <scm/data/analysis/transfer_function/piecewise_function_1d.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/buffer_objects/uniform_buffer_adaptor.h>
#include <scm/gl_core/constants.h>
#include <scm/gl_core/primitives/box.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/viewer/viewer_fwd.h>

#include <renderer/renderer_fwd.h>

namespace scm {
namespace data {

class volume_data
{
public:
    typedef piecewise_function_1d<float, math::vec3f>   color_map_type;
    typedef piecewise_function_1d<float, float>         alpha_map_type;

    typedef shared_ptr<color_map_type>                  color_map_ptr;
    typedef shared_ptr<alpha_map_type>                  alpha_map_ptr;

    struct volume_uniform_data
    {
        math::vec4f _volume_extends;     // w unused
        math::vec4f _scale_obj_to_tex;   // w unused
        math::vec4f _sampling_distance;  // yzw unused
        math::vec4f _os_camera_position;
        math::vec4f _value_range;

        math::mat4f _m_matrix;
        math::mat4f _m_matrix_inverse;
        math::mat4f _m_matrix_inverse_transpose;

        math::mat4f _mv_matrix;
        math::mat4f _mv_matrix_inverse;
        math::mat4f _mv_matrix_inverse_transpose;

        math::mat4f _mvp_matrix;
        math::mat4f _mvp_matrix_inverse;
    }; // struct volume_uniform_data

    typedef gl::uniform_block<volume_uniform_data>  volume_uniform_block;

public:
    volume_data(const gl::render_device_ptr& device,
                const std::string&           file_name,
                const color_map_type&        cmap,
                const alpha_map_type&        amap);
    virtual ~volume_data();

    const math::vec3f&                  extends() const;
    const math::vec3ui&                 data_dimensions() const;
    const math::mat4f&                  transform() const;
    void                                transform(const math::mat4f& m);

    float                               sample_distance() const;
    float                               sample_distance_factor() const;
    void                                sample_distance_factor(float d);

    float                               sample_distance_ref() const;
    float                               sample_distance_ref_factor() const;
    void                                sample_distance_ref_factor(float d);

    float                               min_value() const;
    float                               max_value() const;

    float                               selected_lod() const;
    void                                selected_lod(float l);

    const gl::box&                      bbox() const;
    const gl::box_volume_geometry_ptr&  bbox_geometry() const;

    const gl::texture_3d_ptr&           volume_raw() const;
    const gl::texture_1d_ptr&           color_alpha_map() const;

    const color_map_ptr&                color_map() const;
    const alpha_map_ptr&                alpha_map() const;
    void                                update_color_alpha_maps();

    void                                update(const gl::render_context_ptr& context,
                                               const gl::camera&             cam);
    const volume_uniform_block&         volume_block() const;

protected:
    gl::texture_3d_ptr                  load_volume(const gl::render_device_ptr& in_device,
                                                    const std::string&           in_file_name);
    gl::texture_1d_ptr                  create_color_alpha_map(const gl::render_device_ptr& in_device,
                                                                     unsigned               in_size) const;
    bool                                update_color_alpha_map(const gl::render_context_ptr& context) const;

protected:
    math::vec3f                         _extends;
    math::vec3ui                        _data_dimensions;
    math::mat4f                         _transform;

    float                               _sample_distance;
    float                               _sample_distance_ref;

    float                               _selected_lod;
    float                               _max_lod;

    float                               _min_value;
    float                               _max_value;

    color_map_ptr                       _color_map;
    alpha_map_ptr                       _alpha_map;

    gl::box                             _bbox;
    gl::box_volume_geometry_ptr         _bbox_geometry;

    volume_uniform_block                _volume_block;

    gl::texture_3d_ptr                  _volume_raw;
    gl::texture_1d_ptr                  _color_alpha_map;
    bool                                _color_alpha_map_dirty;

}; // volume_data

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_VOLUME_DATA_H_INCLUDED
