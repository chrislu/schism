
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DEVICE_H_INCLUDED
#define SCM_GL_CORE_DEVICE_H_INCLUDED

#include <iosfwd>
#include <limits>
#include <list>
#include <set>
#include <utility>
#include <vector>

#include <boost/noncopyable.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <scm/config.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>
#include <scm/gl_core/shader_objects/shader_macro.h>
#include <scm/gl_core/state_objects/blend_state.h>
#include <scm/gl_core/state_objects/depth_stencil_state.h>
#include <scm/gl_core/state_objects/rasterizer_state.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

//#include <scm/cl_core/cl_core_fwd.h>


namespace scm {
namespace cu {

class cuda_device;

typedef scm::shared_ptr<cuda_device>        cuda_device_ptr;
typedef scm::shared_ptr<cuda_device const>  cuda_device_cptr;

class cuda_command_stream;

typedef scm::shared_ptr<cuda_command_stream>        cuda_command_stream_ptr;
typedef scm::shared_ptr<cuda_command_stream const>  cuda_command_stream_cptr;

} // namespace cu

namespace cl {

class opencl_device;

typedef scm::shared_ptr<opencl_device>        opencl_device_ptr;
typedef scm::shared_ptr<opencl_device const>  opencl_device_cptr;

} // namespace cl

namespace gl {

namespace opengl {
class gl_core;
} // namespace detail

class __scm_export(gl_core) render_device : boost::noncopyable
{
////// types //////////////////////////////////////////////////////////////////////////////////////
public:
    struct device_capabilities {
        int             _max_vertex_attributes;
        int             _max_draw_buffers;
        int             _max_dual_source_draw_buffers;
        int             _max_texture_size;
        int             _max_texture_3d_size;
        int             _max_samples;
        int             _max_array_texture_layers;
        int             _max_depth_texture_samples;
        int             _max_color_texture_samples;
        int             _max_integer_samples;
        int             _max_texture_image_units;
        int             _max_texture_buffer_size;
        int             _max_frame_buffer_color_attachments;
        int             _max_vertex_uniform_blocks;
        int             _max_geometry_uniform_blocks;
        int             _max_fragment_uniform_blocks;
        int             _max_combined_uniform_blocks;
        int             _max_combined_vertex_uniform_components;
        int             _max_combined_geometry_uniform_components;
        int             _max_combined_fragment_uniform_components;
        int             _max_uniform_buffer_bindings;
        int             _max_uniform_block_size;
        int             _uniform_buffer_offset_alignment;
        int             _max_viewports;
        int             _max_transform_feedback_separate_attribs;
        int             _max_transform_feedback_buffers;
        int             _max_vertex_streams;
        int             _max_image_units;
        int             _max_vertex_atomic_counters;
        int             _max_geometry_atomic_counters;
        int             _max_fragment_atomic_counters;
        int             _max_combined_atomic_counters;
        int             _max_atomic_counter_buffer_bindings;
        int             _min_map_buffer_alignment;

        int             _num_program_binary_formats;
        shared_array<int>   _program_binary_formats;

        int             _max_shader_storage_block_bindings;
        int             _max_shader_storage_block_size;
        int64           _shader_storage_buffer_offset_alignment;
    }; // struct device_capabilities

protected:
    typedef boost::unordered_set<render_device_resource*>   resource_ptr_set;

    typedef boost::unordered_map<std::string, shader_macro> shader_macro_map;
    typedef std::set<std::string>                           string_set;

    typedef std::list<shader_ptr>                           shader_list;

    typedef std::vector<buffer_ptr>                         buffer_array;

////// methods ////////////////////////////////////////////////////////////////////////////////////
public:
    render_device();
    virtual ~render_device();

    // device /////////////////////////////////////////////////////////////////////////////////////
    const opengl::gl_core&          opengl_api() const;
    render_context_ptr              main_context() const;
    render_context_ptr              create_context();
    const device_capabilities&      capabilities() const;

    virtual void                    print_device_informations(std::ostream& os) const;
    const std::string               device_vendor() const;
    const std::string               device_renderer() const;
    const std::string               device_shader_compiler() const;
    const std::string               device_context_version() const;

protected:
    void                            init_capabilities();

    void                            register_resource(render_device_resource* res_ptr);
    void                            release_resource(render_device_resource* res_ptr);

    // buffer api /////////////////////////////////////////////////////////////////////////////////
public:
    buffer_ptr                      create_buffer(const buffer_desc& in_buffer_desc,
                                                  const void*        in_initial_data = 0);
    buffer_ptr                      create_buffer(buffer_binding in_binding,
                                                  buffer_usage   in_usage,
                                                  scm::size_t    in_size,
                                                  const void*    in_initial_data = 0);
    bool                            resize_buffer(const buffer_ptr& in_buffer, scm::size_t in_size);

    vertex_array_ptr                create_vertex_array(const vertex_format& in_vert_fmt,
                                                        const buffer_array&  in_attrib_buffers,
                                                        const program_ptr&   in_program = program_ptr());

    transform_feedback_ptr          create_transform_feedback(const stream_output_setup& in_setup);

    // shader api /////////////////////////////////////////////////////////////////////////////////
public:
    bool                            add_include_files(const std::string& in_path,
                                                      const std::string& in_glsl_root_path      = std::string("/"),
                                                      const std::string& in_file_extensions     = std::string(".glsl .glslh"),
                                                      bool               in_scan_subdirectories = true);
    bool                            add_include_string(const std::string& in_path,
                                                       const std::string& in_source_string);

    void                            add_macro_define(const std::string& in_name, const std::string& in_value);
    void                            add_macro_define(const shader_macro& in_macro);
    void                            add_macro_defines(const shader_macro_array& in_macros);

    shader_ptr                      create_shader(shader_stage       in_stage,
                                                  const std::string& in_source,
                                                  const std::string& in_source_name = "");
    shader_ptr                      create_shader(shader_stage              in_stage,
                                                  const std::string&        in_source,
                                                  const shader_macro_array& in_macros,
                                                  const std::string&        in_source_name = "");
    shader_ptr                      create_shader(shader_stage                    in_stage,
                                                  const std::string&              in_source,
                                                  const shader_include_path_list& in_inc_paths,
                                                  const std::string&              in_source_name = "");
    shader_ptr                      create_shader(shader_stage                    in_stage,
                                                  const std::string&              in_source,
                                                  const shader_macro_array&       in_macros,
                                                  const shader_include_path_list& in_inc_paths,
                                                  const std::string&              in_source_name = "");

    shader_ptr                      create_shader_from_file(shader_stage       in_stage,
                                                            const std::string& in_file_name);
    shader_ptr                      create_shader_from_file(shader_stage              in_stage,
                                                            const std::string&        in_source,
                                                            const shader_macro_array& in_macros);
    shader_ptr                      create_shader_from_file(shader_stage                    in_stage,
                                                            const std::string&              in_source,
                                                            const shader_include_path_list& in_inc_paths);
    shader_ptr                      create_shader_from_file(shader_stage                    in_stage,
                                                            const std::string&              in_source,
                                                            const shader_macro_array&       in_macros,
                                                            const shader_include_path_list& in_inc_paths);

    program_ptr                     create_program(const shader_list& in_shaders,
                                                   const std::string& in_program_name = "");
    program_ptr                     create_program(const shader_list&          in_shaders,
                                                   const stream_capture_array& in_capture,
                                                   bool                        in_rasterization_discard = false,
                                                   const std::string&          in_program_name = "");

protected:
    bool                            add_include_string_internal(const std::string& in_path,
                                                                const std::string& in_source_string,
                                                                      bool         lock_thread);

    // texture api ////////////////////////////////////////////////////////////////////////////////
public:
    texture_1d_ptr                  create_texture_1d(const texture_1d_desc&    in_desc);
    texture_1d_ptr                  create_texture_1d(const texture_1d_desc&    in_desc,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_1d_ptr                  create_texture_1d(const unsigned      in_size,
                                                      const data_format   in_format,
                                                      const unsigned      in_mip_levels = 1,
                                                      const unsigned      in_array_layers = 1);
    texture_1d_ptr                  create_texture_1d(const unsigned            in_size,
                                                      const data_format         in_format,
                                                      const unsigned            in_mip_levels,
                                                      const unsigned            in_array_layers,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_1d_ptr                  create_texture_1d(const texture_1d_ptr&     in_orig_texture,
                                                      const data_format         in_format,
                                                      const math::vec2ui&       in_mip_range,
                                                      const math::vec2ui&       in_layer_range);

    texture_2d_ptr                  create_texture_2d(const texture_2d_desc&    in_desc);
    texture_2d_ptr                  create_texture_2d(const texture_2d_desc&    in_desc,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_2d_ptr                  create_texture_2d(const math::vec2ui& in_size,
                                                      const data_format   in_format,
                                                      const unsigned      in_mip_levels = 1,
                                                      const unsigned      in_array_layers = 1,
                                                      const unsigned      in_samples = 1);
    texture_2d_ptr                  create_texture_2d(const math::vec2ui&       in_size,
                                                      const data_format         in_format,
                                                      const unsigned            in_mip_levels,
                                                      const unsigned            in_array_layers,
                                                      const unsigned            in_samples,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_2d_ptr                  create_texture_2d(const texture_2d_ptr&     in_orig_texture,
                                                      const data_format         in_format,
                                                      const math::vec2ui&       in_mip_range,
                                                      const math::vec2ui&       in_layer_range);

    texture_3d_ptr                  create_texture_3d(const texture_3d_desc&    in_desc);
    texture_3d_ptr                  create_texture_3d(const texture_3d_desc&    in_desc,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_3d_ptr                  create_texture_3d(const math::vec3ui& in_size,
                                                      const data_format   in_format,
                                                      const unsigned      in_mip_levels = 1);
    texture_3d_ptr                  create_texture_3d(const math::vec3ui&       in_size,
                                                      const data_format         in_format,
                                                      const unsigned            in_mip_levels,
                                                      const data_format         in_initial_data_format,
                                                      const std::vector<void*>& in_initial_mip_level_data);
    texture_3d_ptr                  create_texture_3d(const texture_3d_ptr&     in_orig_texture,
                                                      const data_format         in_format,
                                                      const math::vec2ui&       in_mip_range);

    texture_cube_ptr                create_texture_cube(const texture_cube_desc& in_desc);
    texture_cube_ptr                create_texture_cube(const texture_cube_desc& in_desc,
                                                        const data_format          in_initial_data_format,
                                                        const std::vector<void*>& in_initial_mip_level_data_px,
                                                        const std::vector<void*>& in_initial_mip_level_data_nx,
                                                        const std::vector<void*>& in_initial_mip_level_data_py,
                                                        const std::vector<void*>& in_initial_mip_level_data_ny,
                                                        const std::vector<void*>& in_initial_mip_level_data_pz,
                                                        const std::vector<void*>& in_initial_mip_level_data_nz);
    texture_cube_ptr                create_texture_cube(const math::vec2ui& in_size,
                                                        const data_format   in_format,
                                                        const unsigned      in_mip_levels = 1);
    texture_cube_ptr                create_texture_cube(const math::vec2ui&      in_size,
                                                        const data_format          in_format,
                                                        const unsigned             in_mip_levels,
                                                        const data_format          in_initial_data_format,
                                                        const std::vector<void*>& in_initial_mip_level_data_px,
                                                        const std::vector<void*>& in_initial_mip_level_data_nx,
                                                        const std::vector<void*>& in_initial_mip_level_data_py,
                                                        const std::vector<void*>& in_initial_mip_level_data_ny,
                                                        const std::vector<void*>& in_initial_mip_level_data_pz,
                                                        const std::vector<void*>& in_initial_mip_level_data_nz);

    texture_buffer_ptr              create_texture_buffer(const texture_buffer_desc& in_desc);
    texture_buffer_ptr              create_texture_buffer(const data_format   in_format,
                                                          const buffer_ptr&   in_buffer);
    texture_buffer_ptr              create_texture_buffer(const data_format   in_format,
                                                          buffer_usage        in_buffer_usage,
                                                          scm::size_t         in_buffer_size,
                                                          const void*         in_buffer_initial_data = 0);

    texture_handle_ptr              create_resident_handle(const texture_ptr&       in_texture,
                                                           const sampler_state_ptr& in_sampler);

    sampler_state_ptr               create_sampler_state(const sampler_state_desc& in_desc);
    sampler_state_ptr               create_sampler_state(texture_filter_mode  in_filter1,
                                                         texture_wrap_mode    in_wrap,
                                                         unsigned             in_max_anisotropy = 1,
                                                         float                in_min_lod = -(std::numeric_limits<float>::max)(),
                                                         float                in_max_lod = (std::numeric_limits<float>::max)(),
                                                         float                in_lod_bias = 0.0f,
                                                         compare_func         in_compare_func = COMPARISON_LESS_EQUAL,
                                                         texture_compare_mode in_compare_mode = TEXCOMPARE_NONE);
    sampler_state_ptr               create_sampler_state(texture_filter_mode  in_filter,
                                                         texture_wrap_mode    in_wrap_s,
                                                         texture_wrap_mode    in_wrap_t,
                                                         texture_wrap_mode    in_wrap_r,
                                                         unsigned             in_max_anisotropy = 1,
                                                         float                in_min_lod = -(std::numeric_limits<float>::max)(),
                                                         float                in_max_lod = (std::numeric_limits<float>::max)(),
                                                         float                in_lod_bias = 0.0f,
                                                         compare_func         in_compare_func = COMPARISON_LESS_EQUAL,
                                                         texture_compare_mode in_compare_mode = TEXCOMPARE_NONE);

    // frame buffer api ///////////////////////////////////////////////////////////////////////////
    render_buffer_ptr               create_render_buffer(const render_buffer_desc& in_desc);
    render_buffer_ptr               create_render_buffer(const math::vec2ui& in_size,
                                                         const data_format   in_format,
                                                         const unsigned      in_samples = 1);
    frame_buffer_ptr                create_frame_buffer();


    // state api //////////////////////////////////////////////////////////////////////////////////
public:
    depth_stencil_state_ptr         create_depth_stencil_state(const depth_stencil_state_desc& in_desc);
    depth_stencil_state_ptr         create_depth_stencil_state(bool in_depth_test, bool in_depth_mask = true, compare_func in_depth_func = COMPARISON_LESS,
                                                               bool in_stencil_test = false, unsigned in_stencil_rmask = ~0u, unsigned in_stencil_wmask = ~0u,
                                                               stencil_ops in_stencil_ops = stencil_ops());
    depth_stencil_state_ptr         create_depth_stencil_state(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                                               bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                                               stencil_ops in_stencil_front_ops, stencil_ops in_stencil_back_ops);

    rasterizer_state_ptr            create_rasterizer_state(const rasterizer_state_desc& in_desc);
    rasterizer_state_ptr            create_rasterizer_state(fill_mode in_fmode, cull_mode in_cmode = CULL_BACK, polygon_orientation in_fface = ORIENT_CCW,
                                                            bool in_msample = false, bool in_sshading = false, float32 in_min_sshading = 0.0f,
                                                            bool in_sctest = false, bool in_smlines = false,
                                                            const point_raster_state& in_point_state = point_raster_state());

    blend_state_ptr                 create_blend_state(const blend_state_desc& in_desc);
    blend_state_ptr                 create_blend_state(bool in_enabled,
                                                       blend_func in_src_rgb_func,   blend_func in_dst_rgb_func,
                                                       blend_func in_src_alpha_func, blend_func in_dst_alpha_func,
                                                       blend_equation  in_rgb_equation = EQ_FUNC_ADD, blend_equation in_alpha_equation = EQ_FUNC_ADD,
                                                       unsigned in_write_mask = COLOR_ALL, bool in_alpha_to_coverage = false);
    blend_state_ptr                 create_blend_state(const blend_ops_array& in_blend_ops, bool in_alpha_to_coverage = false);

    // query api //////////////////////////////////////////////////////////////////////////////////
public:
    timer_query_ptr                 create_timer_query();
    occlusion_query_ptr             create_occlusion_query(const occlusion_query_mode in_oq_mode);
    transform_feedback_statistics_query_ptr create_transform_feedback_statistics_query(int stream = 0);

    // debug //////////////////////////////////////////////////////////////////////////////////////
public:
    void                            dump_memory_info(std::ostream& os) const;

#if SCM_ENABLE_CUDA_CL_SUPPORT
    // compute interop ////////////////////////////////////////////////////////////////////////////
public:
    bool                            enable_cuda_interop();
    bool                            enable_opencl_interop();

    const cl::opencl_device_ptr     opencl_interop_device() const;
    const cu::cuda_device_ptr       cuda_interop_device() const;
#endif

////// attributes /////////////////////////////////////////////////////////////////////////////////
protected:
    // device /////////////////////////////////////////////////////////////////////////////////////
    struct mutex_impl;
    shared_ptr<mutex_impl>          _mutex_impl;

    // device /////////////////////////////////////////////////////////////////////////////////////
    shared_ptr<opengl::gl_core>     _opengl_api_core;
    render_context_ptr              _main_context;

    // shader api /////////////////////////////////////////////////////////////////////////////////
    shader_macro_map                _default_macro_defines;
    string_set                      _default_include_paths;

    device_capabilities             _capabilities;
    resource_ptr_set                _registered_resources;

#if SCM_ENABLE_CUDA_CL_SUPPORT
    // compute interop ////////////////////////////////////////////////////////////////////////////
    cl::opencl_device_ptr           _opencl_device;
    cu::cuda_device_ptr             _cuda_device;
#endif

}; // class render_device

__scm_export(gl_core) std::ostream& operator<<(std::ostream& os, const render_device& ren_dev);

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_DEVICE_H_INCLUDED
