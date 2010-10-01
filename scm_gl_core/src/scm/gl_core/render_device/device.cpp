
#include "device.h"

#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include <scm/core/io/tools.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/frame_buffer_objects.h>
#include <scm/gl_core/query_objects.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/shader_objects/program.h>
#include <scm/gl_core/shader_objects/shader.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>
#include <scm/gl_core/state_objects/depth_stencil_state.h>
#include <scm/gl_core/state_objects/rasterizer_state.h>
#include <scm/gl_core/state_objects/sampler_state.h>
#include <scm/gl_core/texture_objects.h>

namespace scm {
namespace gl {

render_device::render_device()
{
    _opengl3_api_core.reset(new opengl::gl3_core());

    if (!_opengl3_api_core->initialize()) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core.";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#if SCM_GL_CORE_OPENGL_40
    if (!(   _opengl3_api_core->version_supported(4, 0))) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(at least OpenGL 4.0 requiered, encountered version "
          << _opengl3_api_core->context_information()._version_major << "."
          << _opengl3_api_core->context_information()._version_minor << ").";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#else // SCM_GL_CORE_OPENGL_40
    if (!(   _opengl3_api_core->version_supported(3, 3))) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(at least OpenGL 3.3 requiered, encountered version "
          << _opengl3_api_core->context_information()._version_major << "."
          << _opengl3_api_core->context_information()._version_minor << ").";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#endif // SCM_GL_CORE_OPENGL_40

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    if (!_opengl3_api_core->is_supported("GL_EXT_direct_state_access")) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(missing requiered extension GL_EXT_direct_state_access).";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#endif

    init_capabilities();

    // setup main rendering context
    _main_context.reset(new render_context(*this));
}

render_device::~render_device()
{
    _main_context.reset();

    assert(0 == _registered_resources.size());
}

const opengl::gl3_core&
render_device::opengl3_api() const
{
    return (*_opengl3_api_core);
}

render_context_ptr
render_device::main_context() const
{
    return (_main_context);
}

render_context_ptr
render_device::create_context()
{
    return (render_context_ptr(new render_context(*this)));
}

const render_device::device_capabilities&
render_device::capabilities() const
{
    return (_capabilities);
}

void
render_device::init_capabilities()
{
    const opengl::gl3_core& glcore = opengl3_api();

    glcore.glGetIntegerv(GL_MAX_VERTEX_ATTRIBS,           &_capabilities._max_vertex_attributes);
    glcore.glGetIntegerv(GL_MAX_DRAW_BUFFERS,             &_capabilities._max_draw_buffers);
    glcore.glGetIntegerv(GL_MAX_DUAL_SOURCE_DRAW_BUFFERS, &_capabilities._max_dual_source_draw_buffers);

    assert(_capabilities._max_vertex_attributes > 0);
    assert(_capabilities._max_draw_buffers > 0);
    assert(_capabilities._max_dual_source_draw_buffers > 0);

    glcore.glGetIntegerv(GL_MAX_TEXTURE_SIZE,             &_capabilities._max_texture_size);
    glcore.glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE,          &_capabilities._max_texture_3d_size);
    glcore.glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,     &_capabilities._max_array_texture_layers);
    glcore.glGetIntegerv(GL_MAX_SAMPLES,                  &_capabilities._max_samples);
    glcore.glGetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES,    &_capabilities._max_depth_texture_samples);
    glcore.glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES,    &_capabilities._max_color_texture_samples);
    glcore.glGetIntegerv(GL_MAX_INTEGER_SAMPLES,          &_capabilities._max_integer_samples);
    glcore.glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,      &_capabilities._max_texture_image_units);
    glcore.glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS,        &_capabilities._max_frame_buffer_color_attachments);

    assert(_capabilities._max_texture_size > 0);
    assert(_capabilities._max_texture_3d_size > 0);
    assert(_capabilities._max_array_texture_layers > 0);
    assert(_capabilities._max_samples > 0);
    assert(_capabilities._max_depth_texture_samples > 0);
    assert(_capabilities._max_color_texture_samples > 0);
    assert(_capabilities._max_integer_samples > 0);
    assert(_capabilities._max_texture_image_units > 0);
    assert(_capabilities._max_frame_buffer_color_attachments > 0);

    glcore.glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS,        &_capabilities._max_vertex_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS,      &_capabilities._max_geometry_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS,      &_capabilities._max_fragment_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS,      &_capabilities._max_combined_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS,      &_capabilities._max_uniform_buffer_bindings);
    glcore.glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT,  &_capabilities._uniform_buffer_offset_alignment);

    assert(_capabilities._max_vertex_uniform_blocks > 0);
    assert(_capabilities._max_geometry_uniform_blocks > 0);
    assert(_capabilities._max_fragment_uniform_blocks > 0);
    assert(_capabilities._max_combined_uniform_blocks > 0);
    assert(_capabilities._max_uniform_buffer_bindings > 0);
    assert(_capabilities._uniform_buffer_offset_alignment > 0);

#if SCM_GL_CORE_OPENGL_41
    glcore.glGetIntegerv(GL_MAX_VIEWPORTS,                    &_capabilities._max_viewports);
#else // SCM_GL_CORE_OPENGL_41
    _capabilities._max_viewports = 1;
#endif // SCM_GL_CORE_OPENGL_41

    assert(_capabilities._max_viewports > 0);
}

buffer_ptr
render_device::create_buffer(const buffer::descriptor_type&  in_buffer_desc,
                             const void*                     in_initial_data)
{
    buffer_ptr new_buffer(new buffer(*this, in_buffer_desc, in_initial_data),
                          boost::bind(&render_device::release_resource, this, _1));
    if (new_buffer->fail()) {
        if (new_buffer->bad()) {
            glerr() << log::error << "render_device::create_buffer(): unable to create buffer object ("
                    << new_buffer->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_buffer(): unable to allocate buffer ("
                    << new_buffer->state().state_string() << ")." << log::end;
        }
        return (buffer_ptr());
    }
    else {
        register_resource(new_buffer.get());
        return (new_buffer);
    }
}

buffer_ptr
render_device::create_buffer(buffer_binding in_binding,
                             buffer_usage   in_usage,
                             scm::size_t    in_size,
                             const void*    in_initial_data)
{
    return (create_buffer(buffer::descriptor_type(in_binding, in_usage, in_size), in_initial_data));
}

bool
render_device::resize_buffer(const buffer_ptr& in_buffer, scm::size_t in_size)
{
    buffer::descriptor_type desc = in_buffer->descriptor();
    desc._size = in_size;
    if (!in_buffer->buffer_data(*this, desc, 0)) {
        glerr() << log::error << "render_device::resize_buffer(): unable to reallocate buffer ("
                << in_buffer->state().state_string() << ")." << log::end;
        return (false);
    }
    else {
        return (true);
    }
}

vertex_array_ptr
render_device::create_vertex_array(const vertex_format& in_vert_fmt,
                                   const buffer_array&  in_attrib_buffers,
                                   const program_ptr&   in_program)
{
    vertex_array_ptr new_array(new vertex_array(*this, in_vert_fmt, in_attrib_buffers, in_program));
    if (new_array->fail()) {
        if (new_array->bad()) {
            glerr() << log::error << "render_device::create_vertex_array(): unable to create vertex array object ("
                    << new_array->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_vertex_array(): unable to initialize vertex array object ("
                    << new_array->state().state_string() << ")." << log::end;
        }
        return (vertex_array_ptr());
    }
    return (new_array);
}

shader_ptr
render_device::create_shader(shader_stage       t,
                             const std::string& s)
{
    shader_ptr new_shader(new shader(*this, t, s));
    if (new_shader->fail()) {
        if (new_shader->bad()) {
            glerr() << "render_device::create_shader(): unable to create shader object ("
                    << "stage: " << shader_stage_string(t) << ", "
                    << new_shader->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_shader(): unable to compile shader ("
                    << "stage: " << shader_stage_string(t) << ", "
                    << new_shader->state().state_string() << "):" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (shader_ptr());
    }
    else {
        if (!new_shader->info_log().empty()) {
            glout() << log::info << "render_device::create_shader(): compiler info"
                    << "(stage: " << shader_stage_string(t) << ")" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (new_shader);
    }
}

shader_ptr
render_device::create_shader_from_file(shader_stage       t,
                                       const std::string& f)
{
    namespace bfs = boost::filesystem;
    bfs::path       file_path(f, bfs::native);
    std::string     source_string;

    if (   !io::read_text_file(f, source_string)) {
        glerr() << "render_device::create_shader_from_file(): error reading shader file " << f << log::end;
        return (shader_ptr());
    }

    shader_ptr new_shader(new shader(*this, t, source_string));
    if (new_shader->fail()) {
        if (new_shader->bad()) {
            glerr() << "render_device::create_shader_from_file(): unable to create shader object" << log::nline
                    << "(" << file_path.filename() << ", " << shader_stage_string(t) << ", "
                    << new_shader->state().state_string() << ")" << log::end;
        }
        else {
            glerr() << "render_device::create_shader_from_file(): unable to compile shader" << log::nline
                    << "(" << file_path.filename() << ", " << shader_stage_string(t) << ", "
                    << new_shader->state().state_string() << "):" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (shader_ptr());
    }
    else {
        if (!new_shader->info_log().empty()) {
            glout() << log::info << "render_device::create_shader_from_file(): compiler info" << log::nline
                    << "(" << file_path.filename() << ", " << shader_stage_string(t) << ")" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (new_shader);
    }
}

program_ptr
render_device::create_program(const shader_list& in_shaders)
{
    program_ptr new_program(new program(*this, in_shaders));
    if (new_program->fail()) {
        if (new_program->bad()) {
            glerr() << "render_device::create_program(): unable to create shader object ("
                    << new_program->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_program(): error during link operation ("
                    << new_program->state().state_string() << "):" << log::nline
                    << new_program->info_log() << log::end;
        }
        return (program_ptr());
    }
    else {
        if (!new_program->info_log().empty()) {
            glout() << log::info << "render_device::create_program(): linker info" << log::nline
                    << new_program->info_log() << log::end;
        }
        return (new_program);
    }
}

texture_1d_ptr
render_device::create_texture_1d(const texture_1d_desc&   in_desc)
{
    texture_1d_ptr  new_tex(new texture_1d(*this, in_desc));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_1d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_1d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_1d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_1d_ptr
render_device::create_texture_1d(const texture_1d_desc&    in_desc,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    texture_1d_ptr  new_tex(new texture_1d(*this, in_desc, in_initial_data_format, in_initial_mip_level_data));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_1d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_1d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_1d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_1d_ptr
render_device::create_texture_1d(const unsigned      in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels,
                                 const unsigned      in_array_layers)
{
    return (create_texture_1d(texture_1d_desc(in_size, in_format, in_mip_levels, in_array_layers)));
}

texture_1d_ptr
render_device::create_texture_1d(const unsigned            in_size,
                                 const data_format         in_format,
                                 const unsigned            in_mip_levels,
                                 const unsigned            in_array_layers,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    return (create_texture_1d(texture_1d_desc(in_size, in_format, in_mip_levels, in_array_layers),
                              in_initial_data_format,
                              in_initial_mip_level_data));
}

texture_2d_ptr
render_device::create_texture_2d(const texture_2d_desc&   in_desc)
{
    texture_2d_ptr  new_tex(new texture_2d(*this, in_desc));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_2d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_2d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_2d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_2d_ptr
render_device::create_texture_2d(const texture_2d_desc&    in_desc,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    texture_2d_ptr  new_tex(new texture_2d(*this, in_desc, in_initial_data_format, in_initial_mip_level_data));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_2d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_2d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_2d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_2d_ptr
render_device::create_texture_2d(const math::vec2ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels,
                                 const unsigned      in_array_layers,
                                 const unsigned      in_samples)
{
    return (create_texture_2d(texture_2d_desc(in_size, in_format, in_mip_levels, in_array_layers, in_samples)));
}

texture_2d_ptr
render_device::create_texture_2d(const math::vec2ui&       in_size,
                                 const data_format         in_format,
                                 const unsigned            in_mip_levels,
                                 const unsigned            in_array_layers,
                                 const unsigned            in_samples,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    return (create_texture_2d(texture_2d_desc(in_size, in_format, in_mip_levels, in_array_layers, in_samples),
                              in_initial_data_format,
                              in_initial_mip_level_data));
}

texture_3d_ptr
render_device::create_texture_3d(const texture_3d_desc&   in_desc)
{
    texture_3d_ptr  new_tex(new texture_3d(*this, in_desc));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_3d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_3d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_3d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_3d_ptr
render_device::create_texture_3d(const texture_3d_desc&    in_desc,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    texture_3d_ptr  new_tex(new texture_3d(*this, in_desc, in_initial_data_format, in_initial_mip_level_data));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_3d(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_3d(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return (texture_3d_ptr());
    }
    else {
        return (new_tex);
    }
}

texture_3d_ptr
render_device::create_texture_3d(const math::vec3ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels)
{
    return (create_texture_3d(texture_3d_desc(in_size, in_format, in_mip_levels)));
}

texture_3d_ptr
render_device::create_texture_3d(const math::vec3ui&       in_size,
                                 const data_format         in_format,
                                 const unsigned            in_mip_levels,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    return (create_texture_3d(texture_3d_desc(in_size, in_format, in_mip_levels),
                              in_initial_data_format,
                              in_initial_mip_level_data));
}

sampler_state_ptr
render_device::create_sampler_state(const sampler_state_desc& in_desc)
{
    sampler_state_ptr  new_sstate(new sampler_state(*this, in_desc));
    if (new_sstate->fail()) {
        if (new_sstate->bad()) {
            glerr() << log::error << "render_device::create_sampler_state(): unable to create sampler state object ("
                    << new_sstate->state().state_string() << ")." << log::end;
        }
        return (sampler_state_ptr());
    }
    else {
        return (new_sstate);
    }
}

sampler_state_ptr
render_device::create_sampler_state(texture_filter_mode  in_filter,
                                    texture_wrap_mode    in_wrap,
                                    unsigned             in_max_anisotropy,
                                    float                in_min_lod,
                                    float                in_max_lod,
                                    float                in_lod_bias,
                                    compare_func         in_compare_func,
                                    texture_compare_mode in_compare_mode)
{
    return (create_sampler_state(sampler_state_desc(in_filter, in_wrap, in_wrap, in_wrap,
        in_max_anisotropy, in_min_lod, in_max_lod, in_lod_bias, in_compare_func, in_compare_mode)));
}

sampler_state_ptr
render_device::create_sampler_state(texture_filter_mode  in_filter,
                                    texture_wrap_mode    in_wrap_s,
                                    texture_wrap_mode    in_wrap_t,
                                    texture_wrap_mode    in_wrap_r,
                                    unsigned             in_max_anisotropy,
                                    float                in_min_lod,
                                    float                in_max_lod,
                                    float                in_lod_bias,
                                    compare_func         in_compare_func,
                                    texture_compare_mode in_compare_mode)
{
    return (create_sampler_state(sampler_state_desc(in_filter, in_wrap_s, in_wrap_t, in_wrap_r,
        in_max_anisotropy, in_min_lod, in_max_lod, in_lod_bias, in_compare_func, in_compare_mode)));
}

// frame buffer api ///////////////////////////////////////////////////////////////////////////
render_buffer_ptr
render_device::create_render_buffer(const render_buffer_desc& in_desc)
{
    render_buffer_ptr  new_rb(new render_buffer(*this, in_desc));
    if (new_rb->fail()) {
        if (new_rb->bad()) {
            glerr() << log::error << "render_device::create_render_buffer(): unable to create render buffer object ("
                    << new_rb->state().state_string() << ")." << log::end;
        }
        return (render_buffer_ptr());
    }
    else {
        return (new_rb);
    }
}

render_buffer_ptr
render_device::create_render_buffer(const math::vec2ui& in_size,
                                    const data_format   in_format,
                                    const unsigned      in_samples)
{
    return (create_render_buffer(render_buffer_desc(in_size, in_format, in_samples)));
}

frame_buffer_ptr
render_device::create_frame_buffer()
{
    frame_buffer_ptr  new_rb(new frame_buffer(*this));
    if (new_rb->fail()) {
        if (new_rb->bad()) {
            glerr() << log::error << "render_device::create_render_buffer(): unable to create frame buffer object ("
                    << new_rb->state().state_string() << ")." << log::end;
        }
        return (frame_buffer_ptr());
    }
    else {
        return (new_rb);
    }
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(const depth_stencil_state_desc& in_desc)
{
    depth_stencil_state_ptr new_ds_state(new depth_stencil_state(*this, in_desc));
    return (new_ds_state);
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                          bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                          stencil_ops in_stencil_ops)
{
    return (create_depth_stencil_state(depth_stencil_state_desc(in_depth_test, in_depth_mask, in_depth_func,
                                                                in_stencil_test, in_stencil_rmask, in_stencil_wmask,
                                                                in_stencil_ops)));
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                          bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                          stencil_ops in_stencil_front_ops, stencil_ops in_stencil_back_ops)
{
    return (create_depth_stencil_state(depth_stencil_state_desc(in_depth_test, in_depth_mask, in_depth_func,
                                                                in_stencil_test, in_stencil_rmask, in_stencil_wmask,
                                                                in_stencil_front_ops, in_stencil_front_ops)));
}

rasterizer_state_ptr
render_device::create_rasterizer_state(const rasterizer_state_desc& in_desc)
{
    rasterizer_state_ptr new_r_state(new rasterizer_state(*this, in_desc));
    return (new_r_state);
}

rasterizer_state_ptr
render_device::create_rasterizer_state(fill_mode in_fmode, cull_mode in_cmode, polygon_orientation in_fface,
                                       bool in_msample, bool in_sctest, bool in_smlines)
{
    return (create_rasterizer_state(rasterizer_state_desc(in_fmode, in_cmode, in_fface,
                                                          in_msample, in_sctest, in_smlines)));
}

blend_state_ptr
render_device::create_blend_state(const blend_state_desc& in_desc)
{
    blend_state_ptr new_bl_state(new blend_state(*this, in_desc));
    return (new_bl_state);
}

blend_state_ptr
render_device::create_blend_state(bool in_enabled,
                                  blend_func in_src_rgb_func,   blend_func in_dst_rgb_func,
                                  blend_func in_src_alpha_func, blend_func in_dst_alpha_func,
                                  blend_equation  in_rgb_equation, blend_equation in_alpha_equation,
                                  unsigned in_write_mask, bool in_alpha_to_coverage)
{
    return (create_blend_state(blend_state_desc(blend_ops(in_enabled,
                                                          in_src_rgb_func,   in_dst_rgb_func,
                                                          in_src_alpha_func, in_dst_alpha_func,
                                                          in_rgb_equation,   in_alpha_equation, in_write_mask),
                                                in_alpha_to_coverage)));
}

blend_state_ptr
render_device::create_blend_state(const blend_ops_array& in_blend_ops, bool in_alpha_to_coverage)
{
    return (create_blend_state(blend_state_desc(in_blend_ops, in_alpha_to_coverage)));
}

// query api //////////////////////////////////////////////////////////////////////////////////
timer_query_ptr
render_device::create_timer_query()
{
    timer_query_ptr  new_tq(new timer_query(*this));
    if (new_tq->fail()) {
        if (new_tq->bad()) {
            glerr() << log::error << "render_device::create_timer_query(): unable to create timer query object ("
                    << new_tq->state().state_string() << ")." << log::end;
        }
        return (timer_query_ptr());
    }
    else {
        return (new_tq);
    }
}

void
render_device::print_device_informations(std::ostream& os) const
{
    os << "OpenGL render device" << std::endl;
    os << *_opengl3_api_core;
}

void
render_device::register_resource(render_device_resource* res_ptr)
{
    _registered_resources.insert(res_ptr);
}

void
render_device::release_resource(render_device_resource* res_ptr)
{
    resource_ptr_set::iterator res_iter = _registered_resources.find(res_ptr);
    if (res_iter != _registered_resources.end()) {
        _registered_resources.erase(res_iter);
    }

    delete res_ptr;
}

std::ostream& operator<<(std::ostream& os, const render_device& ren_dev)
{
    ren_dev.print_device_informations(os);
    return (os);
}

} // namespace gl
} // namespace scm
