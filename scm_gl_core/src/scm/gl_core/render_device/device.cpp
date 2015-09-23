
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "device.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/mutex.hpp>

#include <scm/core/io/tools.h>
#include <scm/core/io/iomanip.h>
#include <scm/core/log/logger_state.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/frame_buffer_objects.h>
#include <scm/gl_core/query_objects.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/shader_objects/program.h>
#include <scm/gl_core/shader_objects/shader.h>
#include <scm/gl_core/shader_objects/stream_capture.h>
#include <scm/gl_core/state_objects/depth_stencil_state.h>
#include <scm/gl_core/state_objects/rasterizer_state.h>
#include <scm/gl_core/state_objects/sampler_state.h>
#include <scm/gl_core/texture_objects.h>

#if SCM_ENABLE_CUDA_CL_SUPPORT
#include <scm/cl_core/cuda/device.h>
#include <scm/cl_core/opencl/device.h>
#endif

namespace scm {
namespace gl {

struct render_device::mutex_impl
{
    boost::mutex    _mutex;
};

render_device::render_device()
  : _mutex_impl(new mutex_impl)
{
    _opengl_api_core.reset(new opengl::gl_core());

    if (!_opengl_api_core->initialize()) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core.";
        glerr() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }
    unsigned req_version_major = SCM_GL_CORE_OPENGL_CORE_VERSION / 100;
    unsigned req_version_minor = (SCM_GL_CORE_OPENGL_CORE_VERSION - req_version_major * 100) / 10;

    if (!_opengl_api_core->version_supported(req_version_major, req_version_minor)) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(at least OpenGL "
          << req_version_major << "." << req_version_minor
          << " requiered, encountered version "
          << _opengl_api_core->context_information()._version_major << "."
          << _opengl_api_core->context_information()._version_minor << ").";
        glerr() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }
    else {
        glout() << log::info << "render_device::render_device(): "
                << "scm_gl_core OpenGL "
                << req_version_major << "." << req_version_minor
                << " support enabled on "
                << _opengl_api_core->context_information()._version_major << "."
                << _opengl_api_core->context_information()._version_minor
                << " context." << log::end;
    }

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    if (!_opengl_api_core->is_supported("GL_EXT_direct_state_access")) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(missing requiered extension GL_EXT_direct_state_access).";
        glerr() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }
#endif

    init_capabilities();

    // setup main rendering context
    try {
        _main_context.reset(new render_context(*this));
        _main_context->apply();
    }
    catch (const std::exception& e) {
        std::ostringstream s;
        s << "render_device::render_device(): error creating main_context (evoking error: "
          << e.what()
          << ").";
        glerr() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }
}

render_device::~render_device()
{
    _main_context.reset();

    assert(0 == _registered_resources.size());
}

const opengl::gl_core&
render_device::opengl_api() const
{
    return *_opengl_api_core;
}

render_context_ptr
render_device::main_context() const
{
    return _main_context;
}

render_context_ptr
render_device::create_context()
{
    return render_context_ptr(new render_context(*this));
}

const render_device::device_capabilities&
render_device::capabilities() const
{
    return _capabilities;
}

void
render_device::init_capabilities()
{
    const opengl::gl_core& glcore = opengl_api();

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
    glcore.glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE,      &_capabilities._max_texture_buffer_size);
    glcore.glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS,        &_capabilities._max_frame_buffer_color_attachments);

    assert(_capabilities._max_texture_size > 0);
    assert(_capabilities._max_texture_3d_size > 0);
    assert(_capabilities._max_array_texture_layers > 0);
    assert(_capabilities._max_samples > 0);
    assert(_capabilities._max_depth_texture_samples > 0);
    assert(_capabilities._max_color_texture_samples > 0);
    assert(_capabilities._max_integer_samples > 0);
    assert(_capabilities._max_texture_image_units > 0);
    assert(_capabilities._max_texture_buffer_size > 0);
    assert(_capabilities._max_frame_buffer_color_attachments > 0);

    glcore.glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS,                  &_capabilities._max_vertex_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS,                &_capabilities._max_geometry_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS,                &_capabilities._max_fragment_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS,                &_capabilities._max_combined_uniform_blocks);
    glcore.glGetIntegerv(GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS,     &_capabilities._max_combined_vertex_uniform_components);
    glcore.glGetIntegerv(GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS,   &_capabilities._max_combined_geometry_uniform_components);
    glcore.glGetIntegerv(GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS,   &_capabilities._max_combined_fragment_uniform_components);
    glcore.glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS,                &_capabilities._max_uniform_buffer_bindings);
    glcore.glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT,            &_capabilities._uniform_buffer_offset_alignment);
    glcore.glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE,                     &_capabilities._max_uniform_block_size);

    assert(_capabilities._max_vertex_uniform_blocks > 0);
    assert(_capabilities._max_geometry_uniform_blocks > 0);
    assert(_capabilities._max_fragment_uniform_blocks > 0);
    assert(_capabilities._max_combined_uniform_blocks > 0);
    assert(_capabilities._max_combined_vertex_uniform_components > 0);
    assert(_capabilities._max_combined_geometry_uniform_components > 0);
    assert(_capabilities._max_combined_fragment_uniform_components > 0);
    assert(_capabilities._max_uniform_buffer_bindings > 0);
    assert(_capabilities._uniform_buffer_offset_alignment > 0);
    assert(_capabilities._max_uniform_block_size > 0);

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_410) {
        glcore.glGetIntegerv(GL_MAX_VIEWPORTS,                    &_capabilities._max_viewports);
    }
    else {
        _capabilities._max_viewports = 1;
    }
    assert(_capabilities._max_viewports > 0);

    glcore.glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS, &_capabilities._max_transform_feedback_separate_attribs);
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        glcore.glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_BUFFERS, &_capabilities._max_transform_feedback_buffers);
        glcore.glGetIntegerv(GL_MAX_VERTEX_STREAMS, &_capabilities._max_vertex_streams);
    }
    else {
        _capabilities._max_transform_feedback_buffers = _capabilities._max_transform_feedback_separate_attribs;
        _capabilities._max_vertex_streams             = 1;
    }
    assert(_capabilities._max_transform_feedback_separate_attribs > 0);
    assert(_capabilities._max_transform_feedback_buffers > 0);
    assert(_capabilities._max_vertex_streams > 0);

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {
        glcore.glGetIntegerv(GL_MAX_IMAGE_UNITS, &_capabilities._max_image_units);
    }
    else if (glcore.extension_EXT_shader_image_load_store) {
        glcore.glGetIntegerv(GL_MAX_IMAGE_UNITS_EXT, &_capabilities._max_image_units);
    }
    else {
        _capabilities._max_image_units = 0;
    }

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {
        glcore.glGetIntegerv(GL_MAX_VERTEX_ATOMIC_COUNTERS,         &_capabilities._max_vertex_atomic_counters);
        glcore.glGetIntegerv(GL_MAX_FRAGMENT_ATOMIC_COUNTERS,       &_capabilities._max_geometry_atomic_counters);
        glcore.glGetIntegerv(GL_MAX_GEOMETRY_ATOMIC_COUNTERS,       &_capabilities._max_fragment_atomic_counters);
        glcore.glGetIntegerv(GL_MAX_COMBINED_ATOMIC_COUNTERS,       &_capabilities._max_combined_atomic_counters);
        glcore.glGetIntegerv(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS, &_capabilities._max_atomic_counter_buffer_bindings);

        assert(_capabilities._max_vertex_atomic_counters        >= 0);
        assert(_capabilities._max_geometry_atomic_counters      >= 0);
        assert(_capabilities._max_fragment_atomic_counters       > 0);
        assert(_capabilities._max_combined_atomic_counters       > 0);
        assert(_capabilities._max_atomic_counter_buffer_bindings > 0);
    }
    else {
        _capabilities._max_vertex_atomic_counters         = 0;
        _capabilities._max_geometry_atomic_counters       = 0;
        _capabilities._max_fragment_atomic_counters       = 0;
        _capabilities._max_combined_atomic_counters       = 0;
        _capabilities._max_atomic_counter_buffer_bindings = 0;
    }

    if (   SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420
        || glcore.extension_ARB_map_buffer_alignment) {
        glcore.glGetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &_capabilities._min_map_buffer_alignment);
    }
    else {
        _capabilities._min_map_buffer_alignment = 1;
    }

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_410) {
        glcore.glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &_capabilities._num_program_binary_formats);
        if (_capabilities._num_program_binary_formats > 0) {
            _capabilities._program_binary_formats.reset(new int[_capabilities._num_program_binary_formats]);
            glcore.glGetIntegerv(GL_PROGRAM_BINARY_FORMATS, _capabilities._program_binary_formats.get());
        }
    }
    else {
        _capabilities._num_program_binary_formats = 0;
    }

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_430) {
        glcore.glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,     &_capabilities._max_shader_storage_block_bindings);
        glcore.glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,          &_capabilities._max_shader_storage_block_size);
        glcore.glGetInteger64v(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &_capabilities._shader_storage_buffer_offset_alignment);
    }
    else {
        _capabilities._max_shader_storage_block_bindings        = 0;
        _capabilities._max_shader_storage_block_size            = 0;
        _capabilities._shader_storage_buffer_offset_alignment   = 1;
    }

    log::logger_format_saver ofs(glout().associated_logger());
    glout() << "render_device::init_capabilities(): OpenGL capabilities"
            << log::indent;

    glout() << "general: " << log::nline
            << log::indent
            << "MAX_VERTEX_ATTRIBS                          " << _capabilities._max_vertex_attributes << log::nline
            << "MAX_DRAW_BUFFERS                            " << _capabilities._max_draw_buffers << log::nline
            << "MAX_DUAL_SOURCE_DRAW_BUFFERS                " << _capabilities._max_dual_source_draw_buffers << log::nline
            << "MAX_TEXTURE_SIZE                            " << _capabilities._max_texture_size << log::nline
            << "MAX_3D_TEXTURE_SIZE                         " << _capabilities._max_texture_3d_size << log::nline
            << "MAX_ARRAY_TEXTURE_LAYERS                    " << _capabilities._max_array_texture_layers << log::nline
            << "MAX_SAMPLES                                 " << _capabilities._max_samples << log::nline
            << "MAX_DEPTH_TEXTURE_SAMPLES                   " << _capabilities._max_depth_texture_samples << log::nline
            << "MAX_COLOR_TEXTURE_SAMPLES                   " << _capabilities._max_color_texture_samples << log::nline
            << "MAX_INTEGER_SAMPLES                         " << _capabilities._max_integer_samples << log::nline
            << "MAX_TEXTURE_IMAGE_UNITS                     " << _capabilities._max_texture_image_units << log::nline
            << "MAX_TEXTURE_BUFFER_SIZE                     " << _capabilities._max_texture_buffer_size << log::nline
            << "MAX_COLOR_ATTACHMENTS                       " << _capabilities._max_frame_buffer_color_attachments << log::nline
            << "MAX_VIEWPORTS                               " << _capabilities._max_viewports
            << log::outdent;

    glout() << "uniform blocks: " << log::nline
            << log::indent
            << "MAX_UNIFORM_BLOCK_SIZE                      " << io::data_size(_capabilities._max_uniform_block_size) << log::nline
            << "MAX_VERTEX_UNIFORM_BLOCKS                   " << _capabilities._max_vertex_uniform_blocks << log::nline
            << "MAX_GEOMETRY_UNIFORM_BLOCKS                 " << _capabilities._max_geometry_uniform_blocks << log::nline
            << "MAX_FRAGMENT_UNIFORM_BLOCKS                 " << _capabilities._max_fragment_uniform_blocks << log::nline
            << "MAX_COMBINED_UNIFORM_BLOCKS                 " << _capabilities._max_combined_uniform_blocks << log::nline
            << "MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS      " << _capabilities._max_combined_vertex_uniform_components << log::nline
            << "MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS    " << _capabilities._max_combined_geometry_uniform_components << log::nline
            << "MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS    " << _capabilities._max_combined_fragment_uniform_components << log::nline
            << "MAX_UNIFORM_BUFFER_BINDINGS                 " << _capabilities._max_uniform_buffer_bindings << log::nline
            << "UNIFORM_BUFFER_OFFSET_ALIGNMENT             " << _capabilities._uniform_buffer_offset_alignment
            << log::outdent;

    glout() << "transform feedback: " << log::nline
            << log::indent
            << "MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS     " << _capabilities._max_transform_feedback_separate_attribs << log::nline
            << "MAX_TRANSFORM_FEEDBACK_BUFFERS              " << _capabilities._max_transform_feedback_buffers << log::nline
            << "MAX_VERTEX_STREAMS                          " << _capabilities._max_vertex_streams
            << log::outdent;

    glout() << "image load/store: " << log::nline
            << log::indent
            << "MAX_IMAGE_UNITS                             " << _capabilities._max_image_units
            << log::outdent;

    glout() << "atomic counters: " << log::nline
            << log::indent
            << "MAX_VERTEX_ATOMIC_COUNTERS                  " << _capabilities._max_vertex_atomic_counters << log::nline
            << "MAX_FRAGMENT_ATOMIC_COUNTERS                " << _capabilities._max_geometry_atomic_counters << log::nline
            << "MAX_GEOMETRY_ATOMIC_COUNTERS                " << _capabilities._max_fragment_atomic_counters << log::nline
            << "MAX_COMBINED_ATOMIC_COUNTERS                " << _capabilities._max_combined_atomic_counters << log::nline
            << "MAX_ATOMIC_COUNTER_BUFFER_BINDINGS          " << _capabilities._max_atomic_counter_buffer_bindings
            << log::outdent;

    glout() << "map buffer alignment: " << log::nline
            << log::indent
            << "MIN_MAP_BUFFER_ALIGNMENT                    " << _capabilities._min_map_buffer_alignment
            << log::outdent;

    glout() << "shader storage buffers: " << log::nline
            << log::indent
            << "MAX_SHADER_STORAGE_BUFFER_BINDINGS          " << _capabilities._max_shader_storage_block_bindings << log::nline
            << "MAX_SHADER_STORAGE_BLOCK_SIZE               " << io::data_size(_capabilities._max_shader_storage_block_size) << log::nline
            << "SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT      " << _capabilities._shader_storage_buffer_offset_alignment
            << log::outdent;

    std::stringstream pbf;
    pbf << "(";
    if (0 < _capabilities._num_program_binary_formats) {
        for (int f = 0; f < _capabilities._num_program_binary_formats; ++f) {
            pbf << std::hex << "0x" << _capabilities._program_binary_formats[f];
            if (f < _capabilities._num_program_binary_formats - 1) {
                pbf << ", ";
            }
        }
    } else {
        pbf << "N/A";
    }
    pbf << ")";

    glout() << "program binary formats: " << log::nline
            << log::indent
            << "GL_NUM_PROGRAM_BINARY_FORMATS               " << _capabilities._num_program_binary_formats << log::nline
            << "GL_PROGRAM_BINARY_FORMATS                   " << pbf.str() << log::nline
            << log::outdent;

    //std::cout << "GL_MAX_IMAGE_UNITS_EXT " << _capabilities._max_image_units << std::endl;
}

// buffer api /////////////////////////////////////////////////////////////////////////////////////
buffer_ptr
render_device::create_buffer(const buffer_desc& in_buffer_desc,
                             const void*        in_initial_data)
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
        return buffer_ptr();
    }
    else {
        register_resource(new_buffer.get());
        return new_buffer;
    }
}

buffer_ptr
render_device::create_buffer(buffer_binding in_binding,
                             buffer_usage   in_usage,
                             scm::size_t    in_size,
                             const void*    in_initial_data)
{
    return create_buffer(buffer_desc(in_binding, in_usage, in_size), in_initial_data);
}

bool
render_device::resize_buffer(const buffer_ptr& in_buffer, scm::size_t in_size)
{
    buffer_desc desc = in_buffer->descriptor();
    desc._size = in_size;
    if (!in_buffer->buffer_data(*this, desc, 0)) {
        glerr() << log::error << "render_device::resize_buffer(): unable to reallocate buffer ("
                << in_buffer->state().state_string() << ")." << log::end;
        return false;
    }
    else {
        return true;
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
        return vertex_array_ptr();
    }
    return new_array;
}

transform_feedback_ptr
render_device::create_transform_feedback(const stream_output_setup& in_setup)
{
    transform_feedback_ptr new_feedback(new transform_feedback(*this, in_setup));
    if (new_feedback->fail()) {
        if (new_feedback->bad()) {
            glerr() << log::error << "render_device::create_transform_feedback(): unable to create transform feedback object ("
                    << new_feedback->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_transform_feedback(): unable to initialize transform feedback object ("
                    << new_feedback->state().state_string() << ")." << log::end;
        }
        return transform_feedback_ptr();
    }
    return new_feedback;
}

// shader api /////////////////////////////////////////////////////////////////////////////////////
bool
render_device::add_include_files(const std::string& in_path,
                                 const std::string& in_glsl_root_path,
                                 const std::string& in_file_extensions,
                                 bool               in_scan_subdirectories)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
        boost::char_separator<char> space_separator(" ");
        tokenizer                   file_extensions(in_file_extensions, space_separator);

        namespace bfs = boost::filesystem;

        std::string         output_root_path
            = boost::trim_left_copy_if(
                  boost::trim_right_copy_if(
                    in_glsl_root_path,
                    boost::is_any_of("/")),
                  boost::is_any_of("/"));

        if (!output_root_path.empty()) {
            output_root_path = std::string("/") + output_root_path + std::string("/");
        }
        else {
            output_root_path = std::string("/");
        }

        bfs::path           input_path = bfs::path(in_path);
        bfs::path           input_root;

        //if (input_path.is_relative()) {
        //    input_path = bfs::absolute(input_path);
        //}

        if (!bfs::exists(input_path)) {
            glerr() << log::error << "render_device::add_include_files(): "
                    << "<error> input path does not exist (" << input_path << ")." << log::end;
            return false;
        }

        if (bfs::is_directory(input_path)) {
            input_root = input_path;
        }
        else {
            glerr() << log::error << "render_device::add_include_files(): "
                    << "<error> input path is a file (" << input_path << ")." << log::end;
            return false;
        }

        if (in_scan_subdirectories) {
            bfs::recursive_directory_iterator  file_iter(input_path);
            bfs::recursive_directory_iterator  e = bfs::recursive_directory_iterator();
            for (; file_iter != e; ++file_iter) {
                bfs::path current_file = file_iter->path();
                if (!bfs::is_directory(current_file)) {
                    if (    std::find(file_extensions.begin(),
                                      file_extensions.end(),
                                      current_file.extension().string())
                        != file_extensions.end())
                    {
                        std::string     source_string;
                        if (io::read_text_file(current_file.string(), source_string)) {
                            // me not likey... but does the trick in a portable manner
                            bfs::path::const_iterator first_mis
                                = std::mismatch(input_root.begin(), input_root.end(),
                                                current_file.begin()).second;
                            bfs::path input_rel_path;
                            for (; first_mis != current_file.end(); ++first_mis) input_rel_path /= *first_mis;

                            assert(input_path / input_rel_path == current_file);
                            add_include_string_internal(output_root_path + input_rel_path.generic_string(), source_string, false);
                        }
                        else {
                            glout() << log::warning << "render_device::add_include_files(): error reading shader file " << current_file << log::end;
                        }
                    }
                }
            }
        }
        else {
            bfs::directory_iterator  file_iter(input_path);
            bfs::directory_iterator  e = bfs::directory_iterator();
            for (; file_iter != e; ++file_iter) {
                bfs::path current_file = file_iter->path();
                if (!bfs::is_directory(current_file)) {
                    if (    std::find(file_extensions.begin(),
                                      file_extensions.end(),
                                      current_file.extension().string())
                        != file_extensions.end())
                    {
                        std::string     source_string;
                        if (io::read_text_file(current_file.string(), source_string)) {
                            // me not likey... but does the trick in a portable manner
                            bfs::path::const_iterator first_mis
                                = std::mismatch(input_root.begin(), input_root.end(),
                                                current_file.begin()).second;
                            bfs::path input_rel_path;
                            for (; first_mis != current_file.end(); ++first_mis) input_rel_path /= *first_mis;

                            assert(input_path / input_rel_path == current_file);
                            add_include_string_internal(output_root_path + input_rel_path.generic_string(), source_string, false);
                        }
                        else {
                            glout() << log::warning << "render_device::add_include_files(): error reading shader file " << current_file << log::end;
                        }
                    }
                }
            }
        }
    }

    return true;
}

bool
render_device::add_include_string(const std::string& in_path,
                                  const std::string& in_source_string)
{
    return add_include_string_internal(in_path, in_source_string, true);
}

bool
render_device::add_include_string_internal(const std::string& in_path,
                                           const std::string& in_source_string,
                                                 bool         lock_thread)
{
    { // protect this function from multiple thread access
        scoped_ptr<boost::mutex::scoped_lock> lock;
        if (lock_thread) {
            lock.reset(new boost::mutex::scoped_lock(_mutex_impl->_mutex));
        }

        const opengl::gl_core& glcore = opengl_api();
        util::gl_error          glerror(glcore);

        if (!glcore.extension_ARB_shading_language_include) {
            glout() << log::warning << "render_device::add_include_string(): "
                    << "shader includes not supported (GL_ARB_shading_language_include unsupported), ignoring include string." << log::end;
            return false;
        }

        if (in_path[0] != '/') {
            glerr() << log::error << "render_device::add_include_string(): "
                    << "<error> path not starting with '/'." << log::end;
            return false;
        }

        glcore.glNamedStringARB(GL_SHADER_INCLUDE_ARB,
                                static_cast<int>(in_path.length()),          in_path.c_str(),
                                static_cast<int>(in_source_string.length()), in_source_string.c_str());

        if (glerror) {
            switch (glerror.to_object_state()) {
            case object_state::OS_ERROR_INVALID_VALUE:
                glerr() << log::error << "render_device::add_include_string(): "
                        << "error creating named include string (path or source string empty or path not starting with '/'." << log::end;
                return false;
                break;
            default:
                glerr() << log::error << "render_device::add_include_string(): "
                        << "error creating named include string (an unknown error occured)" << log::end;
                return false;
            }
        }

        size_t      parent_path_end = in_path.find_last_of('/');
        std::string parent_path     = in_path.substr(0, parent_path_end);

        //if (!parent_path.empty()) {
        //    _default_include_paths.insert(parent_path);
        //}

        gl_assert(glcore, leaving render_device::add_include_string());
    }

    return true;
}

void
render_device::add_macro_define(const std::string& in_name,
                                const std::string& in_value)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        _default_macro_defines[in_name] = shader_macro(in_name, in_value);
    }
}

void
render_device::add_macro_define(const shader_macro& in_macro)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        _default_macro_defines[in_macro._name] = in_macro;
    }
}

void
render_device::add_macro_defines(const shader_macro_array& in_macros)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        foreach(const shader_macro& m, in_macros.macros()) {
            _default_macro_defines[m._name] = m;
        }
    }
}

shader_ptr
render_device::create_shader(shader_stage       in_stage,
                             const std::string& in_source,
                             const std::string& in_source_name)
{
    return create_shader(in_stage, in_source, shader_macro_array(), shader_include_path_list(), in_source_name);
}

shader_ptr
render_device::create_shader(shader_stage              in_stage,
                             const std::string&        in_source,
                             const shader_macro_array& in_macros,
                             const std::string&        in_source_name)
{
    return create_shader(in_stage, in_source, in_macros, shader_include_path_list(), in_source_name);
}

shader_ptr
render_device::create_shader(shader_stage                    in_stage,
                             const std::string&              in_source,
                             const shader_include_path_list& in_inc_paths,
                             const std::string&              in_source_name)
{
    return create_shader(in_stage, in_source, shader_macro_array(), in_inc_paths, in_source_name);
}

shader_ptr
render_device::create_shader(shader_stage                    in_stage,
                             const std::string&              in_source,
                             const shader_macro_array&       in_macros,
                             const shader_include_path_list& in_inc_paths,
                             const std::string&              in_source_name)
{
    // combine macro definitions
    shader_macro_array  macro_array(in_macros);

    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        shader_macro_map::const_iterator mb = _default_macro_defines.begin();
        shader_macro_map::const_iterator me = _default_macro_defines.end();

        for(; mb != me; ++mb) {
            macro_array(mb->second._name, mb->second._value);
        }
    }

    // combine shader include paths
    shader_include_path_list   include_paths(in_inc_paths);

    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        string_set::const_iterator ipb = _default_include_paths.begin();
        string_set::const_iterator ipe = _default_include_paths.end();

        for(; ipb != ipe; ++ipb) {
            include_paths.push_back(*ipb);
        }
    }

    shader_ptr new_shader(new shader(*this,
                                     in_stage,
                                     in_source,
                                     in_source_name,
                                     macro_array,
                                     include_paths));
    if (new_shader->fail()) {
        if (new_shader->bad()) {
            glerr() << "render_device::create_shader(): unable to create shader object ("
                    << "name: " << in_source_name << ", "
                    << "stage: " << shader_stage_string(in_stage) << ", "
                    << new_shader->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_shader(): unable to compile shader ("
                    << "name: " << in_source_name << ", "
                    << "stage: " << shader_stage_string(in_stage) << ", "
                    << new_shader->state().state_string() << "):" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return shader_ptr();
    }
    else {
        if (!new_shader->info_log().empty()) {
            glout() << log::info << "render_device::create_shader(): compiler info ("
                    << "name: " << in_source_name << ", "
                    << "stage: " << shader_stage_string(in_stage)
                    << ")" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return new_shader;
    }
}

shader_ptr
render_device::create_shader_from_file(shader_stage       in_stage,
                                       const std::string& in_file_name)
{
    return create_shader_from_file(in_stage, in_file_name, shader_macro_array(), shader_include_path_list());
}

shader_ptr
render_device::create_shader_from_file(shader_stage              in_stage,
                                       const std::string&        in_file_name,
                                       const shader_macro_array& in_macros)
{
    return create_shader_from_file(in_stage, in_file_name, in_macros, shader_include_path_list());
}

shader_ptr
render_device::create_shader_from_file(shader_stage                    in_stage,
                                       const std::string&              in_file_name,
                                       const shader_include_path_list& in_inc_paths)
{
    return create_shader_from_file(in_stage, in_file_name, shader_macro_array(), in_inc_paths);
}

shader_ptr
render_device::create_shader_from_file(shader_stage                    in_stage,
                                       const std::string&              in_file_name,
                                       const shader_macro_array&       in_macros,
                                       const shader_include_path_list& in_inc_paths)
{
    namespace bfs = boost::filesystem;
    bfs::path       file_path(in_file_name);
    std::string     source_string;

    if (!bfs::exists(file_path)) {
        glerr() << "render_device::create_shader_from_file(): unable to find shader file " << in_file_name << log::end;
        return (shader_ptr());
    }
    if (   !io::read_text_file(in_file_name, source_string)) {
        glerr() << "render_device::create_shader_from_file(): error reading shader file " << in_file_name << log::end;
        return (shader_ptr());
    }

    return create_shader(in_stage, source_string, in_macros, in_inc_paths, file_path.filename().string());
}

program_ptr
render_device::create_program(const shader_list& in_shaders,
                              const std::string& in_program_name)
{
    return create_program(in_shaders, stream_capture_array(), false, in_program_name);
}

program_ptr
render_device::create_program(const shader_list&          in_shaders,
                              const stream_capture_array& in_capture,
                              bool                        in_rasterization_discard,
                              const std::string&          in_program_name)
{
    program_ptr new_program(new program(*this, in_shaders, in_capture, in_rasterization_discard));
    if (new_program->fail()) {
        if (new_program->bad()) {
            glerr() << "render_device::create_program(): unable to create shader object ("
                    << "name: " << in_program_name << ", "
                    << new_program->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_program(): error during link operation ("
                    << "name: " << in_program_name << ", "
                    << new_program->state().state_string() << "):" << log::nline
                    << new_program->info_log() << log::end;
        }
        return program_ptr();
    }
    else {
        if (!new_program->info_log().empty()) {
            glout() << log::info << "render_device::create_program(): linker info ("
                    << "name: " << in_program_name << ")" << log::nline
                    << new_program->info_log() << log::end;
        }
        return new_program;
    }
}

// texture api ////////////////////////////////////////////////////////////////////////////////////
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
        return texture_1d_ptr();
    }
    else {
        return new_tex;
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
        return texture_1d_ptr();
    }
    else {
        return new_tex;
    }
}

texture_1d_ptr
render_device::create_texture_1d(const unsigned      in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels,
                                 const unsigned      in_array_layers)
{
    return create_texture_1d(texture_1d_desc(in_size, in_format, in_mip_levels, in_array_layers));
}

texture_1d_ptr
render_device::create_texture_1d(const unsigned            in_size,
                                 const data_format         in_format,
                                 const unsigned            in_mip_levels,
                                 const unsigned            in_array_layers,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    return create_texture_1d(texture_1d_desc(in_size, in_format, in_mip_levels, in_array_layers),
                             in_initial_data_format,
                             in_initial_mip_level_data);
}

texture_1d_ptr
render_device::create_texture_1d(const texture_1d_ptr&     in_orig_texture,
                                 const data_format         in_format,
                                 const math::vec2ui&       in_mip_range,
                                 const math::vec2ui&       in_layer_range)
{
    texture_1d_ptr  new_tex(new texture_1d(*this, *in_orig_texture, in_format, in_mip_range, in_layer_range));
    if (new_tex->fail()) {
        glerr() << log::error << "render_device::create_texture_1d(): unable to create texture view object ("
                << new_tex->state().state_string() << ")." << log::end;
        return texture_1d_ptr();
    }
    else {
        return new_tex;
    }
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
        return texture_2d_ptr();
    }
    else {
        return new_tex;
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
        return texture_2d_ptr();
    }
    else {
        return new_tex;
    }
}

texture_2d_ptr
render_device::create_texture_2d(const math::vec2ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels,
                                 const unsigned      in_array_layers,
                                 const unsigned      in_samples)
{
    return create_texture_2d(texture_2d_desc(in_size, in_format, in_mip_levels, in_array_layers, in_samples));
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
    return create_texture_2d(texture_2d_desc(in_size, in_format, in_mip_levels, in_array_layers, in_samples),
                             in_initial_data_format,
                             in_initial_mip_level_data);
}

texture_2d_ptr
render_device::create_texture_2d(const texture_2d_ptr&     in_orig_texture,
                                 const data_format         in_format,
                                 const math::vec2ui&       in_mip_range,
                                 const math::vec2ui&       in_layer_range)
{
    texture_2d_ptr  new_tex(new texture_2d(*this, *in_orig_texture, in_format, in_mip_range, in_layer_range));
    if (new_tex->fail()) {
        glerr() << log::error << "render_device::create_texture_2d(): unable to create texture view object ("
                << new_tex->state().state_string() << ")." << log::end;
        return texture_2d_ptr();
    }
    else {
        return new_tex;
    }
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
        return texture_3d_ptr();
    }
    else {
        return new_tex;
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
        return texture_3d_ptr();
    }
    else {
        return new_tex;
    }
}

texture_3d_ptr
render_device::create_texture_3d(const math::vec3ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels)
{
    return create_texture_3d(texture_3d_desc(in_size, in_format, in_mip_levels));
}

texture_3d_ptr
render_device::create_texture_3d(const math::vec3ui&       in_size,
                                 const data_format         in_format,
                                 const unsigned            in_mip_levels,
                                 const data_format         in_initial_data_format,
                                 const std::vector<void*>& in_initial_mip_level_data)
{
    return create_texture_3d(texture_3d_desc(in_size, in_format, in_mip_levels),
                             in_initial_data_format,
                             in_initial_mip_level_data);
}

texture_3d_ptr
render_device::create_texture_3d(const texture_3d_ptr&     in_orig_texture,
                                 const data_format         in_format,
                                 const math::vec2ui&       in_mip_range)
{
    texture_3d_ptr  new_tex(new texture_3d(*this, *in_orig_texture, in_format, in_mip_range));
    if (new_tex->fail()) {
        glerr() << log::error << "render_device::create_texture_3d(): unable to create texture view object ("
                << new_tex->state().state_string() << ")." << log::end;
        return texture_3d_ptr();
    }
    else {
        return new_tex;
    }
}

texture_cube_ptr
render_device::create_texture_cube(const texture_cube_desc&   in_desc)
{
    texture_cube_ptr  new_tex(new texture_cube(*this, in_desc));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_cube(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_cube(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return texture_cube_ptr();
    }
    else {
        return new_tex;
    }
}

texture_cube_ptr
render_device::create_texture_cube(const texture_cube_desc&    in_desc,
                                   const data_format         in_initial_data_format,
                                   const std::vector<void*>& in_initial_mip_level_data_px,
                                   const std::vector<void*>& in_initial_mip_level_data_nx,
                                   const std::vector<void*>& in_initial_mip_level_data_py,
                                   const std::vector<void*>& in_initial_mip_level_data_ny,
                                   const std::vector<void*>& in_initial_mip_level_data_pz,
                                   const std::vector<void*>& in_initial_mip_level_data_nz)
{
    texture_cube_ptr  new_tex(new texture_cube(*this, in_desc, in_initial_data_format,
                                                 in_initial_mip_level_data_px,
                                                 in_initial_mip_level_data_nx,
                                                 in_initial_mip_level_data_py,
                                                 in_initial_mip_level_data_ny,
                                                 in_initial_mip_level_data_pz,
                                                 in_initial_mip_level_data_nz));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_cube(): unable to create texture object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_cube(): unable to allocate texture image data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return texture_cube_ptr();
    }
    else {
        return new_tex;
    }
} 

texture_cube_ptr
render_device::create_texture_cube(const math::vec2ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels)
{
    return create_texture_cube(texture_cube_desc(in_size, in_format, in_mip_levels));
}

texture_cube_ptr
render_device::create_texture_cube(const math::vec2ui&       in_size,
                                   const data_format         in_format,
                                   const unsigned            in_mip_levels,
                                   const data_format         in_initial_data_format,
                                   const std::vector<void*>& in_initial_mip_level_data_px,
                                   const std::vector<void*>& in_initial_mip_level_data_nx,
                                   const std::vector<void*>& in_initial_mip_level_data_py,
                                   const std::vector<void*>& in_initial_mip_level_data_ny,
                                   const std::vector<void*>& in_initial_mip_level_data_pz,
                                   const std::vector<void*>& in_initial_mip_level_data_nz)
{
    return create_texture_cube(texture_cube_desc(in_size, in_format, in_mip_levels),
                             in_initial_data_format,
                             in_initial_mip_level_data_px,
                             in_initial_mip_level_data_nx,
                             in_initial_mip_level_data_py,
                             in_initial_mip_level_data_ny,
                             in_initial_mip_level_data_pz,
                             in_initial_mip_level_data_nz);
}

texture_buffer_ptr
render_device::create_texture_buffer(const texture_buffer_desc& in_desc)
{
    texture_buffer_ptr  new_tex(new texture_buffer(*this, in_desc));
    if (new_tex->fail()) {
        if (new_tex->bad()) {
            glerr() << log::error << "render_device::create_texture_buffer(): unable to create texture buffer object ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_texture_buffer(): unable to allocate or attach texture buffer data ("
                    << new_tex->state().state_string() << ")." << log::end;
        }
        return texture_buffer_ptr();
    }
    else {
        return new_tex;
    }
}

texture_buffer_ptr
render_device::create_texture_buffer(const data_format   in_format,
                                     const buffer_ptr&   in_buffer)
{
    return create_texture_buffer(texture_buffer_desc(in_format, in_buffer));
}

texture_buffer_ptr
render_device::create_texture_buffer(const data_format   in_format,
                                     buffer_usage        in_buffer_usage,
                                     scm::size_t         in_buffer_size,
                                     const void*         in_buffer_initial_data)
{
    buffer_ptr  tex_buffer = create_buffer(BIND_TEXTURE_BUFFER, in_buffer_usage, in_buffer_size, in_buffer_initial_data);
    if (!tex_buffer) {
        glerr() << log::error << "render_device::create_texture_buffer(): unable to create texture buffer data buffer." << log::end;
        return texture_buffer_ptr();
    }
    return create_texture_buffer(in_format, tex_buffer);
}

texture_handle_ptr
render_device::create_resident_handle(const texture_ptr&       in_texture,
                                      const sampler_state_ptr& in_sampler)
{
    assert(in_texture);
    assert(in_sampler);

    texture_handle_ptr new_tex_handle(new texture_handle(*this, *in_texture, *in_sampler));
    if (new_tex_handle->fail()) {
        glerr() << log::error << "render_device::create_resident_handle(): unable to create texture handle ("
                << new_tex_handle->state().state_string() << ")." << log::end;
        return texture_handle_ptr();
    }
    else {
        return new_tex_handle;
    }
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
        return sampler_state_ptr();
    }
    else {
        return new_sstate;
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
    return create_sampler_state(sampler_state_desc(in_filter, in_wrap, in_wrap, in_wrap,
        in_max_anisotropy, in_min_lod, in_max_lod, in_lod_bias, in_compare_func, in_compare_mode));
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
    return create_sampler_state(sampler_state_desc(in_filter, in_wrap_s, in_wrap_t, in_wrap_r,
        in_max_anisotropy, in_min_lod, in_max_lod, in_lod_bias, in_compare_func, in_compare_mode));
}

// frame buffer api ///////////////////////////////////////////////////////////////////////////////
render_buffer_ptr
render_device::create_render_buffer(const render_buffer_desc& in_desc)
{
    render_buffer_ptr  new_rb(new render_buffer(*this, in_desc));
    if (new_rb->fail()) {
        if (new_rb->bad()) {
            glerr() << log::error << "render_device::create_render_buffer(): unable to create render buffer object ("
                    << new_rb->state().state_string() << ")." << log::end;
        }
        return render_buffer_ptr();
    }
    else {
        return new_rb;
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
        return frame_buffer_ptr();
    }
    else {
        return new_rb;
    }
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(const depth_stencil_state_desc& in_desc)
{
    depth_stencil_state_ptr new_ds_state(new depth_stencil_state(*this, in_desc));
    return new_ds_state;
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                          bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                          stencil_ops in_stencil_ops)
{
    return create_depth_stencil_state(depth_stencil_state_desc(in_depth_test, in_depth_mask, in_depth_func,
                                                               in_stencil_test, in_stencil_rmask, in_stencil_wmask,
                                                               in_stencil_ops));
}

depth_stencil_state_ptr
render_device::create_depth_stencil_state(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                                          bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                                          stencil_ops in_stencil_front_ops, stencil_ops in_stencil_back_ops)
{
    return create_depth_stencil_state(depth_stencil_state_desc(in_depth_test, in_depth_mask, in_depth_func,
                                                               in_stencil_test, in_stencil_rmask, in_stencil_wmask,
                                                               in_stencil_front_ops, in_stencil_front_ops));
}

rasterizer_state_ptr
render_device::create_rasterizer_state(const rasterizer_state_desc& in_desc)
{
    rasterizer_state_ptr new_r_state(new rasterizer_state(*this, in_desc));
    return new_r_state;
}

rasterizer_state_ptr
render_device::create_rasterizer_state(fill_mode in_fmode, cull_mode in_cmode, polygon_orientation in_fface,
                                       bool in_msample,  bool in_sshading, float32 in_min_sshading,
                                       bool in_sctest, bool in_smlines, const point_raster_state& in_point_state)
{
    return create_rasterizer_state(rasterizer_state_desc(in_fmode, in_cmode, in_fface,
                                                         in_msample, in_sshading, in_min_sshading,
                                                         in_sctest, in_smlines, in_point_state));
}

blend_state_ptr
render_device::create_blend_state(const blend_state_desc& in_desc)
{
    blend_state_ptr new_bl_state(new blend_state(*this, in_desc));
    return new_bl_state;
}

blend_state_ptr
render_device::create_blend_state(bool in_enabled,
                                  blend_func in_src_rgb_func,   blend_func in_dst_rgb_func,
                                  blend_func in_src_alpha_func, blend_func in_dst_alpha_func,
                                  blend_equation  in_rgb_equation, blend_equation in_alpha_equation,
                                  unsigned in_write_mask, bool in_alpha_to_coverage)
{
    return create_blend_state(blend_state_desc(blend_ops(in_enabled,
                                                         in_src_rgb_func,   in_dst_rgb_func,
                                                         in_src_alpha_func, in_dst_alpha_func,
                                                         in_rgb_equation,   in_alpha_equation, in_write_mask),
                                               in_alpha_to_coverage));
}

blend_state_ptr
render_device::create_blend_state(const blend_ops_array& in_blend_ops, bool in_alpha_to_coverage)
{
    return create_blend_state(blend_state_desc(in_blend_ops, in_alpha_to_coverage));
}

// query api //////////////////////////////////////////////////////////////////////////////////////
timer_query_ptr
render_device::create_timer_query()
{
    timer_query_ptr  new_tq(new timer_query(*this));
    if (new_tq->fail()) {
        if (new_tq->bad()) {
            glerr() << log::error << "render_device::create_timer_query(): unable to create timer query object ("
                    << new_tq->state().state_string() << ")." << log::end;
        }
        return timer_query_ptr();
    }
    else {
        return new_tq;
    }
}

transform_feedback_statistics_query_ptr
render_device::create_transform_feedback_statistics_query(int stream)
{
    transform_feedback_statistics_query_ptr  new_xfbq(new transform_feedback_statistics_query(*this, stream));
    if (new_xfbq->fail()) {
        if (new_xfbq->bad()) {
            glerr() << log::error << "render_device::create_transform_feedback_statistics_query(): unable to create transform feedback statistics query object ("
                    << new_xfbq->state().state_string() << ")." << log::end;
        }
        return transform_feedback_statistics_query_ptr();
    }
    else {
        return new_xfbq;
    }
}

occlusion_query_ptr
render_device::create_occlusion_query(const occlusion_query_mode in_oq_mode)
{
    occlusion_query_ptr new_oq(new occlusion_query(*this, in_oq_mode));
    if (new_oq->fail()) {
        if (new_oq->bad()) {
            glerr() << log::error << "render_device::create_occlusion_query(): unable to create occlusion query object("
                    << new_oq->state().state_string() << ")." << log::end;
        }
        return occlusion_query_ptr();
    }
    else {
        return new_oq;
    }
}

// debug //////////////////////////////////////////////////////////////////////////////////////////
void
render_device::dump_memory_info(std::ostream& os) const
{
    const opengl::gl_core& glcore = opengl_api();
    util::gl_error         glerror(glcore);

    if (!glcore.extension_NVX_gpu_memory_info) {
        glout() << log::warning << "render_device::dump_memory_info(): "
                << "shader includes not supported (GL_NVX_gpu_memory_info unsupported), ignoring call." << log::end;
    }
    else {
        { // protect this function from multiple thread access
            boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

            int dedicated_vidmem         = 0;
            int total_available_memory   = 0;
            int current_available_vidmem = 0;
            int eviction_count           = 0;
            int evicted_memory           = 0;

            glcore.glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX        , &dedicated_vidmem        );
            glcore.glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX  , &total_available_memory  );
            glcore.glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &current_available_vidmem);
            glcore.glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX          , &eviction_count          );
            glcore.glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX          , &evicted_memory          );
        
            os << std::fixed << std::setprecision(3)
               << "dedicated_vidmem        : " << static_cast<float>(dedicated_vidmem        ) / 1024.0f << "MiB" << std::endl
               << "total_available_memory  : " << static_cast<float>(total_available_memory  ) / 1024.0f << "MiB" << std::endl
               << "current_available_vidmem: " << static_cast<float>(current_available_vidmem) / 1024.0f << "MiB" << std::endl
               << "eviction_count          : " << eviction_count           << std::endl
               << "evicted_memory          : " << evicted_memory           << std::endl;
        }
    }

    gl_assert(glcore, leaving render_device::dump_memory_info());
}

void
render_device::print_device_informations(std::ostream& os) const
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        os << "OpenGL render device" << std::endl;
        os << *_opengl_api_core;
    }
}
const std::string
render_device::device_vendor() const
{
    return _opengl_api_core->context_information()._vendor;
}

const std::string
render_device::device_renderer() const
{
    return _opengl_api_core->context_information()._renderer;
}

const std::string
render_device::device_shader_compiler() const
{
    return _opengl_api_core->context_information()._glsl_version_info;
}

const std::string
render_device::device_context_version() const
{
    std::stringstream s;
    s << _opengl_api_core->context_information()._version_major << "." 
      << _opengl_api_core->context_information()._version_minor << "." 
      << _opengl_api_core->context_information()._version_release;
    if (!_opengl_api_core->context_information()._version_info.empty())
         s << " " << _opengl_api_core->context_information()._version_info;
    if (!_opengl_api_core->context_information()._profile_string.empty())
         s << " " << _opengl_api_core->context_information()._profile_string;

    return s.str();
}

#if SCM_ENABLE_CUDA_CL_SUPPORT

bool
render_device::enable_cuda_interop()
{
    try {
        _cuda_device.reset(new cu::cuda_device());
        glout() << *_cuda_device << log::end;
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "render_device::enable_cuda_interop(): unable to initialize CUDA system ("
            << "evoking error: " << e.what() << ").";
        glerr() << msg.str() << log::end;

        _cuda_device.reset();

        return false;
    }

    if (!main_context()->enable_cuda_interop(cuda_interop_device())) {
        std::stringstream msg;
        msg << "render_device::enable_cuda_interop(): unable to initialize CUDA system ("
            << "unable to create CUDA command stream for main context" << ").";
        glerr() << msg.str() << log::end;
        return false;
    }

    return true;
}

bool
render_device::enable_opencl_interop()
{
    try {
        _opencl_device.reset(new cl::opencl_device());
        glout() << *_opencl_device << log::end;
    }
    catch (std::exception& e) {
        std::stringstream msg;
        msg << "render_device::enable_opencl_interop(): unable to initialize OpenCL system ("
            << "evoking error: " << e.what() << ").";
        glerr() << msg.str() << log::end;

        _opencl_device.reset();

        return false;
    }

    if (!main_context()->enable_opencl_interop(opencl_interop_device())) {
        std::stringstream msg;
        msg << "render_device::enable_opencl_interop(): unable to initialize OpenCL system ("
            << "unable to create OpenCL command queue for main context" << ").";
        glerr() << msg.str() << log::end;
        return false;
    }

    return true;
}


const cl::opencl_device_ptr
render_device::opencl_interop_device() const
{
    return _opencl_device;
}

const cu::cuda_device_ptr
render_device::cuda_interop_device() const
{
    return _cuda_device;
}
#endif

void
render_device::register_resource(render_device_resource* res_ptr)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        _registered_resources.insert(res_ptr);
    }
}

void
render_device::release_resource(render_device_resource* res_ptr)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        resource_ptr_set::iterator res_iter = _registered_resources.find(res_ptr);
        if (res_iter != _registered_resources.end()) {
            _registered_resources.erase(res_iter);
        }

        delete res_ptr;
    }
}

std::ostream& operator<<(std::ostream& os, const render_device& ren_dev)
{
    ren_dev.print_device_informations(os);
    return os;
}

} // namespace gl
} // namespace scm
