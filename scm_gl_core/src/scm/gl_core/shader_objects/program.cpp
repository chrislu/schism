
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "program.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <scm/core/memory.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/shader_objects/shader.h>
#include <scm/gl_core/shader_objects/stream_capture.h>

namespace scm {
namespace gl {
namespace detail {
} // namespace detail

program::program(render_device&              in_device,
                 const shader_list&          in_shaders,
                 const stream_capture_array& in_capture,
                 bool                        in_rasterization_discard,
                 const named_location_list&  in_attribute_locations,
                 const named_location_list&  in_fragment_locations)
  : render_device_child(in_device)
  , _rasterization_discard(in_rasterization_discard)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    _gl_program_obj = glapi.glCreateProgram();
    if (0 == _gl_program_obj) {
        state().set(object_state::OS_BAD);
    }
    else {
        // attach all shaders
        foreach(const shader_ptr& s, in_shaders) {
            if (s) {
                glapi.glAttachShader(_gl_program_obj, s->_gl_shader_obj);
                if (!glerror) {
                    _shaders.push_back(s);
                }
                else {
                    state().set(object_state::OS_ERROR_INVALID_VALUE);
                }
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
            }
        }
        gl_assert(glapi, program::program() attaching shader objects);
        // set the captured transform feedback varyings
        if (!in_capture.empty()) {
            if (!apply_transform_feedback_varyings(in_device, in_capture)) {
                // error code set in function itself
                return;
            }
        }
        // set default attribute locations
        foreach(const named_location& l, in_attribute_locations) {
            glapi.glBindAttribLocation(_gl_program_obj, l.second, l.first.c_str());
            gl_assert(glapi, program::program() binding attribute location);
        }
        // set default fragdata locations
        foreach(const named_location& l, in_fragment_locations) {
            glapi.glBindFragDataLocation(_gl_program_obj, l.second, l.first.c_str());
            gl_assert(glapi, program::program() binding fragdata location);
        }
        // link program
        link(in_device);

        // retrieve information
        if (ok()) {
            util::program_binding_guard save_guard(glapi);
            glapi.glUseProgram(_gl_program_obj);
            retrieve_attribute_information(in_device);
            retrieve_fragdata_information(in_device);
            retrieve_uniform_information(in_device);
        }
    }
    
    gl_assert(glapi, leaving program::program());
}

program::~program()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();

    // TODO detach all shaders and remove them from _shaders;

    assert(0 != _gl_program_obj);
    glapi.glDeleteProgram(_gl_program_obj);

    gl_assert(glapi, leaving program::~program());
}

unsigned
program::program_id() const
{
    return _gl_program_obj;
}

const std::string&
program::info_log() const
{
    return (_info_log);
}

bool
program::link(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl_core& glapi = ren_dev.opengl_api();
    util::gl_error          glerror(glapi);

    int link_state  = 0;

    glapi.glLinkProgram(_gl_program_obj);
    glapi.glGetProgramiv(_gl_program_obj, GL_LINK_STATUS, &link_state);

    if (GL_TRUE != link_state) {
        state().set(object_state::OS_ERROR_SHADER_LINK);
    }

    int info_len = 0;
    glapi.glGetProgramiv(_gl_program_obj, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len > 10) {
        _info_log.clear();
        _info_log.resize(info_len);
        assert(_info_log.capacity() >= info_len);
        glapi.glGetProgramInfoLog(_gl_program_obj, info_len, NULL, &_info_log[0]);
    }

    gl_assert(glapi, leaving program:link());

    return (GL_TRUE == link_state);
}

bool
program::validate(render_context& ren_ctx)
{
    return (false);
}

void
program::bind(render_context& ren_ctx) const
{
    assert(_gl_program_obj != 0);
    assert(state().ok());

    const opengl::gl_core& glapi = ren_ctx.opengl_api();

    glapi.glUseProgram(_gl_program_obj);

    gl_assert(glapi, leaving program:bind());
}

void
program::bind_uniforms(render_context& ren_ctx) const
{
    assert(_gl_program_obj != 0);
    assert(state().ok());
    //assert();

    const opengl::gl_core& glapi = ren_ctx.opengl_api();

    { // uniforms
        name_uniform_map::const_iterator u = _uniforms.begin();
        name_uniform_map::const_iterator e = _uniforms.end();
        for (; u != e; ++u) {
            //std::cout << u->first << std::endl;
            if (u->second->update_required()) {
                u->second->apply_value(ren_ctx, *this);
                u->second->_status._update_required = false;
            }
            //if (u->second._update_required) {
                //std::cout << u->second._name << std::endl;
//#if SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
//                boost::apply_visitor(detail::uniform_bind_dsa(glapi, _gl_program_obj, u->second), u->second._data);
//#else
//                boost::apply_visitor(detail::uniform_bind(glapi, u->second), u->second._data);
//#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            //    gl_assert(glapi, program::bind_uniforms() after apply uniform);

            //    u->second._update_required = false;
            //}
        }
    }
    { // uniform buffers
        name_uniform_block_map::const_iterator b = _uniform_blocks.begin();
        name_uniform_block_map::const_iterator e = _uniform_blocks.end();
        for (; b != e; ++b) {
            if (b->second._update_required) {
                glapi.glUniformBlockBinding(_gl_program_obj, b->second._block_index, b->second._binding);
                b->second._update_required = false;

                gl_assert(glapi, program::bind_uniforms() after glUniformBlockBinding());
            }
        }
    }
    { // storage buffers
        name_storage_buffer_map::const_iterator b = _storage_buffers.begin();
        name_storage_buffer_map::const_iterator e = _storage_buffers.end();
        for (; b != e; ++b) {
            if (b->second._update_required) {
                glapi.glShaderStorageBlockBinding(_gl_program_obj, b->second._index, b->second._binding);
                b->second._update_required = false;

                gl_assert(glapi, program::bind_uniforms() after glShaderStorageBlockBinding());
            }
        }
    }
    { // subroutines
        for (int s = 0; s < SHADER_STAGE_COUNT; ++s) {
            name_subroutine_uniform_map::const_iterator b = _subroutine_uniforms[s].begin();
            name_subroutine_uniform_map::const_iterator e = _subroutine_uniforms[s].end();
            int indices_size = static_cast<int>(_subroutine_uniforms[s].size());
            scoped_array<unsigned>  indices;
            if (0 < indices_size) {
                indices.reset(new unsigned[indices_size]);
            }
            for (; b != e; ++b) {
                int       l  = b->second._location;
                indices[l] = b->second._selected_routine;
            }
            if (0 < indices_size) {
                glapi.glUniformSubroutinesuiv(util::gl_shader_types(static_cast<shader_stage>(s)), indices_size, indices.get());
            }
        }
    }

    gl_assert(glapi, leaving program::bind_uniforms());
}

bool
program::apply_transform_feedback_varyings(render_device& in_device, const stream_capture_array& in_capture)
{
    assert(_gl_program_obj != 0);

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    int  max_stream_count    = 0;

    // error handling
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        // GL 4.0+
        max_stream_count = in_device.capabilities()._max_transform_feedback_buffers;
        if (in_capture.used_streams() > max_stream_count) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            _info_log =   std::string("program::apply_transform_feedback_varyings(): unsupported amount of feedback buffer encountered ")
                        + std::string("(feedback buffers: ")
                        + boost::lexical_cast<std::string>(in_capture.used_streams())
                        + std::string(", max supported: ")
                        + boost::lexical_cast<std::string>(max_stream_count)
                        + std::string(").");
            return false;
        }
    }
    else {
        // GL 3.3+
        max_stream_count = in_device.capabilities()._max_transform_feedback_separate_attribs;
        if (in_capture.interleaved_streams()) {
            if (in_capture.used_streams() > 1) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                _info_log = "program::apply_transform_feedback_varyings(): OpenGL 3.3+ only allows for a single interleaved transform feedback stream.";
                return false;
            }
            if (in_capture.interleaved_skipped_components()) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                _info_log = "program::apply_transform_feedback_varyings(): OpenGL 3.3+ does not allow for skipped interleaved components.";
                return false;
            }
        }
        else {
            if (in_capture.used_streams() > max_stream_count) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                _info_log =   std::string("program::apply_transform_feedback_varyings(): OpenGL 3.3+ only allows for ")
                            + boost::lexical_cast<std::string>(max_stream_count)
                            + std::string(" separate transform feedback streams.");
                return false;
            }
        }
    }

    assert(max_stream_count > 0);

    static const char* next_buffer_string =
        "gl_NextBuffer";
    static const char* skip_strings[] = {
        "gl_SkipComponents1",
        "gl_SkipComponents2",
        "gl_SkipComponents3",
        "gl_SkipComponents4"
    };
    BOOST_STATIC_ASSERT((sizeof(skip_strings) / sizeof(const char*)) == (stream_capture::skip_4_float + 1));

    std::vector<std::string>    varying_names;
    varying_names.reserve(in_capture.captures_count() + max_stream_count);
    for (int stream = 0; stream < in_capture.used_streams(); ++stream) {
        if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
            if (stream > 0 && in_capture.interleaved_streams()) {
                varying_names.push_back(next_buffer_string);
            }
        }

        const stream_capture::captures_list& captures = in_capture.stream_captures(stream).captures();
        if (!captures.empty()) {
            stream_capture::captures_list::const_iterator b = captures.begin();
            stream_capture::captures_list::const_iterator e = captures.end();
            for (; b != e; ++b) {
                stream_capture::capture_element cap = *b;
                if (boost::get<std::string>(&cap)) {
                    varying_names.push_back(boost::get<std::string>(cap));
                }
                else if (boost::get<stream_capture::skip_components_type>(&cap)) {
                    std::string skip_string = skip_strings[boost::get<stream_capture::skip_components_type>(cap)];
                    varying_names.push_back(skip_string);
                }
            }
        }
    }

    shared_array<const char*> varying_name_ptrs(new const char*[varying_names.size()]);
    for (int i = 0; i < varying_names.size(); ++i) {
        const char* name_string = varying_names[i].c_str();
        varying_name_ptrs[i] = name_string;
    }

    glapi.glTransformFeedbackVaryings(_gl_program_obj,
                                      static_cast<int>(varying_names.size()),
                                      varying_name_ptrs.get(),
                                      in_capture.interleaved_streams() ? GL_INTERLEAVED_ATTRIBS : GL_SEPARATE_ATTRIBS);

    if (glerror) {
        state().set(glerror.to_object_state());
        return false;
    }

    gl_assert(glapi, leaving program::apply_transform_feedback_varyings());

    return true;
}

void
program::retrieve_attribute_information(render_device& in_device)
{
    assert(_gl_program_obj != 0);

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    int act_attribs = 0;
    int act_attib_max_len = 0;
    scoped_array<char>  temp_name;
    glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_ATTRIBUTES, &act_attribs);
    glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &act_attib_max_len);
    if (act_attib_max_len > 0) {
        temp_name.reset(new char[act_attib_max_len + 1]); // reserve for null termination
    }
    for (int i = 0; i < act_attribs; ++i) {
        int             actual_attrib_size = 0;
        unsigned        actual_attrib_type = 0;
        int             actual_attrib_location = -1;
        std::string     actual_attrib_name;

        glapi.glGetActiveAttrib(_gl_program_obj,
                                i,                          // attribute index
                                act_attib_max_len + 1,      // max length of attrib name incl null term
                                0,                          // returned name length
                                &actual_attrib_size,        // attribute size (multiples of type size)
                                &actual_attrib_type,        // type
                                temp_name.get());           // name
        actual_attrib_name.assign(temp_name.get());

        if (!boost::starts_with(actual_attrib_name, "gl_")) {
            actual_attrib_location = glapi.glGetAttribLocation(_gl_program_obj, actual_attrib_name.c_str());
            assert(util::from_gl_data_type(actual_attrib_type) != TYPE_UNKNOWN);
            _attributes[actual_attrib_name] = variable_type(actual_attrib_name,
                                                            actual_attrib_location,
                                                            actual_attrib_size,
                                                            util::from_gl_data_type(actual_attrib_type));
        }
    }

    gl_assert(glapi, leaving program::retrieve_attribute_information());
}

int
program::attribute_location(const std::string& name) const
{
    name_variable_map::const_iterator a = _attributes.find(name);
    if (a != _attributes.end()) {
        return (a->second._location);
    }
    return (-1);
}

bool
program::rasterization_discard() const
{
    return _rasterization_discard;
}

void
program::retrieve_fragdata_information(render_device& in_device)
{
    assert(_gl_program_obj != 0);

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    // TODO

    gl_assert(glapi, leaving program::retrieve_fragdata_information());
}

void
program::retrieve_uniform_information(render_device& in_device)
{
    assert(_gl_program_obj != 0);

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    { // uniform blocks
        int act_uniforms = 0;
        int act_uniform_max_len = 0;
        scoped_array<char>  temp_name;
        glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_UNIFORM_BLOCKS, &act_uniforms);
        glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &act_uniform_max_len);
        if (act_uniform_max_len > 0) {
            temp_name.reset(new char[act_uniform_max_len + 1]); // reserve for null termination
        }
        for (int i = 0; i < act_uniforms; ++i) {
            int             actual_uniform_size = 0;
            int             actual_uniform_location = -1;
            std::string     actual_uniform_name;

            glapi.glGetActiveUniformBlockiv(_gl_program_obj, i, GL_UNIFORM_BLOCK_DATA_SIZE, &actual_uniform_size);
            glapi.glGetActiveUniformBlockName(_gl_program_obj, i, act_uniform_max_len, 0, temp_name.get());
            actual_uniform_name.assign(temp_name.get());
            actual_uniform_location = glapi.glGetUniformBlockIndex(_gl_program_obj, actual_uniform_name.c_str());
            
            //std::cout << "uniform block: " << actual_uniform_name << " size: " << actual_uniform_size << std::endl;

            _uniform_blocks[actual_uniform_name] = uniform_block_type(actual_uniform_name,
                                                                      actual_uniform_location,
                                                                      actual_uniform_size);
        }
    }
    { // uniforms
        int act_uniforms = 0;
        int act_uniform_max_len = 0;
        scoped_array<char>  temp_name;
        glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_UNIFORMS, &act_uniforms);
        glapi.glGetProgramiv(_gl_program_obj, GL_ACTIVE_UNIFORM_MAX_LENGTH, &act_uniform_max_len);
        if (act_uniform_max_len > 0) {
            temp_name.reset(new char[act_uniform_max_len + 1]); // reserve for null termination
        }
        for (int i = 0; i < act_uniforms; ++i) {
            int             actual_uniform_size = 0;
            unsigned        actual_uniform_type = 0;
            int             actual_uniform_location = -1;
            std::string     actual_uniform_name;

            glapi.glGetActiveUniform(_gl_program_obj,
                                    i,                          // attribute index
                                    act_uniform_max_len + 1,    // max lenght of attrib name incl null term
                                    0,                          // returned name length
                                    &actual_uniform_size,       // attribute size (multiples of type size)
                                    &actual_uniform_type,       // type
                                    temp_name.get());           // name
            actual_uniform_name.assign(temp_name.get());

            // filter out uniform block elements
            size_t  dot_delim_pos = actual_uniform_name.find('.');
            if (dot_delim_pos != std::string::npos) {
                std::string parent_name = actual_uniform_name.substr(0, dot_delim_pos);
                if (_uniform_blocks.count(parent_name) > 0) {
                    continue;
                }
                //std::cout << actual_uniform_name << std::endl;
                //continue;
            }

            // uniform arrays may be reported as {array_name}[0]
            // truncate to {array_name}
            size_t br_delim_pos = actual_uniform_name.find('[');
            if (br_delim_pos != std::string::npos) {
                if (actual_uniform_name[br_delim_pos + 1] != '0') {
                    continue;
                } else {
                    actual_uniform_name = actual_uniform_name.substr(0, br_delim_pos);
                }
            }

            if (!boost::starts_with(actual_uniform_name, "gl_")) {
                actual_uniform_location = glapi.glGetUniformLocation(_gl_program_obj, actual_uniform_name.c_str());

                uniform_ptr current_uniform;

                if (util::is_sampler_type(actual_uniform_type)) { // samplers as integer uniforms
                    current_uniform.reset(new scm::gl::uniform_sampler(actual_uniform_name, actual_uniform_location, actual_uniform_size, TYPE_SAMPLER));
                }
                else if (util::is_image_type(actual_uniform_type)) { // images as integer uniforms
                    current_uniform.reset(new scm::gl::uniform_image(actual_uniform_name, actual_uniform_location, actual_uniform_size, TYPE_IMAGE));
                }
                else {
                    switch (actual_uniform_type) {
                        case GL_FLOAT:                  current_uniform.reset(new scm::gl::uniform_1f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_VEC2:             current_uniform.reset(new scm::gl::uniform_vec2f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_VEC3:             current_uniform.reset(new scm::gl::uniform_vec3f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_VEC4:             current_uniform.reset(new scm::gl::uniform_vec4f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_MAT2:             current_uniform.reset(new scm::gl::uniform_mat2f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_MAT3:             current_uniform.reset(new scm::gl::uniform_mat3f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_FLOAT_MAT4:             current_uniform.reset(new scm::gl::uniform_mat4f(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
                        case GL_DOUBLE:                 current_uniform.reset(new scm::gl::uniform_1d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_VEC2:            current_uniform.reset(new scm::gl::uniform_vec2d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_VEC3:            current_uniform.reset(new scm::gl::uniform_vec3d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_VEC4:            current_uniform.reset(new scm::gl::uniform_vec4d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_MAT2:            current_uniform.reset(new scm::gl::uniform_mat2d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_MAT3:            current_uniform.reset(new scm::gl::uniform_mat3d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_DOUBLE_MAT4:            current_uniform.reset(new scm::gl::uniform_mat4d(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
                        case GL_INT:                    current_uniform.reset(new scm::gl::uniform_1i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_INT_VEC2:               current_uniform.reset(new scm::gl::uniform_vec2i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_INT_VEC3:               current_uniform.reset(new scm::gl::uniform_vec3i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_INT_VEC4:               current_uniform.reset(new scm::gl::uniform_vec4i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;

                        case GL_BOOL:                   current_uniform.reset(new scm::gl::uniform_1i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_BOOL_VEC2:              current_uniform.reset(new scm::gl::uniform_vec2i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_BOOL_VEC3:              current_uniform.reset(new scm::gl::uniform_vec3i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_BOOL_VEC4:              current_uniform.reset(new scm::gl::uniform_vec4i(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;

                        case GL_UNSIGNED_INT:           current_uniform.reset(new scm::gl::uniform_1ui(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_UNSIGNED_INT_VEC2:      current_uniform.reset(new scm::gl::uniform_vec2ui(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_UNSIGNED_INT_VEC3:      current_uniform.reset(new scm::gl::uniform_vec3ui(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
                        case GL_UNSIGNED_INT_VEC4:      current_uniform.reset(new scm::gl::uniform_vec4ui(actual_uniform_name, actual_uniform_location, actual_uniform_size, util::from_gl_data_type(actual_uniform_type)));break;
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420
                        case GL_UNSIGNED_INT_ATOMIC_COUNTER:break; // uniform atomic counters are just discovered here... nothing to save
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420
                        default:
                            glerr() << log::error << "program::retrieve_uniform_information(): "
                                    << "unsupported uniform type for uniform '" << actual_uniform_name << "'." << log::end;
                    }
                }

                if (current_uniform) {
                    _uniforms[actual_uniform_name] = current_uniform;
                }
            }
        }
    }
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    { // subroutines
        for (int stge = 0; stge < SHADER_STAGE_COUNT; ++stge) {
            { // subrountine uniforms
                int act_routines = 0;
                int act_routine_max_len = 0;
                scoped_array<char>  temp_name;
                glapi.glGetProgramStageiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                          GL_ACTIVE_SUBROUTINE_UNIFORMS, &act_routines);
                glapi.glGetProgramStageiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                          GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH, &act_routine_max_len);
                if (act_routine_max_len > 0) {
                    temp_name.reset(new char[act_routine_max_len + 1]); // reserve for null termination
                }
                for (int i = 0; i < act_routines; ++i) {
                    std::string         actual_routine_name;
                    int                 actual_routine_location = -1;

                    glapi.glGetActiveSubroutineUniformName(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                           i, act_routine_max_len, 0, temp_name.get());
                    actual_routine_name.assign(temp_name.get());

                    actual_routine_location = 
                        glapi.glGetSubroutineUniformLocation(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                             actual_routine_name.c_str());
                    // init the subroutine struct
                    subroutine_uniform_type actual_routine(actual_routine_name, actual_routine_location);
                    _subroutine_uniforms[stge][actual_routine_name] = actual_routine;


                    // compatible routines
                    int num_comp_routines = 0;
                    scoped_array<int> comp_routines;
                    glapi.glGetActiveSubroutineUniformiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                         i, GL_NUM_COMPATIBLE_SUBROUTINES, &num_comp_routines);
                    gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);
                    if (0 < num_comp_routines) {
                        temp_name.reset(new char[act_routine_max_len + 20]);
                        comp_routines.reset(new int[num_comp_routines]);
                        glapi.glGetActiveSubroutineUniformiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                             i, GL_COMPATIBLE_SUBROUTINES, comp_routines.get());
                        gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);
                    }

                    for (int r = 0; r < num_comp_routines; ++r) {
                        std::string actual_routine_name;
                        unsigned    actual_routine_index = 0;

                        unsigned ri = comp_routines[r];
                        glapi.glGetActiveSubroutineName(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                        ri, act_routine_max_len + 20, 0, temp_name.get());
                        gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);
                        actual_routine_name.assign(temp_name.get());

                        actual_routine_index = 
                        glapi.glGetSubroutineIndex(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                   temp_name.get());
                                                   //actual_routine_name.c_str());
                        gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine info);

                        // init the subroutine struct
                        subroutine_type actual_routine(actual_routine_name, actual_routine_index);
                        _subroutines[stge][actual_routine_name] = actual_routine;
                        gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine info);
                    }
                }
                gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);
            }
        }
    }
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

    // shader storage buffers
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_430) {
        int act_sblocks      = 0;
        int act_sblocks_nlen = 0;
        scoped_array<char>  temp_name;

        glapi.glGetProgramInterfaceiv(_gl_program_obj, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &act_sblocks);
        glapi.glGetProgramInterfaceiv(_gl_program_obj, GL_SHADER_STORAGE_BLOCK, GL_MAX_NAME_LENGTH,  &act_sblocks_nlen);
        gl_assert(glapi, program::retrieve_uniform_information() after retrieving shader storage block infos);

        act_sblocks_nlen = math::max(act_sblocks_nlen, 100);

        temp_name.reset(new char[act_sblocks_nlen + 1]); // reserve for null termination

        for (int i = 0; i < act_sblocks; ++i) {
            int         sblock_size     = 0;
            int         sblock_location = -1;
            int         sblock_idx       = -1;
            std::string sblock_name;


            glapi.glGetProgramResourceName(_gl_program_obj, GL_SHADER_STORAGE_BLOCK, i, act_sblocks_nlen, 0, temp_name.get());
            sblock_name.assign(temp_name.get());

            sblock_location = glapi.glGetProgramResourceIndex(_gl_program_obj, GL_SHADER_STORAGE_BLOCK, sblock_name.c_str());

            GLenum      sblock_params[2]       = {GL_BUFFER_DATA_SIZE, GL_BUFFER_BINDING };
            int         sblock_param_values[2] = {-1, -1};
            glapi.glGetProgramResourceiv(_gl_program_obj, GL_SHADER_STORAGE_BLOCK, i, 2, sblock_params, 2, 0, sblock_param_values);
            
            sblock_size = sblock_param_values[0];
            sblock_idx  = sblock_param_values[1];

            _storage_buffers[sblock_name] = storage_buffer_type(sblock_name,
                                                                sblock_location,
                                                                sblock_size,
                                                                sblock_idx);

            gl_assert(glapi, program::retrieve_uniform_information() after retrieving shader storage block infos);
        }
    }

    gl_assert(glapi, leaving program::retrieve_uniform_information());
}

uniform_ptr
program::uniform_raw(const std::string& name) const
{
    name_uniform_map::const_iterator  u = _uniforms.find(name);
    if (u != _uniforms.end()) {
        return (u->second);
    }
    else {
        return (uniform_ptr());
    }
}

void
program::uniform_sampler(const std::string& name, scm::int32 u)
{
    if (uniform_sampler_ptr p = uniform_sampler(name)) {
        p->bound_unit(u);
    }
    else {
        SCM_GL_DGB("program::uniform_sampler(): unable to find uniform sampler ('" << name << "').");
    }
}

void
program::uniform_sampler_handle(const std::string& name, scm::uint64 h)
{
    if (uniform_sampler_ptr p = uniform_sampler(name)) {
        p->resident_handle(h);
    }
    else {
        SCM_GL_DGB("program::uniform_sampler_handle(): unable to find uniform sampler ('" << name << "').");
    }
}

void
program::uniform_image(const std::string& name, scm::int32 u)
{
    if (uniform_image_ptr p = uniform_image(name)) {
        p->bound_unit(u);
    }
    else {
        SCM_GL_DGB("program::uniform_image(): unable to find uniform image ('" << name << "').");
    }
}

void
program::uniform_image_handle(const std::string& name, scm::uint64 h)
{
    if (uniform_image_ptr p = uniform_image(name)) {
        p->resident_handle(h);
    }
    else {
        SCM_GL_DGB("program::uniform_image_handle(): unable to find uniform image ('" << name << "').");
    }
}

void
program::uniform_buffer(const std::string& name, const unsigned binding)
{
    name_uniform_block_map::iterator  u = _uniform_blocks.find(name);

    if (u != _uniform_blocks.end()) {
        if (u->second._binding != binding) {
            u->second._binding = binding;
            u->second._update_required = true;
        }
    }
    else {
        SCM_GL_DGB("program::uniform_buffer(): unable to find uniform buffer ('" << name << "').");
    }
}

void
program::uniform_subroutine(const shader_stage stage, const std::string& name, const std::string& routine)
{
#if 1
    name_subroutine_uniform_map::iterator subr = _subroutine_uniforms[stage].find(name);
    name_subroutine_map::iterator         rout = _subroutines[stage].find(routine);

    if (subr != _subroutine_uniforms[stage].end()) {
        if (rout != _subroutines[stage].end()) {
            subr->second._selected_routine = rout->second._index;
        }
        else {
            SCM_GL_DGB("program::uniform_subroutine(): unable to find routine ('" << name << "').");
        }
    }
    else {
        SCM_GL_DGB("program::uniform_subroutine(): unable to find subroutine ('" << name << "').");
    }
#endif
}

void
program::storage_buffer(const std::string& name, const unsigned binding)
{
    name_storage_buffer_map::iterator  u = _storage_buffers.find(name);

    if (u != _storage_buffers.end()) {
        if (u->second._binding != binding) {
            u->second._binding = binding;
            u->second._update_required = true;

            if (u->second._binding != u->second._static_binding) {
                SCM_GL_DGB("program::storage_buffer(): overriding shader defined binding on storage buffer ('" << name << "').");
            }
        }
    }
    else {
        SCM_GL_DGB("program::storage_buffer(): unable to find storage buffer ('" << name << "').");
    }
}




} // namespace gl
} // namespace scm
