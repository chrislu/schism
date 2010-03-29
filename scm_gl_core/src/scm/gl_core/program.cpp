
#include "program.h"

#include <cassert>
#include <cstring>

#include <boost/algorithm/string/predicate.hpp>

#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_context.h>
#include <scm/gl_core/shader.h>
#include <scm/gl_core/opengl/config.h>
#include <scm/gl_core/opengl/gl3_core.h>
#include <scm/gl_core/opengl/util/assert.h>
#include <scm/gl_core/opengl/util/data_type_helper.h>
#include <scm/gl_core/opengl/util/error_helper.h>

namespace scm {
namespace gl {
namespace detail {

class uniform_bind : public boost::static_visitor<>
{
    const opengl::gl3_core&         _glapi;
    const program::variable_type&   _uniform;

public:
    uniform_bind(const opengl::gl3_core& api, const program::variable_type& u) : _glapi(api), _uniform(u) {}

    void operator()(const float v) const            { _glapi.glUniform1fv(_uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2f& v) const     { _glapi.glUniform2fv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3f& v) const     { _glapi.glUniform3fv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4f& v) const     { _glapi.glUniform4fv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::mat2f& m) const     { _glapi.glUniformMatrix2fv(_uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat3f& m) const     { _glapi.glUniformMatrix3fv(_uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat4f& m) const     { _glapi.glUniformMatrix4fv(_uniform._location, _uniform._elements, false, m.data_array); }

    void operator()(const int v) const              { _glapi.glUniform1iv(_uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2i& v) const     { _glapi.glUniform2iv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3i& v) const     { _glapi.glUniform3iv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4i& v) const     { _glapi.glUniform4iv(_uniform._location, _uniform._elements, v.data_array); }

    void operator()(const unsigned v) const         { _glapi.glUniform1uiv(_uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2ui& v) const    { _glapi.glUniform2uiv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3ui& v) const    { _glapi.glUniform3uiv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4ui& v) const    { _glapi.glUniform4uiv(_uniform._location, _uniform._elements, v.data_array); }
}; // class uniform_bind

class uniform_bind_dsa : public boost::static_visitor<>
{
    const opengl::gl3_core&         _glapi;
    unsigned                        _program;
    const program::variable_type&   _uniform;

public:
    uniform_bind_dsa(const opengl::gl3_core& api, unsigned p, const program::variable_type& u) : _glapi(api), _program(p), _uniform(u) {}

    void operator()(const float v) const            { _glapi.glProgramUniform1fvEXT(_program, _uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2f& v) const     { _glapi.glProgramUniform2fvEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3f& v) const     { _glapi.glProgramUniform3fvEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4f& v) const     { _glapi.glProgramUniform4fvEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::mat2f& m) const     { _glapi.glProgramUniformMatrix2fvEXT(_program, _uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat3f& m) const     { _glapi.glProgramUniformMatrix3fvEXT(_program, _uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat4f& m) const     { _glapi.glProgramUniformMatrix4fvEXT(_program, _uniform._location, _uniform._elements, false, m.data_array); }

    void operator()(const int v) const              { _glapi.glProgramUniform1ivEXT(_program, _uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2i& v) const     { _glapi.glProgramUniform2ivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3i& v) const     { _glapi.glProgramUniform3ivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4i& v) const     { _glapi.glProgramUniform4ivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }

    void operator()(const unsigned v) const         { _glapi.glProgramUniform1uivEXT(_program, _uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2ui& v) const    { _glapi.glProgramUniform2uivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3ui& v) const    { _glapi.glProgramUniform3uivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4ui& v) const    { _glapi.glProgramUniform4uivEXT(_program, _uniform._location, _uniform._elements, v.data_array); }
}; // class uniform_bind_dsa

} // namespace detail

program::program(render_device&                 ren_dev,
                 const shader_list&             in_shaders,
                 const named_location_list&     in_attibute_locations,
                 const named_location_list&     in_fragment_locations)
  : render_device_child(ren_dev)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
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
        // set default attribute locations
        foreach(const named_location& l, in_attibute_locations) {
            glapi.glBindAttribLocation(_gl_program_obj, l.second, l.first.c_str());
            gl_assert(glapi, program::program() binding attribute location);
        }
        // set default fragdata locations
        foreach(const named_location& l, in_fragment_locations) {
            glapi.glBindFragDataLocation(_gl_program_obj, l.second, l.first.c_str());
            gl_assert(glapi, program::program() binding fragdata location);
        }
        // link program
        link(ren_dev);

        // retrieve information
        if (ok()) {
            retrieve_attribute_information(ren_dev);
            retrieve_fragdata_information(ren_dev);
            retrieve_uniform_information(ren_dev);
        }
    }
    
    gl_assert(glapi, leaving program::program());
}

program::~program()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    // TODO detach all shaders and remove them from _shaders;

    assert(0 != _gl_program_obj);
    glapi.glDeleteProgram(_gl_program_obj);

    gl_assert(glapi, leaving program::~program());
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

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
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

    const opengl::gl3_core& glapi = ren_ctx.opengl_api();

    glapi.glUseProgram(_gl_program_obj);

    gl_assert(glapi, leaving program:bind());
}

void
program::bind_uniforms(render_context& ren_ctx) const
{
    assert(_gl_program_obj != 0);
    assert(state().ok());
    //assert();

    const opengl::gl3_core& glapi = ren_ctx.opengl_api();

    name_uniform_map::iterator u = _uniforms.begin();
    name_uniform_map::iterator e = _uniforms.end();
    for (; u != e; ++u) {
        if (u->second._update_required) {
#if SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            boost::apply_visitor(detail::uniform_bind_dsa(glapi, _gl_program_obj, u->second), u->second._data);
#else
            boost::apply_visitor(detail::uniform_bind(glapi, u->second), u->second._data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            u->second._update_required = false;
        }
    }

    gl_assert(glapi, leaving program:bind_uniforms());
}

void
program::retrieve_attribute_information(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
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
                                act_attib_max_len + 1,      // max lenght of attrib name incl null term
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

void
program::retrieve_fragdata_information(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    gl_assert(glapi, leaving program::retrieve_fragdata_information());
}

void
program::retrieve_uniform_information(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

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

        if (!boost::starts_with(actual_uniform_name, "gl_")) {
            actual_uniform_location = glapi.glGetUniformLocation(_gl_program_obj, actual_uniform_name.c_str());
            if (util::is_sampler_type(actual_uniform_type)) {
                // TODO
                assert(false);
            }
            else {
                assert(util::from_gl_data_type(actual_uniform_type) != TYPE_UNKNOWN);
                _uniforms[actual_uniform_name] = uniform_type(actual_uniform_name,
                                                              actual_uniform_location,
                                                              actual_uniform_size,
                                                              util::from_gl_data_type(actual_uniform_type));
            }
        }
    }

    gl_assert(glapi, leaving program::retrieve_uniform_information());
}
#if 0
void
program::uniform_raw(uniform_type& u, const void *data, const int size)
{
    //if (std::memcmp(u._data_array.get(), data, size)) {
    //    if (std::memcpy(u._data_array.get(), data, size) != u._data_array.get()) {
    //        SCM_GL_DGB("program::uniform_raw(): error copying uniform data.");
    //    }
    //    else {
    //        u._update_required = true;
    //    }
    //}
}

void
program::uniform(const std::string& name, float value)
{
    name_uniform_map::iterator  u = _uniforms.find(name);

    if (u != _uniforms.end()) {
        if (uniform_type_id<float>::id != u->second._type) {

        }
        if (u->second._type == TYPE_FLOAT) {
            u->second._data = value;
            //uniform_raw(u->second, &value, sizeof(value));
        }
        else {
            SCM_GL_DGB("program::uniform(): found uniform not of type float ('" << name << "').");
        }
    }
    else {
        SCM_GL_DGB("program::uniform(): unable to find uniform ('" << name << "').");
    }
}

void
program::uniform(const std::string& name, const math::vec2f& value)
{
    name_uniform_map::iterator  u = _uniforms.find(name);

    if (u != _uniforms.end()) {
        if (u->second._type == TYPE_FLOAT) {
            u->second._data = value;
            //uniform_raw(u->second, &value, sizeof(value));
        }
        else {
            SCM_GL_DGB("program::uniform(): found uniform not of type float ('" << name << "').");
        }
    }
    else {
        SCM_GL_DGB("program::uniform(): unable to find uniform ('" << name << "').");
    }
}

void
program::uniform(const std::string& name, const math::vec3f& value)
{
}

void
program::uniform(const std::string& name, const math::vec4f& value)
{
}

void
program::uniform(const std::string& name, int value)
{
}

void
program::uniform(const std::string& name, const math::vec2i& value)
{
}

void
program::uniform(const std::string& name, const math::vec3i& value)
{
}

void
program::uniform(const std::string& name, const math::vec4i& value)
{
}

void
program::uniform(const std::string& name, unsigned value)
{
}

void
program::uniform(const std::string& name, const math::vec2ui& value)
{
}

void
program::uniform(const std::string& name, const math::vec3ui& value)
{
}

void
program::uniform(const std::string& name, const math::vec4ui& value)
{
}

void
program::uniform(const std::string& name, const math::mat2f& value, bool transpose)
{
}

void
program::uniform(const std::string& name, const math::mat3f& value, bool transpose)
{
}

void
program::uniform(const std::string& name, const math::mat4f& value, bool transpose)
{
}

#endif

} // namespace gl
} // namespace scm
