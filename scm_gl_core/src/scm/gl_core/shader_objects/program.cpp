
#include "program.h"

#include <cassert>
#include <cstring>
#include <iostream>

#include <boost/algorithm/string/predicate.hpp>

#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_type_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/shader_objects/shader.h>

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
#if SCM_GL_CORE_OPENGL_40
    void operator()(const double v) const           { _glapi.glUniform1dv(_uniform._location, _uniform._elements, &v); }
    void operator()(const math::vec2d& v) const     { _glapi.glUniform2dv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec3d& v) const     { _glapi.glUniform3dv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::vec4d& v) const     { _glapi.glUniform4dv(_uniform._location, _uniform._elements, v.data_array); }
    void operator()(const math::mat2d& m) const     { _glapi.glUniformMatrix2dv(_uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat3d& m) const     { _glapi.glUniformMatrix3dv(_uniform._location, _uniform._elements, false, m.data_array); }
    void operator()(const math::mat4d& m) const     { _glapi.glUniformMatrix4dv(_uniform._location, _uniform._elements, false, m.data_array); }
#endif
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

#if SCM_GL_CORE_OPENGL_40
    void operator()(const double v) const           { assert(0); }
    void operator()(const math::vec2d& v) const     { assert(0); }
    void operator()(const math::vec3d& v) const     { assert(0); }
    void operator()(const math::vec4d& v) const     { assert(0); }
    void operator()(const math::mat2d& m) const     { assert(0); }
    void operator()(const math::mat3d& m) const     { assert(0); }
    void operator()(const math::mat4d& m) const     { assert(0); }
#endif

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
            util::program_binding_guard save_guard(glapi);
            glapi.glUseProgram(_gl_program_obj);
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

    { // uniforms
        name_uniform_map::const_iterator u = _uniforms.begin();
        name_uniform_map::const_iterator e = _uniforms.end();
        for (; u != e; ++u) {
            if (u->second._update_required) {
                //std::cout << u->second._name << std::endl;
#if SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                boost::apply_visitor(detail::uniform_bind_dsa(glapi, _gl_program_obj, u->second), u->second._data);
#else
                boost::apply_visitor(detail::uniform_bind(glapi, u->second), u->second._data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                gl_assert(glapi, program::bind_uniforms() after apply uniform);

                u->second._update_required = false;
            }
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
#if 1
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
#endif
    gl_assert(glapi, leaving program::bind_uniforms());
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

void
program::retrieve_fragdata_information(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    // TODO

    gl_assert(glapi, leaving program::retrieve_fragdata_information());
}

void
program::retrieve_uniform_information(render_device& ren_dev)
{
    assert(_gl_program_obj != 0);

    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

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

            if (!boost::starts_with(actual_uniform_name, "gl_")) {
                actual_uniform_location = glapi.glGetUniformLocation(_gl_program_obj, actual_uniform_name.c_str());

                //std::cout << "uniform:       " << actual_uniform_name << " loc: " << actual_uniform_location << std::endl;
                
                if (util::is_sampler_type(actual_uniform_type)) {
                    // samplers as integer uniforms
                    _uniforms[actual_uniform_name] = uniform_type(actual_uniform_name,
                                                                  actual_uniform_location,
                                                                  actual_uniform_size,
                                                                  TYPE_INT);
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
    }
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
#if 1 //SCM_GL_CORE_OPENGL_40
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
                        std::string rname;
                        unsigned ri = comp_routines[r];
                        glapi.glGetActiveSubroutineName(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                        ri, act_routine_max_len + 20, 0, temp_name.get());
                gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);

                        rname.assign(temp_name.get());
                    }


                }
                gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine uniform info);
            }
            if (1){ // subroutines
                int act_routines = 0;
                int act_routine_max_len = 0;
                char*  temp_name = 0;
                glapi.glGetProgramStageiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                          GL_ACTIVE_SUBROUTINES, &act_routines);
                glapi.glGetProgramStageiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                          GL_ACTIVE_SUBROUTINE_MAX_LENGTH, &act_routine_max_len);
                if (act_routine_max_len > 0) {
                    temp_name = new char[act_routine_max_len + 1]; // reserve for null termination
                }
                for (int i = 1; i <= act_routines; ++i) {
                    std::string         actual_routine_name;
                    unsigned            actual_routine_index = 0;

                    int ret_size = 0;

                    glapi.glGetActiveSubroutineName(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                    unsigned(i), act_routine_max_len, &ret_size, temp_name);
                    actual_routine_name.assign(temp_name);
                    gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine info);

                    actual_routine_index = 
                        glapi.glGetSubroutineIndex(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                   temp_name);
                                                   //actual_routine_name.c_str());
                    gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine info);
                    // init the subroutine struct
                    subroutine_type actual_routine(actual_routine_name, actual_routine_index);
                    _subroutines[stge][actual_routine_name] = actual_routine;
                    gl_assert(glapi, program::retrieve_uniform_information() after retrieving subroutine info);
                }
                delete [] temp_name;
            }
#if 0
                // compatible routines
                int                 num_comp_routines = 0;
                scoped_array<int>   comp_routines;
                glapi.glGetActiveSubroutineUniformiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                     i, GL_NUM_COMPATIBLE_SUBROUTINES, &num_comp_routines);

    gl_assert(glapi, leaving program::retrieve_uniform_information());
                if (0 < num_comp_routines) {
                    comp_routines.reset(new int[num_comp_routines]);
                    glapi.glGetActiveSubroutineUniformiv(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                         i, GL_COMPATIBLE_SUBROUTINES, comp_routines.get());
                }

    gl_assert(glapi, leaving program::retrieve_uniform_information());
                for (int r = 0; r < num_comp_routines; ++r) {
                    std::string rname;
                    glapi.glGetActiveSubroutineName(_gl_program_obj, util::gl_shader_types(static_cast<shader_stage>(stge)),
                                                    comp_routines[r], max_act_routine_len, 0, temp_name.get());
    gl_assert(glapi, leaving program::retrieve_uniform_information());
                    rname.assign(temp_name.get());
                    actual_routine._routine_indices[rname] = comp_routines[r];
                }
#endif
        }
    }
#endif

    gl_assert(glapi, leaving program::retrieve_uniform_information());
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
