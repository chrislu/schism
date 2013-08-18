
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PROGRAM_H_INCLUDED
#define SCM_GL_CORE_PROGRAM_H_INCLUDED

#include <list>
#include <vector>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/variant.hpp>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>
#include <scm/gl_core/shader_objects/uniform.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) program : public render_device_child
{
public:
    typedef boost::unordered_map<std::string, int>  name_location_map;
    typedef std::pair<std::string, int>             named_location;
    typedef std::list<named_location>               named_location_list;
    typedef std::list<shader_ptr>                   shader_list;

    // program information
    struct variable_type {
        variable_type() : _location(-1), _type(TYPE_UNKNOWN), _elements(0) {}
        variable_type(const std::string& n, int l, unsigned e, data_type t) : _name(n), _location(l), _type(t), _elements(e) {}
        std::string     _name;
        int             _location;
        unsigned        _elements;
        data_type       _type;
    };
    struct uniform_block_type {
        uniform_block_type() : _block_index(-1), _size(-1), _binding(-1), _update_required(false) {}
        uniform_block_type(const std::string& n, int i, scm::size_t s) : _name(n), _block_index(i), _size(s), _binding(-1), _update_required(false) {}
        std::string     _name;
        int             _block_index;
        scm::size_t     _size;
        int             _binding;
        mutable bool    _update_required;
    };
    struct subroutine_uniform_type {
        subroutine_uniform_type()  : _location(-1), _selected_routine(-1) {}
        subroutine_uniform_type(const std::string& n, int l) : _name(n), _location(l), _selected_routine(0) {}
        std::string         _name;
        int                 _location;
        //name_location_map   _routine_indices;
        unsigned            _selected_routine;
    };
    struct subroutine_type {
        subroutine_type() : _index(0) {}
        subroutine_type(const std::string& n, unsigned l) : _name(n), _index(l) {}
        std::string         _name;
        unsigned            _index;
    };
    struct storage_buffer_type {
        storage_buffer_type() : _index(-1), _size(-1), _binding(-1), _update_required(false) {}
        storage_buffer_type(const std::string& n, int i, scm::size_t s, int b) : _name(n), _index(i), _size(s), _static_binding(b), _binding(-1), _update_required(false) {}
        std::string     _name;
        int             _index;
        scm::size_t     _size;
        int             _static_binding;
        int             _binding;
        mutable bool    _update_required;
    };


    typedef boost::unordered_map<std::string, variable_type>            name_variable_map;
    typedef boost::unordered_map<std::string, uniform_ptr>              name_uniform_map;
    typedef boost::unordered_map<std::string, uniform_block_type>       name_uniform_block_map;
    typedef boost::unordered_map<std::string, subroutine_uniform_type>  name_subroutine_uniform_map;
    typedef boost::unordered_map<std::string, subroutine_type>          name_subroutine_map;
    typedef boost::unordered_map<std::string, storage_buffer_type>      name_storage_buffer_map;

public:
    virtual ~program();

    unsigned                    program_id() const;
    const std::string&          info_log() const;

    template<typename T> void   uniform(const std::string& name, const T& v) const;
    template<typename T> void   uniform(const std::string& name, int i, const T& v) const;

    uniform_ptr                 uniform_raw(const std::string& name) const;

    uniform_sampler_ptr         uniform_sampler(const std::string& name) const;
    uniform_image_ptr           uniform_image(const std::string& name) const;

    uniform_1f_ptr              uniform_1f(const std::string& name) const;
    uniform_vec2f_ptr           uniform_vec2f(const std::string& name) const;
    uniform_vec3f_ptr           uniform_vec3f(const std::string& name) const;
    uniform_vec4f_ptr           uniform_vec4f(const std::string& name) const;
    uniform_mat2f_ptr           uniform_mat2f(const std::string& name) const;
    uniform_mat3f_ptr           uniform_mat3f(const std::string& name) const;
    uniform_mat4f_ptr           uniform_mat4f(const std::string& name) const;

    uniform_1i_ptr              uniform_1i(const std::string& name) const;
    uniform_vec2i_ptr           uniform_vec2i(const std::string& name) const;
    uniform_vec3i_ptr           uniform_vec3i(const std::string& name) const;
    uniform_vec4i_ptr           uniform_vec4i(const std::string& name) const;

    uniform_1ui_ptr             uniform_1ui(const std::string& name) const;
    uniform_vec2ui_ptr          uniform_vec2ui(const std::string& name) const;
    uniform_vec3ui_ptr          uniform_vec3ui(const std::string& name) const;
    uniform_vec4ui_ptr          uniform_vec4ui(const std::string& name) const;

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    uniform_1d_ptr              uniform_1d(const std::string& name) const;
    uniform_vec2d_ptr           uniform_vec2d(const std::string& name) const;
    uniform_vec3d_ptr           uniform_vec3d(const std::string& name) const;
    uniform_vec4d_ptr           uniform_vec4d(const std::string& name) const;
    uniform_mat2d_ptr           uniform_mat2d(const std::string& name) const;
    uniform_mat3d_ptr           uniform_mat3d(const std::string& name) const;
    uniform_mat4d_ptr           uniform_mat4d(const std::string& name) const;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

    void                        uniform_sampler(const std::string& name, scm::int32 u);
    void                        uniform_sampler_handle(const std::string& name, scm::uint64 h);

    void                        uniform_image(const std::string& name, scm::int32 u);
    void                        uniform_image_handle(const std::string& name, scm::uint64 h);

    void                        uniform_buffer(const std::string& name, const unsigned binding);
    void                        uniform_subroutine(const shader_stage stage, const std::string& name, const std::string& routine);

    void                        storage_buffer(const std::string& name, const unsigned binding);

    int                         attribute_location(const std::string& name) const;

    bool                        rasterization_discard() const;

protected:
    program(render_device&              in_device,
            const shader_list&          in_shaders,
            const stream_capture_array& in_capture,
            bool                        in_rasterization_discard = false,
            const named_location_list&  in_attribute_locations = named_location_list(),
            const named_location_list&  in_fragment_locations  = named_location_list());

    bool                        link(render_device& ren_dev);
    bool                        validate(render_context& ren_ctx);
    
    void                        bind(render_context& ren_ctx) const;
    void                        bind_uniforms(render_context& ren_ctx) const;

    bool                        apply_transform_feedback_varyings(render_device& in_device, const stream_capture_array& in_capture); 

    void                        retrieve_attribute_information(render_device& in_device);
    void                        retrieve_fragdata_information(render_device& in_device);
    void                        retrieve_uniform_information(render_device& in_device);

protected:
    shader_list                 _shaders;

    bool                        _rasterization_discard;

    name_uniform_map            _uniforms;
    name_uniform_block_map      _uniform_blocks;
    name_variable_map           _attributes;
    name_location_map           _samplers;
    name_subroutine_uniform_map _subroutine_uniforms[SHADER_STAGE_COUNT];
    name_subroutine_map         _subroutines[SHADER_STAGE_COUNT];
    name_storage_buffer_map     _storage_buffers;

    unsigned                    _gl_program_obj;
    std::string                 _info_log;

    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class program

} // namespace gl
} // namespace scm

#include "program.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_PROGRAM_H_INCLUDED
