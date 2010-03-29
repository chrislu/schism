
#ifndef SCM_GL_CORE_PROGRAM_H_INCLUDED
#define SCM_GL_CORE_PROGRAM_H_INCLUDED

#include <list>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/variant.hpp>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/data_types.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device_child.h>
#include <scm/gl_core/program_uniform_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) program : public render_device_child
{
public:
    typedef int location_type;
    struct variable_type {
        variable_type() : _location(-1), _type(TYPE_UNKNOWN), _elements(0) {}
        variable_type(const std::string& n, location_type l, unsigned e, data_type t) : _name(n), _location(l), _type(t), _elements(e) {}
        std::string                 _name;
        location_type               _location;
        unsigned                    _elements;
        data_type                   _type;
    };
    struct uniform_type : public variable_type {
        uniform_type() : /*_data_array_size(0),*/ _update_required(true) {}
        uniform_type(const std::string& n, location_type l, unsigned e, data_type t) : variable_type(n, l, e, t), _update_required(true) {}
        //uniform_type(const std::string& n, location_type l, unsigned e, data_type t) : variable_type(n, l, e, t), _data_array_size(size_of_type(t) * e), _update_required(true) { _data_array.reset(new uint8[_data_array_size]); }
        //scm::shared_array<uint8>    _data_array;
        //scm::size_t                 _data_array_size;
        boost::make_variant_over<uniform_types>::type _data;
        bool                        _update_required;
    };

    typedef boost::unordered_map<std::string, variable_type>    name_variable_map;
    typedef boost::unordered_map<std::string, uniform_type>     name_uniform_map;
    typedef boost::unordered_map<std::string, location_type>    name_location_map;
    typedef std::list<const shader_ptr>                         shader_list;

    typedef std::pair<std::string, unsigned>                    named_location;
    typedef std::list<named_location>                           named_location_list;

public:
    virtual ~program();

    const std::string&          info_log() const;

    template<typename utype>
    void                        uniform(const std::string& name, const utype& value);
    //void                        uniform(const std::string& name, float value);
    //void                        uniform(const std::string& name, const math::vec2f& value);
    //void                        uniform(const std::string& name, const math::vec3f& value);
    //void                        uniform(const std::string& name, const math::vec4f& value);

    //void                        uniform(const std::string& name, int value);
    //void                        uniform(const std::string& name, const math::vec2i& value);
    //void                        uniform(const std::string& name, const math::vec3i& value);
    //void                        uniform(const std::string& name, const math::vec4i& value);

    //void                        uniform(const std::string& name, unsigned value);
    //void                        uniform(const std::string& name, const math::vec2ui& value);
    //void                        uniform(const std::string& name, const math::vec3ui& value);
    //void                        uniform(const std::string& name, const math::vec4ui& value);

    //void                        uniform(const std::string& name, const math::mat2f& value, bool transpose = false);
    //void                        uniform(const std::string& name, const math::mat3f& value, bool transpose = false);
    //void                        uniform(const std::string& name, const math::mat4f& value, bool transpose = false);

    //void                        uniform_raw(uniform_type& u, const void *data, const int size);

protected:
    program(render_device&              ren_dev,
            const shader_list&          in_shaders,
            const named_location_list&  in_attibute_locations = named_location_list(),
            const named_location_list&  in_fragment_locations = named_location_list());

    bool                        link(render_device& ren_dev);
    bool                        validate(render_context& ren_ctx);
    
    void                        bind(render_context& ren_ctx) const;
    void                        bind_uniforms(render_context& ren_ctx) const;

    void                        retrieve_attribute_information(render_device& ren_dev);
    void                        retrieve_fragdata_information(render_device& ren_dev);
    void                        retrieve_uniform_information(render_device& ren_dev);

protected:
    shader_list                 _shaders;

    mutable name_uniform_map            _uniforms;
    name_variable_map           _attributes;
    name_location_map           _samplers;

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
