
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_UNIFORM_H_INCLUDED
#define SCM_GL_CORE_UNIFORM_H_INCLUDED

#include <string>
#include <vector>

#include <boost/call_traits.hpp>

#include <scm/gl_core/config.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) uniform_base
{
public:
    uniform_base(const std::string& n, const int l, const unsigned e, const data_type t);
    virtual ~uniform_base();

    const std::string&      name() const;
    const int               location() const;
    const unsigned          elements() const;
    const data_type         type() const;

    bool                    update_required() const;
    virtual void            apply_value(const render_context& context, const program& p) = 0;

protected:
    std::string             _name;
    int                     _location;
    unsigned                _elements;
    data_type               _type;

    struct {
        bool                _update_required : 1;
        bool                _initialized     : 1;
    }                       _status;

private:
    // declared, never defined
    uniform_base(const uniform_base&);
    const uniform_base& operator=(const uniform_base&);

    friend class scm::gl::program;
}; // class uniform_base

template<typename T, data_type D>
class uniform : public uniform_base
{
public:
    typedef T       value_type;

protected:
    typedef typename boost::call_traits<value_type>::param_type value_param_type;
    typedef std::vector<value_type>                             value_array;

public:
    uniform(const std::string& n, const int l, const unsigned e, const data_type t);
    virtual ~uniform();

    value_param_type        value() const;
    void                    set_value(value_param_type v);

    value_param_type        value(int i) const;
    void                    set_value(int i, value_param_type v);
    
    void                    apply_value(const render_context& context, const program& p);

protected:
    value_array             _value;

}; // class uniform

template<typename T>
struct uniform_type {
};

template<typename T>
struct uniform_data_type {
};

#define SCM_UNIFORM_TYPE_DECLARE(type_raw, type_id, uniform_type_name)                                  \
    typedef scm::gl::uniform<type_raw, type_id>          uniform_type_name;                             \
    typedef shared_ptr<scm::gl::uniform_type_name>       uniform_type_name##_ptr;                       \
    typedef shared_ptr<scm::gl::uniform_type_name const> uniform_type_name##_cptr;                      \
    template<> struct uniform_type<type_raw>      { typedef uniform_type_name  type; };                 \
    template<> struct uniform_data_type<type_raw> { static const data_type  type = type_id; };

SCM_UNIFORM_TYPE_DECLARE(float,             TYPE_FLOAT,  uniform_1f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec2f,  TYPE_VEC2F,  uniform_vec2f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec3f,  TYPE_VEC3F,  uniform_vec3f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec4f,  TYPE_VEC4F,  uniform_vec4f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat2f,  TYPE_MAT2F,  uniform_mat2f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat3f,  TYPE_MAT3F,  uniform_mat3f)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat4f,  TYPE_MAT4F,  uniform_mat4f)

SCM_UNIFORM_TYPE_DECLARE(int,               TYPE_INT,    uniform_1i)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec2i,  TYPE_VEC2I,  uniform_vec2i)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec3i,  TYPE_VEC3I,  uniform_vec3i)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec4i,  TYPE_VEC4I,  uniform_vec4i)

SCM_UNIFORM_TYPE_DECLARE(unsigned,          TYPE_UINT,   uniform_1ui)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec2ui, TYPE_VEC2UI, uniform_vec2ui)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec3ui, TYPE_VEC3UI, uniform_vec3ui)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec4ui, TYPE_VEC4UI, uniform_vec4ui)

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
SCM_UNIFORM_TYPE_DECLARE(double,            TYPE_DOUBLE, uniform_1d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec2d,  TYPE_VEC2D,  uniform_vec2d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec3d,  TYPE_VEC3D,  uniform_vec3d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::vec4d,  TYPE_VEC4D,  uniform_vec4d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat2d,  TYPE_MAT2D,  uniform_mat2d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat3d,  TYPE_MAT3D,  uniform_mat3d)
SCM_UNIFORM_TYPE_DECLARE(scm::math::mat4d,  TYPE_MAT4D,  uniform_mat4d)
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

//SCM_UNIFORM_TYPE_DECLARE(scm::uint64,       TYPE_SAMPLER, uniform_sampler)
//SCM_UNIFORM_TYPE_DECLARE(scm::uint64,       TYPE_IMAGE,   uniform_image)

// convenience
//typedef uniform_1i                                      uniform_sampler;
//typedef shared_ptr<scm::gl::uniform_sampler>            uniform_sampler_ptr;
//typedef shared_ptr<scm::gl::uniform_sampler const>      uniform_sampler_cptr;
//
//typedef uniform_1i                                      uniform_image;
//typedef shared_ptr<scm::gl::uniform_image>              uniform_image_ptr;
//typedef shared_ptr<scm::gl::uniform_image const>        uniform_image_cptr;

template<> struct uniform_type<bool>      { typedef uniform_1i  type; };
template<> struct uniform_data_type<bool> { static const data_type  type = TYPE_INT; };

class uniform_image_sampler_base : public uniform_base
{
public:
    uniform_image_sampler_base(const std::string& n, const int l, const unsigned e, const data_type t);
    virtual ~uniform_image_sampler_base();

    scm::int32              bound_unit() const;
    void                    bound_unit(scm::int32 v);

    scm::uint64             resident_handle();
    void                    resident_handle(scm::uint64 v);
    
    void                    apply_value(const render_context& context, const program& p);

protected:
    scm::int32              _bound_unit;
    scm::uint64             _resident_handle;

}; // class uniform_image_sampler_base

typedef uniform_image_sampler_base  uniform_sampler;
typedef shared_ptr<scm::gl::uniform_sampler>            uniform_sampler_ptr;
typedef shared_ptr<scm::gl::uniform_sampler const>      uniform_sampler_cptr;

typedef uniform_image_sampler_base  uniform_image;
typedef shared_ptr<scm::gl::uniform_image>              uniform_image_ptr;
typedef shared_ptr<scm::gl::uniform_image const>        uniform_image_cptr;

#undef SCM_UNIFORM_TYPE_DECLARE

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_UNIFORM_H_INCLUDED
