
#ifndef SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED
#define SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED

#include <boost/mpl/vector.hpp>
#include <boost/variant.hpp>

#include <scm/core/math.h>

#include <scm/gl_core/data_types.h>

namespace scm {
namespace gl {

typedef boost::mpl::vector<float,    scm::math::vec2f,  scm::math::vec3f,  scm::math::vec4f, scm::math::mat2f, scm::math::mat3f, scm::math::mat4f,
                           int,      scm::math::vec2i,  scm::math::vec3i,  scm::math::vec4i,
                           unsigned, scm::math::vec2ui, scm::math::vec3ui, scm::math::vec4ui> uniform_types;

template<typename utype> struct uniform_type_id {};
template<> struct uniform_type_id<float>            { static const data_type id = TYPE_FLOAT; };
template<> struct uniform_type_id<scm::math::vec2f> { static const data_type id = TYPE_VEC2F; };
template<> struct uniform_type_id<scm::math::vec3f> { static const data_type id = TYPE_VEC3F; };
template<> struct uniform_type_id<scm::math::vec4f> { static const data_type id = TYPE_VEC4F; };
template<> struct uniform_type_id<scm::math::mat2f> { static const data_type id = TYPE_MAT2F; };
template<> struct uniform_type_id<scm::math::mat3f> { static const data_type id = TYPE_MAT3F; };
template<> struct uniform_type_id<scm::math::mat4f> { static const data_type id = TYPE_MAT4F; };

template<> struct uniform_type_id<int>              { static const data_type id = TYPE_INT; };
template<> struct uniform_type_id<scm::math::vec2i> { static const data_type id = TYPE_VEC2I; };
template<> struct uniform_type_id<scm::math::vec3i> { static const data_type id = TYPE_VEC3I; };
template<> struct uniform_type_id<scm::math::vec4i> { static const data_type id = TYPE_VEC4I; };

template<> struct uniform_type_id<unsigned>          { static const data_type id = TYPE_UINT; };
template<> struct uniform_type_id<scm::math::vec2ui> { static const data_type id = TYPE_VEC2UI; };
template<> struct uniform_type_id<scm::math::vec3ui> { static const data_type id = TYPE_VEC3UI; };
template<> struct uniform_type_id<scm::math::vec4ui> { static const data_type id = TYPE_VEC4UI; };

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_PROGRAM_UNIFORM_TYPES_H_INCLUDED
