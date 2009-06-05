
#ifndef SCM_GL_PRIMITIVES_RAY_H_INCLUDED
#define SCM_GL_PRIMITIVES_RAY_H_INCLUDED

#include <scm/core/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) ray
{
public:
    typedef scm::math::vec4f    vec4_type;
    typedef scm::math::vec3f    vec3_type;
    typedef scm::math::mat4f    mat4_type;

public:
    ray();
    ray(const ray& p);
    explicit ray(const vec3_type& org,
                 const vec3_type& dir);

    ray&                    operator=(const ray& rhs);
    void                    swap(ray& rhs);

    void                    transform(const mat4_type& t);

    const vec3_type&        origin() const;
    const vec3_type&        direction() const;

protected:
    void                    normalize();

protected:
    vec3_type               _origin;
    vec3_type               _direction;

}; // class ray

} // namespace gl
} // namespace scm

namespace std {

template<>
inline void swap(scm::gl::ray& lhs,
                 scm::gl::ray& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_PRIMITIVES_RAY_H_INCLUDED
