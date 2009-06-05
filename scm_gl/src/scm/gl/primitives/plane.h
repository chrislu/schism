
#ifndef SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED
#define SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class box;

// describing plane equation
//  - a*x + b*y + c*z + d = 0
//  - so d gives you the distance of the origin related to the plane!
class __scm_export(ogl) plane
{
public:
    typedef scm::math::vec4f    vec4_type;
    typedef scm::math::vec3f    vec3_type;
    typedef scm::math::mat4f    mat4_type;

    typedef enum {
        front,
        back,
        intersect
    } classification_result;

public:
    plane();
    plane(const plane& p);
    explicit plane(const vec4_type& p);

    plane&                  operator=(const plane& rhs);
    void                    swap(plane& rhs);

    const vec3_type         normal() const;
    vec3_type::value_type   distance(const vec3_type& p) const;

    const vec4_type&        vector() const;

    void                    transform(const mat4_type& t);
    void                    transform_preinverted(const mat4_type& t);
    void                    transform_preinverted_transposed(const mat4_type& t);

    unsigned                p_corner() const;
    unsigned                n_corner() const;

    classification_result   classify(const box& b) const;

protected:
    void                    normalize();
    void                    update_corner_indices();

protected:
    vec4_type               _vector;

    unsigned                _p_corner;
    unsigned                _n_corner;

}; // class frustum

} // namespace gl
} // namespace scm

namespace std {

template<>
inline void swap(scm::gl::plane& lhs,
                 scm::gl::plane& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED
