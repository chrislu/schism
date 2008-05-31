
#ifndef SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED
#define SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) plane
{
public:
    plane();
    plane(const plane& p);
    explicit plane(const scm::math::vec4f& p);

    plane&                          operator=(const plane& rhs);

    const scm::math::vec3f&         normal() const;
    scm::math::vec3f::value_type    distance() const;
    const scm::math::vec4f&         vector() const;

protected:
    union {
        struct {
            scm::math::vec3f                _normal;
            scm::math::vec3f::value_type    _distance;
        };
        // to try on gcc, do not know if he eats this
        struct {
            scm::math::vec4f                    _vector;
        };
    };

}; // class frustum

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_PLANE_H_INCLUDED
