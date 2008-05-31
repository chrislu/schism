
#ifndef SCM_OGL_PRIMITIVES_AABBOX_H_INCLUDED
#define SCM_OGL_PRIMITIVES_AABBOX_H_INCLUDED

#include <limits>

#include <scm/ogl/primitives/box.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class plane;

class __scm_export(ogl) aabbox : public scm::gl::box
{
public:
    typedef enum {
        front,
        back,
        intersect
    } plane_classification_type;

public:
    aabbox(const scm::math::vec3f& min_vert = scm::math::vec3f((std::numeric_limits<scm::math::vec3f::value_type>::max)()),
           const scm::math::vec3f& max_vert = scm::math::vec3f((std::numeric_limits<scm::math::vec3f::value_type>::min)()));
    virtual ~aabbox();

    // negative and positive far points
    const scm::math::vec3f          n_vertex(const scm::math::vec3f& n) const;
    const scm::math::vec3f          p_vertex(const scm::math::vec3f& n) const;

    plane_classification_type       classify(const plane& p) const;

}; // class aabbox

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_AABBOX_H_INCLUDED
