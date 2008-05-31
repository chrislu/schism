
#ifndef SCM_OGL_PRIMITIVES_CLASSIFIER_H_INCLUDED
#define SCM_OGL_PRIMITIVES_CLASSIFIER_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class box;
class frustum;
class plane;

class __scm_export(ogl) plane_classifier
{
public:
    typedef enum {
        front,
        back,
        intersect
    } classification_type;

    static classification_type      classify(const box&     b,
                                             const plane&   p);

    // helper
    static const scm::math::vec3f   n_vertex(const box&     b,
                                             const plane&   p);
    static const scm::math::vec3f   p_vertex(const box&     b,
                                             const plane&   p);
}; // class plane_classifier

class __scm_export(ogl) frustum_classifier
{
public:
    typedef enum {
        inside,
        outside,
        intersect
    } classification_type;

    static classification_type      classify(const box&     b,
                                             const frustum& f);

}; // class frustum_classifier

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_CLASSIFIER_H_INCLUDED
