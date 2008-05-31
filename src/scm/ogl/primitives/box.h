
#ifndef SCM_OGL_PRIMITIVES_BOX_H_INCLUDED
#define SCM_OGL_PRIMITIVES_BOX_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) box
{
public:
    box(const scm::math::vec3f& min_vert = scm::math::vec3f(0.0f, 0.0f, 0.0f),
        const scm::math::vec3f& max_vert = scm::math::vec3f(1.0f, 1.0f, 1.0f));
    /*virtual*/ ~box();

    const scm::math::vec3f&     min_vertex() const;
    const scm::math::vec3f&     max_vertex() const;
    const scm::math::vec3f      centroid() const;

    void                        min_vertex(const scm::math::vec3f& vert);
    void                        max_vertex(const scm::math::vec3f& vert);

protected:

private:
    scm::math::vec3f            _min_vertex;
    scm::math::vec3f            _max_vertex;

}; //class box

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_OGL_PRIMITIVES_BOX_H_INCLUDED
