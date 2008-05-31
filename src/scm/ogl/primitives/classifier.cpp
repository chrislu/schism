
#include "classifier.h"

#include <scm/ogl/primitives/box.h>
#include <scm/ogl/primitives/frustum.h>
#include <scm/ogl/primitives/plane.h>

namespace scm {
namespace gl {

/*static*/
plane_classifier::classification_type
plane_classifier::classify(const box&     b,
                           const plane&   p)
{
    using namespace scm::math;

    if ((dot(n_vertex(b, p), p.normal()) + p.distance()) > 0.0f) {
        return (plane_classifier::front);
    }
    else if ((dot(p_vertex(b, p), p.normal()) + p.distance()) > 0.0f) {
        return (plane_classifier::intersect);
    }
    else {
        return (plane_classifier::back);
    }
}

/*static*/
const scm::math::vec3f
plane_classifier::n_vertex(const box&     b,
                           const plane&   p)
{
    scm::math::vec3f n_vert((p.normal().x < 0.0f) ? b.max_vertex().x : b.min_vertex().x,
                            (p.normal().y < 0.0f) ? b.max_vertex().y : b.min_vertex().y,
                            (p.normal().z < 0.0f) ? b.max_vertex().z : b.min_vertex().z);
    
    return (n_vert); 
}

/*static*/
const scm::math::vec3f
plane_classifier::p_vertex(const box&     b,
                           const plane&   p)
{
    scm::math::vec3f p_vert((p.normal().x < 0.0f) ? b.min_vertex().x : b.max_vertex().x,
                            (p.normal().y < 0.0f) ? b.min_vertex().y : b.max_vertex().y,
                            (p.normal().z < 0.0f) ? b.min_vertex().z : b.max_vertex().z);

    return (p_vert);
}

/*static*/
frustum_classifier::classification_type
frustum_classifier::classify(const box&     b,
                             const frustum& f)
{
    bool plane_intersect = false;

    // normals of frustum planes point inside the frustum
    for (unsigned i = 0; i < 6; ++i) {
        plane_classifier::classification_type  cur_class = plane_classifier::classify(b, f.get_plane(i));

        if (cur_class == plane_classifier::back) {
            return (frustum_classifier::outside);
        }
        else if (cur_class == plane_classifier::intersect) {
            plane_intersect = true;
        }
    }

    return (plane_intersect ? frustum_classifier::intersect : frustum_classifier::inside);
}

} // namespace gl
} // namespace scm
