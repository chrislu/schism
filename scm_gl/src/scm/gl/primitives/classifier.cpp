
#include "classifier.h"

#include <scm/gl/primitives/box.h>
#include <scm/gl/primitives/frustum.h>
#include <scm/gl/primitives/plane.h>

namespace scm {
namespace gl {

/*static*/
box_classifier::classification_type
box_classifier::classify(const scm::math::vec3f& p,
                         const box&              b)
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   b.min_vertex()[i] > p[i]
            || b.max_vertex()[i] < p[i]) {
            return (box_classifier::outside);
        }
    }

    return (box_classifier::inside);
}

box_classifier::classification_type
box_classifier::classify(const box& a,
                         const box& b)
{
    for (unsigned i = 0; i < 3; ++i) {
        if (   a.min_vertex()[i] > b.max_vertex()[i]
            || a.max_vertex()[i] < b.min_vertex()[i]) {
            return (box_classifier::outside);
        }
    }

    return (box_classifier::intersect);
}

/*static*/
plane_classifier::classification_type
plane_classifier::classify(const box&     b,
                           const plane&   p)
{
    using namespace scm::math;

    if (p.distance(b.corner(p.n_corner())) > 0.0f) {
        return (plane_classifier::front);
    }
    else if (p.distance(b.corner(p.p_corner())) > 0.0f) {
        return (plane_classifier::intersect);
    }
    else {
        return (plane_classifier::back);
    }
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
