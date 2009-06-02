
#include "aabbox.h"
#if 0

#include <scm/core/math/math.h>

#include <scm/gl/primitives/plane.h>

namespace scm {
namespace gl {

aabbox::aabbox(const scm::math::vec3f& min_vert,
               const scm::math::vec3f& max_vert)
  : box(min_vert, max_vert)
{
}

aabbox::~aabbox()
{
}

const scm::math::vec3f
aabbox::n_vertex(const scm::math::vec3f& n) const
{
    scm::math::vec3f n_vert((n.x < 0.0f) ? _max_vertex.x : _min_vertex.x,
                            (n.y < 0.0f) ? _max_vertex.y : _min_vertex.y,
                            (n.z < 0.0f) ? _max_vertex.z : _min_vertex.z);
    
    return (n_vert); 
}

const scm::math::vec3f
aabbox::p_vertex(const scm::math::vec3f& n) const
{
    scm::math::vec3f p_vert((n.x < 0.0f) ? _min_vertex.x : _max_vertex.x,
                            (n.y < 0.0f) ? _min_vertex.y : _max_vertex.y,
                            (n.z < 0.0f) ? _min_vertex.z : _max_vertex.z);

    return (p_vert);
}

aabbox::plane_classification_type
aabbox::classify(const plane& p) const
{
    using namespace scm::math;

    if (dot(n_vertex(p.normal()), p.normal()) > -p.distance()) {
        return (aabbox::front);
    }
    else if (dot(p_vertex(p.normal()), p.normal()) > -p.distance()) {
        return (aabbox::intersect);
    }
    else {
        return (aabbox::back);
    }
}

} // namespace gl
} // namespace scm
#endif / 0

