
#ifndef SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED
#define SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/ogl/primitives/plane.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) frustum
{
public:
    typedef enum {
        left_plane        = 0x00,
        right_plane       = 0x01,
        top_plane         = 0x02,
        bottom_plane      = 0x03,
        near_plane        = 0x04,
        far_plane         = 0x05
    } plane_identifier;

public:
    frustum(const scm::math::mat4f& mvp_matrix = scm::math::mat4f::identity());

    void                update(const scm::math::mat4f& mvp_matrix);
    const plane&        get_plane(unsigned int p) const;

protected:
    plane               _planes[6];

}; // class frustum

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED
