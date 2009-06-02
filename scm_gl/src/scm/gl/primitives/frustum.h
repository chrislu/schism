
#ifndef SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED
#define SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED

#include <vector>

#include <scm/core/math/math.h>

#include <scm/gl/primitives/plane.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) frustum
{
    typedef scm::math::mat4f    mat4_type;

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
    frustum(const mat4_type& mvp_matrix = mat4_type::identity());
    frustum(const frustum& f);

    frustum&            operator=(const frustum& rhs);
    void                swap(frustum& rhs);

    void                update(const mat4_type& mvp_matrix);
    const plane&        get_plane(unsigned int p) const;

    void                transform(const mat4_type& t);
    void                transform_preinverted(const mat4_type& t);

protected:
    std::vector<plane> _planes;

}; // class frustum

} // namespace gl
} // namespace scm

namespace std {

template<>
inline void swap(scm::gl::frustum& lhs,
                 scm::gl::frustum& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_PRIMITIVES_FRUSTUM_H_INCLUDED
