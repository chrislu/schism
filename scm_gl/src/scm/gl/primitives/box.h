
#ifndef SCM_OGL_PRIMITIVES_BOX_H_INCLUDED
#define SCM_OGL_PRIMITIVES_BOX_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) box
{
typedef scm::math::vec3f    vec3_type;

public:
    box(const vec3_type& min_vert = vec3_type(0.0f),
        const vec3_type& max_vert = vec3_type(1.0f));
    box(const box& b);
    virtual ~box();

    box&                        operator=(const box& b);
    void                        swap(box& b);

    const vec3_type&            min_vertex() const;
    const vec3_type&            max_vertex() const;
    const vec3_type             center() const;
    const vec3_type             corner(unsigned ind) const;

    void                        min_vertex(const vec3_type& vert);
    void                        max_vertex(const vec3_type& vert);

protected:
    vec3_type                   _min_vertex;
    vec3_type                   _max_vertex;

}; //class box

} // namespace gl
} // namespace scm

namespace std {

template<>
inline void swap(scm::gl::box& lhs,
                 scm::gl::box& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_OGL_PRIMITIVES_BOX_H_INCLUDED
