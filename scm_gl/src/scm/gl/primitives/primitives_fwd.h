
#ifndef SCM_GL_PRIMITIVES_FWD_H_INCLUDED
#define SCM_GL_PRIMITIVES_FWD_H_INCLUDED

namespace scm {
namespace gl {

template<typename s> class box_impl;
template<typename s> class frustum_impl;
template<typename s> class plane_impl;
template<typename s> class ray_impl;

typedef box_impl<float>         boxf;
typedef box_impl<double>        boxd;

typedef frustum_impl<float>     frustumf;
typedef frustum_impl<double>    frustumd;

typedef plane_impl<float>       planef;
typedef plane_impl<double>      planed;

typedef ray_impl<float>         rayf;
typedef ray_impl<double>        rayd;

// default
typedef box_impl<float>         box;
typedef frustum_impl<float>     frustum;
typedef plane_impl<float>       plane;
typedef ray_impl<float>         ray;

} // namespace gl
} // namespace scm

#endif // SCM_GL_PRIMITIVES_FWD_H_INCLUDED
