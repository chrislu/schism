
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_PRIMITIVES_FWD_H_INCLUDED
#define SCM_GL_CORE_PRIMITIVES_FWD_H_INCLUDED

namespace scm {
namespace gl {

template<typename s> class box_impl;
template<typename s> class frustum_impl;
template<typename s> class plane_impl;
template<typename s> class rect_impl;
template<typename s> class ray_impl;

typedef box_impl<float>         boxf;
typedef box_impl<double>        boxd;

typedef frustum_impl<float>     frustumf;
typedef frustum_impl<double>    frustumd;

typedef plane_impl<float>       planef;
typedef plane_impl<double>      planed;

typedef ray_impl<float>         rayf;
typedef ray_impl<double>        rayd;

typedef rect_impl<float>        rectf;
typedef rect_impl<double>       rectd;

// default
typedef box_impl<float>         box;
typedef frustum_impl<float>     frustum;
typedef plane_impl<float>       plane;
typedef ray_impl<float>         ray;
typedef rect_impl<float>        rect;

template<typename scal_type>
struct epsilon
{
};

template<>
struct epsilon<float>
{
    static float value() { return (1.0e-4f); }
};

template<>
struct epsilon<double>
{
    static double value() { return (1.0e-6); }
};

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_PRIMITIVES_FWD_H_INCLUDED
