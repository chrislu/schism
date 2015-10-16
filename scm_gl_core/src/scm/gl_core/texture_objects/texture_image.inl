
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.


namespace scm {
namespace gl {
namespace util {

inline
unsigned
max_mip_levels(const unsigned      in_tex_size)
{
    using namespace scm::math;

    double max_size   = in_tex_size;
    double log2_msize = math::log(max_size) / math::log(2.0);

    return static_cast<unsigned>(floor(log2_msize)) + 1;
}

inline
unsigned
max_mip_levels(const math::vec2ui& in_tex_size)
{
    using namespace scm::math;

    double max_size   = max(in_tex_size.x, in_tex_size.y);
    double log2_msize = math::log(max_size) / math::log(2.0);

    return static_cast<unsigned>(floor(log2_msize)) + 1;
}

inline
unsigned
max_mip_levels(const math::vec3ui& in_tex_size)
{
    using namespace scm::math;

    double max_size   = max(max(in_tex_size.x, in_tex_size.y), in_tex_size.z);
    double log2_msize = math::log(max_size) / math::log(2.0);

    return static_cast<unsigned>(floor(log2_msize)) + 1;
}

inline
unsigned
mip_level_dimensions(const unsigned      in_tex_size,
                     unsigned            in_level)
{
    using namespace scm::math;

    double pow2i = pow(2.0, static_cast<double>(in_level));
    double lsize = double(in_tex_size) / pow2i;

    unsigned ret_value;
    ret_value = max(1u, static_cast<unsigned>(floor(lsize)));

    return ret_value;
}


inline
math::vec2ui
mip_level_dimensions(const math::vec2ui& in_tex_size,
                     unsigned            in_level)
{
    using namespace scm::math;

    double pow2i = pow(2.0, static_cast<double>(in_level));
    vec2d  lsize = vec2d(in_tex_size) / pow2i;

    vec2ui ret_value;
    ret_value.x = max(1u, static_cast<unsigned>(floor(lsize.x)));
    ret_value.y = max(1u, static_cast<unsigned>(floor(lsize.y)));

    return ret_value;
}

inline
math::vec3ui
mip_level_dimensions(const math::vec3ui& in_tex_size,
                     unsigned            in_level)
{
    using namespace scm::math;

    double pow2i = pow(2.0, static_cast<double>(in_level));
    vec3d  lsize = vec3d(in_tex_size) / pow2i;

    vec3ui ret_value;
    ret_value.x = max(1u, static_cast<unsigned>(floor(lsize.x)));
    ret_value.y = max(1u, static_cast<unsigned>(floor(lsize.y)));
    ret_value.z = max(1u, static_cast<unsigned>(floor(lsize.z)));

    return ret_value;
}
} // namespace util
} // namespace gl
} // namespace scm
