
#include "texture_image_data.h"

#include <cassert>
#include <vector>
#include <memory.h>

#include <scm/core/math.h>
#include <scm/core/memory.h>

namespace scm {
namespace gl {

texture_image_data::texture_image_data(const data_format   img_format,
                                       const level_vector& img_mip_data)
  : _format(img_format)
  , _mip_levels(img_mip_data)
{
}

texture_image_data::~texture_image_data()
{
    _mip_levels.clear();
}

const data_format
texture_image_data::format() const
{
    return _format;
}

const texture_image_data::level&
texture_image_data::mip_level(const int i) const
{
    assert(i < mip_level_count());

    return _mip_levels[i];
}

int
texture_image_data::mip_level_count() const
{
    return static_cast<int>(_mip_levels.size());
}

} // namespace gl
} // namespace scm
