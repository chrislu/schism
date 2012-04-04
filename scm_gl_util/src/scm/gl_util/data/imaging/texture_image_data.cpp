
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_image_data.h"

#include <cassert>
#include <vector>
#include <memory.h>

#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/gl_util/data/imaging/texture_data_util.h>

namespace scm {
namespace gl {

texture_image_data::texture_image_data(const data_origin   img_origin,
                                       const data_format   img_format,
                                       const level_vector& img_mip_data)
  : _origin(img_origin)
  , _format(img_format)
  , _mip_levels(img_mip_data)
  , _layers(1)
{
}

texture_image_data::texture_image_data(const data_origin   img_origin,
                                       const data_format   img_format,
                                       const int           layers,
                                       const level_vector& img_mip_data)
  : _origin(img_origin)
  , _format(img_format)
  , _mip_levels(img_mip_data)
  , _layers(math::max(1, layers))
{
}

texture_image_data::~texture_image_data()
{
    _mip_levels.clear();
}

const texture_image_data::data_origin
texture_image_data::origin() const
{
    return _origin;
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

int
texture_image_data::array_layers() const
{
    return _layers;
}

bool
texture_image_data::flip_vertical()
{
    unsigned img_mip_count = mip_level_count();
    for (unsigned l = 0; l < img_mip_count; ++l) {
        if (!util::volume_flip_vertical(mip_level(l).data(), format(), mip_level(l).size().x, mip_level(l).size().y, mip_level(l).size().z)) {
            return false;
        }
    }

    if (origin() == ORIGIN_LOWER_LEFT) {
        _origin = ORIGIN_UPPER_LEFT;
    }
    else {
        _origin = ORIGIN_LOWER_LEFT;
    }

    return true;
}

} // namespace gl
} // namespace scm
