
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_DATA_SCM_VOL_H_INCLUDED
#define SCM_DATA_SCM_VOL_H_INCLUDED

#include <istream>
#include <ostream>
#include <string>

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class __scm_export(data) volume_file_descriptor
{
public:
    volume_file_descriptor();
    virtual ~volume_file_descriptor();

    volume_file_descriptor(const volume_file_descriptor& /*rhs*/);
    const volume_file_descriptor& operator=(const volume_file_descriptor&  /*rhs*/);


    scm::math::vec3ui   _data_dimensions;
    unsigned            _data_num_channels;
    unsigned            _data_byte_per_channel;

    scm::math::vec3f    _volume_origin;
    scm::math::vec3f    _volume_aspect;

    scm::math::vec3ui   _brick_index;

    std::string         _sraw_file;
    std::string         _name;

    const unsigned      _version;
}; // class volume_descriptor

__scm_export(data) std::ostream& operator<<(std::ostream& /*os*/, const volume_file_descriptor& /*desc*/);
__scm_export(data) std::istream& operator>>(std::istream& /*i*/, volume_file_descriptor& /*desc*/);

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_DATA_SCM_VOL_H_INCLUDED
